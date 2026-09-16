# server.py

from flask import Flask, request, send_file, jsonify, Response
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
import os
import traceback
import json
import base64

from config import DEFAULT_CONFIGS
from handle_text import prepare_tts_input_with_context
from tts_handler import (
    AudioFormatError,
    AudioFormatUnavailableError,
    VoiceValidationError,
    generate_speech,
    generate_speech_stream,
    get_models_formatted,
    get_voices,
    get_voices_formatted,
    model_data,
    validate_voice,
)
from utils import api_error, getenv_bool, require_api_key, AUDIO_FORMAT_MIME_TYPES, DETAILED_ERROR_LOGGING

app = Flask(__name__)
load_dotenv()

API_KEY = os.getenv('API_KEY', DEFAULT_CONFIGS["API_KEY"])
PORT = int(os.getenv('PORT', str(DEFAULT_CONFIGS["PORT"])))

DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', DEFAULT_CONFIGS["DEFAULT_VOICE"])
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', DEFAULT_CONFIGS["DEFAULT_RESPONSE_FORMAT"])
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', str(DEFAULT_CONFIGS["DEFAULT_SPEED"])))

REMOVE_FILTER = getenv_bool('REMOVE_FILTER', DEFAULT_CONFIGS["REMOVE_FILTER"])
EXPAND_API = getenv_bool('EXPAND_API', DEFAULT_CONFIGS["EXPAND_API"])

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', DEFAULT_CONFIGS['DEFAULT_MODEL'])
AUDIO_SAMPLE_RATE = os.getenv('AUDIO_SAMPLE_RATE', str(DEFAULT_CONFIGS['AUDIO_SAMPLE_RATE']))
AUDIO_CHANNELS = os.getenv('AUDIO_CHANNELS', str(DEFAULT_CONFIGS['AUDIO_CHANNELS']))

def generate_sse_audio_stream(text, voice, speed):
    """Generate OpenAI-compatible SSE events for MP3 audio chunks."""
    try:
        for chunk in generate_speech_stream(text, voice, speed):
            encoded_audio = base64.b64encode(chunk).decode('utf-8')

            yield f"data: {json.dumps({'audio': encoded_audio})}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        app.logger.error("Error during SSE streaming: %s", e)
        error_event = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "speech_generation_failed",
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        yield "data: [DONE]\n\n"

# OpenAI endpoint format
@app.route('/v1/audio/speech', methods=['POST'])
@app.route('/audio/speech', methods=['POST'])  # Add this line for the alias
@require_api_key
def text_to_speech():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return api_error("Request body must be a JSON object.", "invalid_json")
        if 'input' not in data:
            return api_error("Missing 'input' in request body.", "missing_input")

        text = data.get('input')
        if not isinstance(text, str) or not text.strip():
            return api_error("'input' must be a non-empty string.", "invalid_input")

        if not REMOVE_FILTER:
            text = prepare_tts_input_with_context(text)

        model = data.get('model', DEFAULT_MODEL)
        if not isinstance(model, str) or not model:
            return api_error("'model' must be a non-empty string.", "invalid_model")
        supported_models = {item['id'] for item in model_data}
        if model not in supported_models:
            return api_error(f"Unsupported model '{model}'.", "unsupported_model")

        voice = data.get('voice', DEFAULT_VOICE)
        if not isinstance(voice, str) or not voice:
            return api_error("'voice' must be a non-empty string.", "invalid_voice")

        response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
        if not isinstance(response_format, str) or not response_format:
            return api_error(
                "'response_format' must be a non-empty string.",
                "invalid_response_format",
            )
        if response_format not in AUDIO_FORMAT_MIME_TYPES:
            return api_error(
                f"Unsupported response_format '{response_format}'.",
                "unsupported_response_format",
            )

        try:
            speed = float(data.get('speed', DEFAULT_SPEED))
        except (TypeError, ValueError):
            return api_error("'speed' must be a number between 0 and 2.", "invalid_speed")
        if not 0 <= speed <= 2:
            return api_error("'speed' must be between 0 and 2.", "invalid_speed")

        stream_format = data.get('stream_format', 'audio')
        if not isinstance(stream_format, str) or not stream_format:
            return api_error(
                "'stream_format' must be a non-empty string.",
                "invalid_stream_format",
            )
        if stream_format not in {'audio', 'sse'}:
            return api_error(
                f"Unsupported stream_format '{stream_format}'.",
                "unsupported_stream_format",
            )
        if stream_format == 'sse' and response_format != 'mp3':
            return api_error(
                "stream_format=sse currently supports response_format=mp3 only.",
                "unsupported_response_format",
            )

        try:
            validate_voice(voice)
        except VoiceValidationError as exc:
            return api_error(str(exc), "unsupported_voice")

        mime_type = AUDIO_FORMAT_MIME_TYPES[response_format]

        if stream_format == 'sse':
            def generate_sse():
                for event in generate_sse_audio_stream(text, voice, speed):
                    yield event

            return Response(
                generate_sse(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                    'X-Audio-Sample-Rate': AUDIO_SAMPLE_RATE,
                    'X-Audio-Channels': AUDIO_CHANNELS,
                }
            )
        else:
            try:
                output_file_path = generate_speech(text, voice, response_format, speed)
                with open(output_file_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
            except (AudioFormatError, AudioFormatUnavailableError) as exc:
                return api_error(str(exc), "unsupported_response_format")
            finally:
                if 'output_file_path' in locals():
                    try:
                        os.unlink(output_file_path)
                    except OSError:
                        pass

            return Response(
                audio_data,
                mimetype=mime_type,
                headers={
                    'Content-Type': mime_type,
                    'Content-Length': str(len(audio_data)),
                    'X-Audio-Sample-Rate': AUDIO_SAMPLE_RATE,
                    'X-Audio-Channels': AUDIO_CHANNELS,
                }
            )

    except (AudioFormatError, AudioFormatUnavailableError) as exc:
        return api_error(str(exc), "unsupported_response_format")
    except VoiceValidationError as exc:
        return api_error(str(exc), "unsupported_voice")
    except Exception as e:
        if DETAILED_ERROR_LOGGING:
            app.logger.error(f"Error in text_to_speech: {str(e)}\n{traceback.format_exc()}")
        else:
            app.logger.error(f"Error in text_to_speech: {str(e)}")
        return api_error(
            "An internal server error occurred.",
            "internal_server_error",
            500,
            "server_error",
        )

# OpenAI endpoint format
@app.route('/v1/models', methods=['GET', 'POST'])
@app.route('/models', methods=['GET', 'POST'])
@app.route('/v1/audio/models', methods=['GET', 'POST'])
@app.route('/audio/models', methods=['GET', 'POST'])
def list_models():
    return jsonify({"models": get_models_formatted()})

# OpenAI endpoint format
@app.route('/v1/audio/voices', methods=['GET', 'POST'])
@app.route('/audio/voices', methods=['GET', 'POST'])
def list_voices_formatted():
    return jsonify({"voices": get_voices_formatted()})

@app.route('/v1/voices', methods=['GET', 'POST'])
@app.route('/voices', methods=['GET', 'POST'])
@require_api_key
def list_voices():
    specific_language = None

    data = request.args if request.method == 'GET' else request.json
    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET', 'POST'])
@app.route('/voices/all', methods=['GET', 'POST'])
@require_api_key
def list_all_voices():
    return jsonify({"voices": get_voices('all')})

"""
Support for ElevenLabs and Azure AI Speech
    (currently in beta)
"""

# http://localhost:5050/elevenlabs/v1/text-to-speech
# http://localhost:5050/elevenlabs/v1/text-to-speech/en-US-AndrewNeural
@app.route('/elevenlabs/v1/text-to-speech/<voice_id>', methods=['POST'])
@require_api_key
def elevenlabs_tts(voice_id):
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the incoming JSON payload
    try:
        payload = request.json
        if not payload or 'text' not in payload:
            return jsonify({"error": "Missing 'text' in request body"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400

    text = payload['text']

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    voice = voice_id  # ElevenLabs uses the voice_id in the URL

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED  # Optional customization via payload.get('speed', DEFAULT_SPEED)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

# tts.speech.microsoft.com/cognitiveservices/v1
# https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
# http://localhost:5050/azure/cognitiveservices/v1
@app.route('/azure/cognitiveservices/v1', methods=['POST'])
@require_api_key
def azure_tts():
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the SSML payload
    try:
        ssml_data = request.data.decode('utf-8')
        if not ssml_data:
            return jsonify({"error": "Missing SSML payload"}), 400

        # Extract the text and voice from SSML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(ssml_data)
        text = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').text
        voice = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').get('name')
    except Exception as e:
        return jsonify({"error": f"Invalid SSML payload: {str(e)}"}), 400

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

print(f" Edge TTS (Free Azure TTS) Replacement for OpenAI's TTS API")
print(f" ")
print(f" * Serving OpenAI Edge TTS")
print(f" * Server running on http://localhost:{PORT}")
print(f" * TTS Endpoint: http://localhost:{PORT}/v1/audio/speech")
print(f" ")

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
