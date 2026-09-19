# tts_handler.py

import edge_tts
import asyncio
import tempfile
import subprocess
import os
from pathlib import Path
from functools import lru_cache

from utils import DETAILED_ERROR_LOGGING
from config import DEFAULT_CONFIGS


class AudioFormatError(ValueError):
    pass


class AudioFormatUnavailableError(AudioFormatError):
    pass


class VoiceValidationError(ValueError):
    pass

# Language default (environment variable)
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])

# OpenAI voice names mapped to edge-tts equivalents
voice_mapping = {
    'alloy': 'en-US-JennyNeural',
    'ash': 'en-US-AndrewNeural',
    'ballad': 'en-GB-ThomasNeural',
    'coral': 'en-AU-NatashaNeural',
    'echo': 'en-US-GuyNeural',
    'fable': 'en-GB-SoniaNeural',
    'nova': 'en-US-AriaNeural',
    'onyx': 'en-US-EricNeural',
    'sage': 'en-US-JennyNeural',
    'shimmer': 'en-US-EmmaNeural',
    'verse': 'en-US-BrianNeural',
    'marin': 'en-US-AvaNeural',
    'cedar': 'en-US-GuyNeural',
}

model_data = [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"},
        {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"},
        {"id": "gpt-4o-mini-tts-2025-12-15", "name": "GPT-4o mini TTS (2025-12-15)"}
    ]

SUPPORTED_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        ffmpeg_executable()
        return True
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError):
        return False


def ffmpeg_executable():
    """Return the host ffmpeg executable used for non-MP3 formats."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return 'ffmpeg'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError("ffmpeg is not installed")

async def _generate_audio_stream(text, voice, speed):
    """Generate streaming TTS audio using edge-tts."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is
    
    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"
    
    # Create the communicator for streaming
    communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
    
    stream_iterator = communicator.stream()
    try:
        async for chunk in stream_iterator:
            if chunk["type"] == "audio":
                yield chunk["data"]
    finally:
        await stream_iterator.aclose()

def generate_speech_stream(text, voice, speed=1.0):
    event_loop = asyncio.new_event_loop()
    async_iterator = _generate_audio_stream(text, voice, speed).__aiter__()

    try:
        while True:
            try:
                yield event_loop.run_until_complete(async_iterator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        close_method = getattr(async_iterator, "aclose", None)
        if close_method is not None:
            try:
                event_loop.run_until_complete(close_method())
            except Exception:
                pass
        try:
            event_loop.run_until_complete(event_loop.shutdown_asyncgens())
            pending_tasks = [
                task for task in asyncio.all_tasks(event_loop)
                if not task.done()
            ]
            if pending_tasks:
                event_loop.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )
        except Exception:
            pass
        event_loop.close()

async def _generate_audio(text, voice, response_format, speed):
    """Generate TTS audio and optionally convert to a different format."""
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        supported_formats = ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS))
        raise AudioFormatError(
            f"Unsupported response_format '{response_format}'. Supported formats: {supported_formats}."
        )

    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is

    # Generate the TTS output in mp3 format first
    temp_mp3_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3_file_obj.name

    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    # Generate the MP3 file. Close and remove the placeholder if Edge TTS fails.
    communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
    try:
        await communicator.save(temp_mp3_path)
    except Exception:
        Path(temp_mp3_path).unlink(missing_ok=True)
        raise
    finally:
        temp_mp3_file_obj.close()

    # If the requested format is mp3, return the generated file directly
    if response_format == "mp3":
        return temp_mp3_path

    # Check if FFmpeg is installed
    if not is_ffmpeg_installed():
        Path(temp_mp3_path).unlink(missing_ok=True)
        raise AudioFormatUnavailableError(
            f"response_format={response_format} requires ffmpeg, which is not installed."
        )

    # Create a new temporary file for the converted output
    converted_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    converted_path = converted_file_obj.name
    converted_file_obj.close() # Close file object, ffmpeg will write to the path

    # Build the FFmpeg command
    ffmpeg_command = [ffmpeg_executable(), "-i", temp_mp3_path]
    if response_format == "pcm":
        ffmpeg_command.extend([
            "-ar", "24000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "s16le",
        ])
    else:
        ffmpeg_command.extend([
            "-ar", "24000",
            "-ac", "1",
            "-c:a", {
                "aac": "aac",
                "opus": "libopus",
                "flac": "flac",
                "wav": "pcm_s16le",
            }[response_format],
            "-f", {
                "aac": "adts",
                "opus": "ogg",
                "flac": "flac",
                "wav": "wav",
            }[response_format],
        ])

    if response_format not in {"wav", "pcm", "flac"}:
        ffmpeg_command.extend(["-b:a", "192k"])

    ffmpeg_command.extend(["-y", converted_path])

    try:
        # Run FFmpeg command and ensure no errors occur
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Clean up potentially created (but incomplete) converted file
        Path(converted_path).unlink(missing_ok=True)
        # Clean up the original mp3 file as well, since conversion failed
        Path(temp_mp3_path).unlink(missing_ok=True)
        
        if DETAILED_ERROR_LOGGING:
            error_message = f"FFmpeg error during audio conversion. Command: '{' '.join(e.cmd)}'. Stderr: {e.stderr.decode('utf-8', 'ignore')}"
            print(error_message) # Log for server-side diagnosis
        else:
            error_message = f"FFmpeg error during audio conversion: {e}"
            print(error_message) # Log a simpler message
        raise RuntimeError(f"FFmpeg error during audio conversion: {e}") # The raised error will still have details via e

    # Clean up the original temporary file (original mp3) as it's now converted
    Path(temp_mp3_path).unlink(missing_ok=True)

    return converted_path

def generate_speech(text, voice, response_format, speed=1.0):
    return asyncio.run(_generate_audio(text, voice, response_format, speed))


async def _list_voice_names():
    voices = await edge_tts.list_voices()
    return frozenset(voice["ShortName"] for voice in voices)


@lru_cache(maxsize=1)
def get_available_voice_names():
    return asyncio.run(_list_voice_names())


def validate_voice(voice):
    if voice in voice_mapping:
        return

    try:
        available_voice_names = get_available_voice_names()
    except Exception as exc:
        raise VoiceValidationError("Unable to load the Edge TTS voice catalog.") from exc

    if voice not in available_voice_names:
        raise VoiceValidationError(f"Unsupported voice '{voice}'.")

def get_models():
    return model_data

def get_models_formatted():
    return [{ "id": x["id"] } for x in model_data]

def get_voices_formatted():
    return [{ "id": k, "name": v } for k, v in voice_mapping.items()]

async def _get_voices(language=None):
    # List all voices, filter by language if specified
    all_voices = await edge_tts.list_voices()
    language = language or DEFAULT_LANGUAGE  # Use default if no language specified
    filtered_voices = [
        {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
        for v in all_voices if language == 'all' or language is None or v['Locale'] == language
    ]
    return filtered_voices

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    Converts a multiplicative speed value to the edge-tts "rate" format.
    
    Args:
        speed (float): The multiplicative speed value (e.g., 1.5 for +50%, 0.5 for -50%).
    
    Returns:
        str: The formatted "rate" string (e.g., "+50%" or "-50%").
    """
    if speed < 0.25 or speed > 4:
        raise ValueError("Speed must be between 0.25 and 4 (inclusive).")

    # Convert speed to percentage change
    percentage_change = (speed - 1) * 100

    # Format with a leading "+" or "-" as required
    return f"{percentage_change:+.0f}%"
