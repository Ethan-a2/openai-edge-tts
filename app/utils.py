# utils.py

from flask import request, jsonify
from functools import wraps
import os
from dotenv import load_dotenv

from config import DEFAULT_CONFIGS

load_dotenv()

def getenv_bool(name: str, default: bool = False) -> bool:
    # The default parameter for getenv_bool is used if the config default itself needs a fallback,
    # or if the call site specifically wants to override the global default.
    # For typical usage, the config default (passed at call site) is preferred.
    return os.getenv(name, str(default)).lower() in ("yes", "y", "true", "1", "t")

API_KEY = os.getenv('API_KEY', DEFAULT_CONFIGS["API_KEY"])
REQUIRE_API_KEY = getenv_bool('REQUIRE_API_KEY', DEFAULT_CONFIGS["REQUIRE_API_KEY"])
DETAILED_ERROR_LOGGING = getenv_bool('DETAILED_ERROR_LOGGING', DEFAULT_CONFIGS["DETAILED_ERROR_LOGGING"])


def api_error(message: str, code: str, status_code: int = 400, error_type: str = "invalid_request_error"):
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }), status_code

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not REQUIRE_API_KEY:
            return f(*args, **kwargs)
        candidate_tokens = []
        auth_header = request.headers.get('Authorization', '')
        if auth_header:
            scheme, separator, token = auth_header.partition(' ')
            if separator and scheme.lower() == 'bearer' and token:
                candidate_tokens.append(token)

        api_key_header = request.headers.get('api-key')
        if api_key_header:
            candidate_tokens.append(api_key_header)

        if not candidate_tokens:
            return api_error("Missing API key", "missing_api_key", 401, "authentication_error")
        if API_KEY not in candidate_tokens:
            return api_error("Invalid API key", "invalid_api_key", 401, "authentication_error")
        return f(*args, **kwargs)
    return decorated_function

# Mapping of audio format to MIME type
AUDIO_FORMAT_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm"
}
