import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("REQUIRE_API_KEY", "false")

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import server
import tts_handler
import utils


class SpeechApiTestCase(unittest.TestCase):
    def setUp(self):
        server.REMOVE_FILTER = True
        utils.REQUIRE_API_KEY = False
        self.client = server.app.test_client()

    def _request_payload(self, **overrides):
        payload = {
            "model": "tts-1",
            "input": "你好",
            "voice": "zh-CN-XiaoxiaoNeural",
            "response_format": "mp3",
        }
        payload.update(overrides)
        return payload

    def test_default_mode_returns_raw_mp3(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(b"\xff\xf3test")
            audio_path = audio_file.name

        try:
            with patch.object(server, "validate_voice"), patch.object(
                server, "generate_speech", return_value=audio_path
            ):
                response = self.client.post(
                    "/v1/audio/speech",
                    json=self._request_payload(),
                )
        finally:
            Path(audio_path).unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "audio/mpeg")
        self.assertEqual(response.data, b"\xff\xf3test")

    def test_wav_response_returns_wav_bytes_and_mime_type(self):
        wav_data = b"RIFF\x24\x00\x00\x00WAVEfmt "
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_file.write(wav_data)
            audio_path = audio_file.name

        try:
            with patch.object(server, "validate_voice"), patch.object(
                server, "generate_speech", return_value=audio_path
            ):
                response = self.client.post(
                    "/v1/audio/speech",
                    json=self._request_payload(response_format="wav"),
                )
        finally:
            Path(audio_path).unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "audio/wav")
        self.assertEqual(response.data, wav_data)

    def test_latest_model_and_voice_names_are_accepted(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_path = audio_file.name

        try:
            with patch.object(server, "validate_voice"), patch.object(
                server, "generate_speech", return_value=audio_path
            ):
                response = self.client.post(
                    "/v1/audio/speech",
                    json=self._request_payload(
                        model="gpt-4o-mini-tts-2025-12-15",
                        voice="marin",
                        instructions="Speak warmly.",
                        speed=4,
                    ),
                )
        finally:
            Path(audio_path).unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)

    def test_voice_catalog_does_not_require_api_key(self):
        with patch.object(server, "get_voices", return_value=[{"name": "voice"}]) as get_voices:
            response = self.client.get("/v1/voices/all")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"voices": [{"name": "voice"}]})
        get_voices.assert_called_once_with("all")

    def test_voice_catalog_accepts_language_query_filter(self):
        with patch.object(server, "get_voices", return_value=[]) as get_voices:
            response = self.client.get("/v1/voices?language=zh-CN")

        self.assertEqual(response.status_code, 200)
        get_voices.assert_called_once_with("zh-CN")

    def test_models_expose_standard_openai_list_shape(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["object"], "list")
        self.assertEqual(response.json["data"], response.json["models"])

    def test_sse_returns_audio_events_and_done_marker(self):
        with patch.object(server, "validate_voice"), patch.object(
            server, "generate_speech_stream", return_value=iter([b"a", b"b"])
        ):
            response = self.client.post(
                "/v1/audio/speech",
                json=self._request_payload(stream_format="sse"),
            )

        body = response.data.decode("utf-8")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith("text/event-stream"))
        self.assertIn('data: {"audio": "YQ=="}\n\n', body)
        self.assertIn('data: {"audio": "Yg=="}\n\n', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_unknown_stream_format_is_rejected(self):
        response = self.client.post(
            "/v1/audio/speech",
            json=self._request_payload(stream_format="unknown"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "unsupported_stream_format")

    def test_speed_uses_current_openai_range(self):
        for speed in (0.24, 4.01):
            response = self.client.post(
                "/v1/audio/speech",
                json=self._request_payload(speed=speed),
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["error"]["code"], "invalid_speed")

    def test_unsupported_model_is_rejected(self):
        response = self.client.post(
            "/v1/audio/speech",
            json=self._request_payload(model="not-a-real-model"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "unsupported_model")

    def test_unsupported_voice_is_rejected_with_json_error(self):
        with patch.object(
            server,
            "validate_voice",
            side_effect=tts_handler.VoiceValidationError("Unsupported voice 'bad-voice'."),
        ):
            response = self.client.post(
                "/v1/audio/speech",
                json=self._request_payload(voice="bad-voice"),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "unsupported_voice")
        self.assertEqual(response.json["error"]["type"], "invalid_request_error")

    def test_unavailable_audio_converter_is_not_returned_as_mp3(self):
        with patch.object(server, "validate_voice"), patch.object(
            server,
            "generate_speech",
            side_effect=tts_handler.AudioFormatUnavailableError(
                "response_format=wav requires ffmpeg, which is not installed."
            ),
        ):
            response = self.client.post(
                "/v1/audio/speech",
                json=self._request_payload(response_format="wav"),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "unsupported_response_format")

    def test_bearer_and_api_key_headers_are_accepted(self):
        original_required = utils.REQUIRE_API_KEY
        original_key = utils.API_KEY
        utils.REQUIRE_API_KEY = True
        utils.API_KEY = "secret"
        try:
            for headers in ({"Authorization": "Bearer secret"}, {"api-key": "secret"}):
                response = self.client.post(
                    "/v1/audio/speech",
                    headers=headers,
                    json={},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["error"]["code"], "missing_input")
        finally:
            utils.REQUIRE_API_KEY = original_required
            utils.API_KEY = original_key

    def test_empty_api_key_disables_authentication(self):
        original_required = utils.REQUIRE_API_KEY
        original_key = utils.API_KEY
        utils.REQUIRE_API_KEY = True
        utils.API_KEY = ""
        try:
            with patch.object(server, "validate_voice"), patch.object(
                server, "generate_speech", side_effect=lambda *args: self._temporary_audio_path(b"audio")
            ):
                response = self.client.post(
                    "/v1/audio/speech",
                    json=self._request_payload(),
                )
        finally:
            utils.REQUIRE_API_KEY = original_required
            utils.API_KEY = original_key

        self.assertEqual(response.status_code, 200)

    @staticmethod
    def _temporary_audio_path(data):
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_file.write(data)
        audio_file.close()
        return audio_file.name


class SpeechStreamBridgeTestCase(unittest.TestCase):
    def test_async_generator_is_exposed_as_sync_generator(self):
        async def fake_async_stream(*args, **kwargs):
            yield b"first"
            yield b"second"

        with patch.object(tts_handler, "_generate_audio_stream", side_effect=fake_async_stream):
            chunks = list(tts_handler.generate_speech_stream("text", "voice", 1.0))

        self.assertEqual(chunks, [b"first", b"second"])


if __name__ == "__main__":
    unittest.main()
