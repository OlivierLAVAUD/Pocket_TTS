"""Tests for the Gradio playground / API in ``pocket_tts.py`` (repository root).

The module is loaded from its path because it shares its name with the
``pocket_tts`` package: ``import pocket_tts`` must keep resolving to the package.

The model is replaced by a stub: these tests cover the plumbing (voice
resolution, WAV encoding, API routes, locking), not the synthesis quality.
"""

# Copyright (c) 2026 oLV - Olivier LAVAUD
# SPDX-License-Identifier: MIT

import base64
import importlib.util
import io
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from pocket_tts.models.tts_model import TTSModel

pytest.importorskip("gradio")

APP_PATH = Path(__file__).resolve().parent.parent / "pocket_tts.py"


def _load_app() -> object:
    """Import ``pocket_tts.py`` under its own name, registering it as a module."""
    spec = importlib.util.spec_from_file_location("pocket_tts_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registering it is what importlib's own machinery does, and what dataclasses
    # needs to resolve the annotations of the module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


# The app module is loaded by path (see _load_app), so its attributes cannot be
# resolved statically: keep it as Any for the type checker.
app_module: Any = _load_app()

SAMPLE_RATE = 24000


class _StubModel:
    """Stand-in for ``TTSModel``: the same surface the engine uses, no weights."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.device = "cpu"
        self.encoded: list[tuple[str, bool]] = []

    def to(self, device: str) -> "_StubModel":
        self.device = device
        return self

    def get_state_for_audio_prompt(
        self, source: str | Path, truncate: bool = False
    ) -> dict[str, dict[str, torch.Tensor]]:
        self.encoded.append((str(source), truncate))
        return {"stub": {}}

    def generate_audio_stream(
        self,
        model_state: dict[str, dict[str, torch.Tensor]],
        text_to_generate: str,
        max_tokens: int = 50,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: object | None = None,
    ):
        assert text_to_generate.strip(), "the app must not send an empty prompt"
        yield torch.zeros(int(SAMPLE_RATE * 0.1))
        yield torch.full((int(SAMPLE_RATE * 0.05),), 0.5)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubModel:
    model = _StubModel()
    monkeypatch.setattr(TTSModel, "load_model", classmethod(lambda cls, **kwargs: model))
    return model


@pytest.fixture
def engine(stub: _StubModel) -> "app_module.TTSEngine":
    settings = app_module.Settings(language="english", config=None, checkpoint=None, voice=[])
    return app_module.TTSEngine(settings)


@pytest.fixture
def client(engine: "app_module.TTSEngine") -> TestClient:
    return TestClient(app_module.build_api(engine, engine.settings))


def _read_wav(data: bytes) -> wave.Wave_read:
    return wave.open(io.BytesIO(data), "rb")


def _write_wav(path: Path, seconds: float = 0.2) -> Path:
    samples = (np.sin(np.arange(int(SAMPLE_RATE * seconds)) / 20) * 1000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return path


def test_health(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "healthy"
    assert payload["sample_rate"] == SAMPLE_RATE
    assert payload["language"] == "english"


def test_voices(client: TestClient) -> None:
    payload = client.get("/api/voices").json()
    assert payload["default_voice"] == "alba"
    assert set(app_module.PREDEFINED_VOICES) <= set(payload["voices"])
    assert payload["local_voices"] == []


def test_generate_returns_a_wav(client: TestClient) -> None:
    response = client.post("/api/generate", json={"text": "Hello world."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with _read_wav(response.content) as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        assert len(handle.readframes(100)) > 0


def test_generate_json_returns_base64_audio(client: TestClient) -> None:
    response = client.post(
        "/api/generate", json={"text": "Hello world.", "response_format": "json"}
    )
    payload = response.json()
    assert payload["sample_rate"] == SAMPLE_RATE
    with _read_wav(base64.b64decode(payload["audio_base64"])) as handle:
        assert len(handle.readframes(100)) > 0


def test_generate_rejects_empty_text(client: TestClient) -> None:
    assert client.post("/api/generate", json={"text": "   "}).status_code == 400


def test_generate_rejects_out_of_range_parameters(client: TestClient) -> None:
    assert client.post("/api/generate", json={"text": "hi", "max_tokens": 5}).status_code == 422


def test_streaming_endpoint_returns_a_wav(client: TestClient) -> None:
    response = client.post("/api/tts", data={"text": "Hello world."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with _read_wav(response.content) as handle:
        assert len(handle.readframes(1000)) > 0


def test_streaming_endpoint_rejects_empty_text(client: TestClient) -> None:
    assert client.post("/api/tts", data={"text": "  "}).status_code == 400


def test_streaming_endpoint_uses_the_uploaded_voice(
    client: TestClient, stub: _StubModel, tmp_path: Path
) -> None:
    reference = _write_wav(tmp_path / "reference.wav")
    response = client.post(
        "/api/tts",
        data={"text": "Hello world."},
        files={"voice_wav": (reference.name, reference.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    # The uploaded file is encoded from a temporary copy, truncated to 30 s.
    assert stub.encoded and stub.encoded[-1][1] is True


def test_voice_state_is_encoded_once(engine: "app_module.TTSEngine") -> None:
    with engine._lock:
        engine.voice_state("alba")
        engine.voice_state("alba")
    assert [name for name, _ in engine.model.encoded] == ["alba"]


def test_reference_audio_is_cached_but_not_listed(
    engine: "app_module.TTSEngine", tmp_path: Path
) -> None:
    reference = _write_wav(tmp_path / "reference.wav")
    with engine._lock:
        engine.voice_state(None, str(reference))
        engine.voice_state(None, str(reference))
    assert [name for name, _ in engine.model.encoded] == [str(reference)]
    # An uploaded reference must not appear in the voice list: the API deletes the
    # temporary file as soon as the response is sent.
    assert "reference" not in engine.voice_names()


def test_local_voice_is_registered_and_used(engine: "app_module.TTSEngine", tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "my_voice.wav")
    name = engine.register_local_voice(path)
    assert name == "my_voice"
    assert name in engine.voice_names()
    assert engine.resolve_voice(name) == path
    # With a local voice registered it becomes the default, as an absolute path.
    assert engine.resolve_voice("") == path
    assert engine.default_voice() == name


def test_register_local_voice_rejects_a_missing_file(engine: "app_module.TTSEngine") -> None:
    with pytest.raises(FileNotFoundError):
        engine.register_local_voice(Path("/does/not/exist.wav"))


def test_synthesize_returns_samples(engine: "app_module.TTSEngine") -> None:
    samples = engine.synthesize("Hello world.", voice="alba")
    assert samples.dtype == np.float32
    assert samples.shape == (int(SAMPLE_RATE * 0.15),)


def test_synthesize_rejects_empty_text(engine: "app_module.TTSEngine") -> None:
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", voice="alba")


def test_samples_to_wav_keeps_the_buffer_readable() -> None:
    samples = np.zeros(100, dtype=np.float32)
    wav = app_module.samples_to_wav(samples, SAMPLE_RATE)
    assert wav.startswith(b"RIFF")
    with _read_wav(wav) as handle:
        # StreamingWAVWriter.finalize() pads 200 ms of silence for playback.
        assert handle.getnframes() == 100 + int(0.2 * SAMPLE_RATE)


def test_stream_wav_yields_riff_bytes(engine: "app_module.TTSEngine") -> None:
    chunks = list(engine.stream_wav("Hello world.", voice="alba"))
    assert chunks and b"".join(chunks).startswith(b"RIFF")


def test_ui_is_mounted_beside_the_api(engine: "app_module.TTSEngine") -> None:
    demo = app_module.build_ui(engine, engine.settings)
    assert isinstance(demo, app_module.gr.Blocks)
    app = app_module.build_app(engine, engine.settings)
    routes = {getattr(route, "path", None) for route in app.routes}
    # The UI is mounted last, so the API routes keep matching first.
    assert {"/api/health", "/api/voices", "/api/tts", "/api/generate"} <= routes


def test_no_ui_serves_only_the_api(engine: "app_module.TTSEngine") -> None:
    app = app_module.build_app(engine, app_module.Settings(with_ui=False))
    assert all(getattr(route, "path", "") != "/" for route in app.routes)


def test_parse_args_defaults_and_overrides() -> None:
    settings = app_module.parse_args([])
    assert settings.language is None
    assert settings.with_ui is True

    settings = app_module.parse_args(
        ["--language", "french", "--port", "9000", "--voice", "/a.wav", "--voice", "/b.wav"]
    )
    assert settings.language == "french"
    assert settings.port == 9000
    assert settings.voice == ["/a.wav", "/b.wav"]

    assert app_module.parse_args(["--no-ui"]).with_ui is False
