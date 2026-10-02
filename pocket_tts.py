"""Pocket TTS playground: a Gradio web UI and a JSON HTTP API in one process.

One model instance is shared by the Gradio UI (mounted at ``/``) and by the API
(``/api/...``). pocket-tts is CPU-first, batch size 1 and *not* thread-safe, so
every generation is serialized through a single lock instead of loading several
copies of the 100M-parameter weights.

This file sits next to the ``pocket_tts`` package and shares its name, which is
fine: a directory with an ``__init__.py`` always wins over a module of the same
name, so ``import pocket_tts`` still resolves to the package.

Run it with::

    uv run --with gradio --with soundfile python pocket_tts.py --language french

Or with Docker::

    docker compose up pocket-tts-app

The API mirrors the ``pocket-tts serve`` server: ``POST /api/tts`` streams a WAV
file (the endpoint the browser UI of ``serve`` uses) and ``POST /api/generate``
returns either the WAV bytes or a base64 JSON payload.
"""

# Copyright (c) 2026 oLV - Olivier LAVAUD
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import base64
import inspect
import io
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from pydantic import Field as PydanticField

from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger("pocket_tts.app")

# Voice sources whose state is kept in memory between two generations. Encoding a
# prompt costs a full Mimi pass over the reference audio, so it must not happen
# again for every request.
VOICE_CACHE_SIZE = 8

PREDEFINED_VOICES = sorted(_ORIGINS_OF_PREDEFINED_VOICES)


class _BytesSink(io.BytesIO):
    """BytesIO that survives the ``with f:`` of ``stream_audio_chunks``.

    ``stream_audio_chunks`` closes the stream it is given; keeping the buffer
    open is what makes it usable as an in-memory WAV encoder.
    """

    def close(self) -> None:
        self.flush()


class _QueueWriter(io.IOBase):
    """File-like object pushing the WAV bytes it receives into a queue.

    Subclasses ``io.IOBase`` so that ``stream_audio_chunks`` can use it in a
    ``with`` statement and treat it as an unseekable stream.
    """

    def __init__(self, queue: "Queue[bytes | None]") -> None:
        super().__init__()
        self.queue = queue
        self._done = False

    def write(self, data: bytes) -> int:
        self.queue.put(data)
        return len(data)

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        if not self._done:
            self._done = True
            # None is the end-of-stream marker consumed by `TTSEngine.stream_wav`.
            self.queue.put(None)


@dataclass
class Settings:
    """Command line / environment options of the app."""

    host: str = field(default_factory=lambda: os.environ.get("POCKET_TTS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("POCKET_TTS_PORT", "7860")))
    language: str | None = field(
        default_factory=lambda: os.environ.get("POCKET_TTS_LANGUAGE") or None
    )
    config: str | None = field(default_factory=lambda: os.environ.get("POCKET_TTS_CONFIG") or None)
    checkpoint: str | None = field(
        default_factory=lambda: os.environ.get("POCKET_TTS_CHECKPOINT") or None
    )
    # Local reference files, colon separated, exposed in the UI under their stem.
    voice: list[str] = field(
        default_factory=lambda: [v for v in os.environ.get("POCKET_TTS_VOICE", "").split(":") if v]
    )
    device: str = field(default_factory=lambda: os.environ.get("POCKET_TTS_DEVICE", "auto"))
    quantize: bool = field(
        default_factory=lambda: os.environ.get("POCKET_TTS_QUANTIZE", "0") == "1"
    )
    temperature: float | None = None
    sampler_decode_steps: int = 1
    eos_threshold: float = -4.0
    max_tokens: int = 50
    log_level: str = field(default_factory=lambda: os.environ.get("POCKET_TTS_LOG_LEVEL", "INFO"))
    with_ui: bool = True
    share: bool = False

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class TTSEngine:
    """Loads the model once and serializes the generations that use it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.resolved_device()
        logger.info(
            "Loading Pocket TTS (language=%s, config=%s, checkpoint=%s, quantize=%s, device=%s)",
            settings.language,
            settings.config,
            settings.checkpoint,
            settings.quantize,
            self.device,
        )
        self.model = TTSModel.load_model(
            language=settings.language,
            config=settings.config,
            temp=settings.temperature,
            sampler_decode_steps=settings.sampler_decode_steps,
            eos_threshold=settings.eos_threshold,
            quantize=settings.quantize,
            checkpoint=settings.checkpoint,
        )
        self.model.to(self.device)
        self.sample_rate: int = self.model.sample_rate
        # A generation mutates the model state and its KV cache; the model is
        # explicitly not thread-safe, so a single lock covers every request, and
        # the model state is deep-copied per request by generate_audio_stream.
        self._lock = threading.Lock()
        self._voices: OrderedDict[str, ModelState] = OrderedDict()
        self._local_voices: dict[str, str] = {}
        for path in settings.voice:
            self.register_local_voice(Path(path))

    # ---------------------------------------------------------------- voices

    def register_local_voice(self, path: Path) -> str:
        """Make a local audio file available under its stem in the UI."""
        if not path.exists():
            raise FileNotFoundError(f"Voice file not found: {path}")
        name = path.stem
        self._local_voices[name] = str(path)
        logger.info("Registered local voice %r -> %s", name, path)
        return name

    @property
    def local_voices(self) -> dict[str, str]:
        """Local reference files, by name."""
        return dict(self._local_voices)

    def voice_names(self) -> list[str]:
        extra = sorted(set(self._local_voices) - set(PREDEFINED_VOICES))
        return PREDEFINED_VOICES + extra

    def default_voice(self) -> str:
        names = self.voice_names()
        if not names:
            raise RuntimeError("No voice is available")
        if self._local_voices:
            return sorted(self._local_voices)[0]
        language = self.settings.language or ""
        for language_key, voice in {
            "italian": "giovanni",
            "spanish": "lola",
            "german": "juergen",
            "portuguese": "rafael",
            "french": "estelle",
            "dutch": "daan",
        }.items():
            if language_key in language and voice in names:
                return voice
        return "alba" if "alba" in names else names[0]

    def resolve_voice(self, voice: str | Path | None) -> str | Path:
        """Map a UI/API voice value onto something ``get_state_for_audio_prompt`` accepts."""
        if voice is None or str(voice).strip() == "":
            voice = self.default_voice()
        name = str(voice).strip()
        if name in self._local_voices:
            return Path(self._local_voices[name])
        return name

    def voice_state(
        self, voice: str | Path | None = None, reference_audio: str | None = None
    ) -> ModelState:
        """Cached voice state for a named voice, or for a reference recording.

        ``reference_audio`` wins over ``voice`` and is only cached: an uploaded or
        temporary file is not added to the voice list, which would leave a name
        pointing at a file deleted once the request is over.
        Must be called while holding ``self._lock``.
        """
        if reference_audio:
            return self._cached_voice_state(str(reference_audio), reference_audio)
        source = self.resolve_voice(voice)
        return self._cached_voice_state(str(source), source)

    def _cached_voice_state(self, key: str, source: str | Path) -> ModelState:
        """Encode the source once and keep the state for the next requests."""
        cached = self._voices.get(key)
        if cached is not None:
            self._voices.move_to_end(key)
            return cached
        state = self.model.get_state_for_audio_prompt(source, truncate=True)
        self._voices[key] = state
        while len(self._voices) > VOICE_CACHE_SIZE:
            self._voices.popitem(last=False)
        return state

    # ------------------------------------------------------------ generation

    def _check_text(self, text: str | None) -> str:
        if text is None or not text.strip():
            raise ValueError("Text cannot be empty")
        return text

    def synthesize(
        self,
        text: str,
        voice: str | Path | None = None,
        frames_after_eos: int | None = None,
        max_tokens: int | None = None,
        reference_audio: str | None = None,
    ) -> np.ndarray:
        """Generate the whole utterance and return mono float32 samples."""
        text = self._check_text(text)
        with self._lock:
            state = self.voice_state(voice, reference_audio)
            chunks = [
                chunk
                for chunk in self.model.generate_audio_stream(
                    model_state=state,
                    text_to_generate=text,
                    max_tokens=max_tokens or self.settings.max_tokens,
                    frames_after_eos=frames_after_eos,
                )
            ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return torch.cat(chunks, dim=0).detach().cpu().numpy()

    def synthesize_wav(
        self,
        text: str,
        voice: str | Path | None = None,
        frames_after_eos: int | None = None,
        max_tokens: int | None = None,
        reference_audio: str | None = None,
    ) -> bytes:
        """Same as ``synthesize`` but returns the bytes of a complete WAV file."""
        samples = self.synthesize(
            text=text,
            voice=voice,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
            reference_audio=reference_audio,
        )
        return samples_to_wav(samples, self.sample_rate)

    def stream_wav(
        self,
        text: str,
        voice: str | Path | None = None,
        frames_after_eos: int | None = None,
        max_tokens: int | None = None,
        reference_audio: str | None = None,
    ) -> Iterator[bytes]:
        """Yield WAV bytes as soon as the decoder has produced them.

        The generation happens in a worker thread because the lock has to be held
        for its whole duration, and because the first bytes must reach the client
        before the last frame is generated.
        """
        text = self._check_text(text)
        queue: Queue[bytes | None] = Queue()
        stop = threading.Event()
        writer = _QueueWriter(queue)
        error: list[BaseException] = []

        def worker() -> None:
            try:
                with self._lock:
                    state = self.voice_state(voice, reference_audio)
                    chunks = self.model.generate_audio_stream(
                        model_state=state,
                        text_to_generate=text,
                        max_tokens=max_tokens or self.settings.max_tokens,
                        frames_after_eos=frames_after_eos,
                        stop=stop,
                    )
                    stream_audio_chunks(writer, chunks, self.sample_rate)
            except BaseException as exc:  # surfaced to the client below
                error.append(exc)
                logger.exception("Streaming generation failed")
            finally:
                writer.close()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                data = queue.get()
                if data is None:
                    break
                yield data
        finally:
            # Also runs when the client disconnects: stop generating for nobody.
            stop.set()
            thread.join()
        if error:
            raise error[0]


def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float samples as a 16-bit PCM WAV file, in memory."""
    sink = _BytesSink()
    stream_audio_chunks(sink, iter([torch.from_numpy(np.asarray(samples))]), sample_rate)
    return sink.getvalue()


class TTSRequest(BaseModel):
    """JSON body of ``POST /api/generate``."""

    text: str = PydanticField(..., description="Text to synthesize")
    voice: str | None = PydanticField(
        None, description="Predefined voice name, local file, http(s):// or hf:// URL"
    )
    language: str | None = PydanticField(
        None, description="Language of the loaded model (informative, read-only per server)"
    )
    frames_after_eos: int | None = PydanticField(None, ge=0, le=30)
    max_tokens: int | None = PydanticField(None, ge=10, le=200)
    response_format: str = PydanticField("wav", pattern="^(wav|json)$")


def _wav_response(engine: TTSEngine, request: TTSRequest) -> Response:
    if request.response_format == "json":
        wav = engine.synthesize_wav(
            text=request.text,
            voice=request.voice,
            frames_after_eos=request.frames_after_eos,
            max_tokens=request.max_tokens,
        )
        return JSONResponse(
            {
                "sample_rate": engine.sample_rate,
                "format": "wav",
                "audio_base64": base64.b64encode(wav).decode("ascii"),
            }
        )
    samples = engine.synthesize(
        text=request.text,
        voice=request.voice,
        frames_after_eos=request.frames_after_eos,
        max_tokens=request.max_tokens,
    )
    return Response(
        content=samples_to_wav(samples, engine.sample_rate),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def build_api(engine: TTSEngine, settings: Settings) -> FastAPI:
    """The HTTP API served next to the Gradio UI."""
    api = FastAPI(
        title="Kyutai Pocket TTS API",
        description="Gradio playground and JSON API for Pocket TTS",
        version="1.0.0",
    )

    @api.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "language": settings.language,
            "config": settings.config,
            "checkpoint": settings.checkpoint,
            "device": engine.device,
            "quantize": settings.quantize,
            "sample_rate": engine.sample_rate,
        }

    @api.get("/api/voices")
    def voices() -> dict[str, Any]:
        return {
            "default_voice": engine.default_voice(),
            "voices": engine.voice_names(),
            "local_voices": sorted(engine.local_voices),
        }

    @api.post("/api/generate")
    def generate(request: TTSRequest) -> Response:
        try:
            return _wav_response(engine, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/tts")
    def tts(
        text: str = Form(...),
        voice: str | None = Form(None),
        frames_after_eos: int | None = Form(None),
        max_tokens: int | None = Form(None),
        voice_wav: UploadFile | None = File(None),
    ) -> StreamingResponse:
        """Streaming endpoint, shaped like the one of ``pocket-tts serve``."""
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        reference: str | None = None
        try:
            if voice_wav is not None:
                suffix = Path(voice_wav.filename or "voice.wav").suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(voice_wav.file.read())
                    reference = temp_file.name
            stream = engine.stream_wav(
                text=text,
                voice=voice,
                frames_after_eos=frames_after_eos,
                max_tokens=max_tokens,
                reference_audio=reference,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            _cleanup_after(stream, reference),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="generated_speech.wav"',
                "Transfer-Encoding": "chunked",
            },
        )

    return api


def _cleanup_after(stream: Iterator[bytes], temp_file: str | None) -> Iterator[bytes]:
    """Yield the stream, then delete the uploaded reference audio."""
    try:
        yield from stream
    finally:
        if temp_file is not None:
            try:
                os.unlink(temp_file)
            except OSError:
                logger.warning("Could not delete temporary voice file %s", temp_file)


DEFAULT_TEXT = (
    "Hello world. I am Kyutai's Pocket TTS. "
    "I'm fast enough to run on small CPUs. "
    "I hope you'll like me."
)


def _audio_download_kwargs() -> dict[str, Any]:
    """The download-button option of ``gr.Audio``, whatever the installed Gradio calls it.

    Gradio 5 spells it ``show_download_button``; Gradio 6 replaced it with a
    ``buttons`` list, so build the keyword only when the component accepts it.
    """
    parameters = inspect.signature(gr.Audio.__init__).parameters
    if "show_download_button" in parameters:
        return {"show_download_button": True}
    if "buttons" in parameters:
        return {"buttons": ["download"]}
    return {}


def build_ui(engine: TTSEngine, settings: Settings) -> gr.Blocks:
    """The Gradio playground, exposed as ``/`` and as the Gradio API."""
    voices = engine.voice_names()
    default_voice = engine.default_voice()

    def ui_generate(
        text: str, voice: str, reference: str | None, frames_after_eos: float, max_tokens: float
    ) -> tuple[int, Any]:
        try:
            samples = engine.synthesize(
                text=text,
                voice=voice,
                frames_after_eos=int(frames_after_eos) if frames_after_eos else None,
                max_tokens=int(max_tokens) if max_tokens else None,
                reference_audio=reference,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # keep the UI alive on model failures
            logger.exception("Generation failed")
            raise gr.Error(f"Generation failed: {exc}") from exc
        if samples.size == 0:
            raise gr.Error("The model produced no audio, try a longer text")
        duration = samples.shape[-1] / engine.sample_rate
        gr.Info(f"Generated {duration:.2f}s of audio with {engine.device}")
        return engine.sample_rate, samples

    with gr.Blocks(title="Kyutai Pocket TTS") as demo:
        gr.Markdown(
            "# Pocket TTS\n"
            "Kyutai's CPU-first text-to-speech model. Pick a voice from the catalog, or drop "
            "a reference recording to clone a voice.\n\n"
            "The same server exposes the JSON API on `/api/health`, `/api/voices`, "
            "`/api/generate` and the streaming `/api/tts`."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text",
                    value=DEFAULT_TEXT,
                    lines=5,
                    placeholder="Type the text to synthesize...",
                )
                voice = gr.Dropdown(
                    label="Voice",
                    choices=voices,
                    value=default_voice,
                    allow_custom_value=True,
                    info="Predefined voice, a local file path, or an http(s):// / hf:// URL",
                )
                reference = gr.Audio(
                    label="Reference audio (optional, overrides the voice above)",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                with gr.Accordion("Advanced parameters", open=False):
                    frames_after_eos = gr.Slider(
                        label="Frames after EOS",
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=0,
                        info="0 lets the model pick its recommended value",
                    )
                    max_tokens = gr.Slider(
                        label="Max tokens per chunk",
                        minimum=10,
                        maximum=200,
                        step=1,
                        value=settings.max_tokens,
                    )
                generate = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                output = gr.Audio(
                    label="Generated audio", type="numpy", autoplay=True, **_audio_download_kwargs()
                )
                gr.Examples(
                    examples=[
                        [DEFAULT_TEXT, "alba"],
                        ["Bonjour, je suis le TTS de poche de Kyutai.", "estelle"],
                        ["Hallo Welt, ich bin Pocket TTS von Kyutai.", "juergen"],
                    ],
                    inputs=[text, voice],
                )
                gr.Markdown(
                    "Generated audio is 24 kHz mono. Long texts are split into "
                    "sentence-sized chunks and streamed one chunk at a time."
                )

        inputs = [text, voice, reference, frames_after_eos, max_tokens]
        generate.click(ui_generate, inputs=inputs, outputs=output, api_name="generate")
        text.submit(ui_generate, inputs=inputs, outputs=output, api_name=False)

    return demo.queue(default_concurrency_limit=1, max_size=8)


def parse_args(argv: list[str] | None = None) -> Settings:
    """Read the CLI options, defaulting to the POCKET_TTS_* environment variables."""
    defaults = Settings()
    parser = argparse.ArgumentParser(
        description="Pocket TTS Gradio playground with a JSON API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=defaults.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=defaults.port, help="Port to bind to")
    parser.add_argument("--language", default=defaults.language, help="Language config to load")
    parser.add_argument(
        "--config",
        default=defaults.config,
        help="Path to a model config .yaml: local path, https:// URL or hf:// path",
    )
    parser.add_argument(
        "--checkpoint", default=defaults.checkpoint, help="Training checkpoint (.pt) to load"
    )
    parser.add_argument(
        "--voice",
        action="append",
        default=None,
        help="Local reference audio to expose in the UI (repeatable)",
    )
    parser.add_argument("--device", default=defaults.device, help="cpu, cuda or auto")
    parser.add_argument(
        "--quantize", action="store_true", default=defaults.quantize, help="int8 quantization"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--sampler-decode-steps", type=int, default=defaults.sampler_decode_steps)
    parser.add_argument("--eos-threshold", type=float, default=defaults.eos_threshold)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--no-ui", action="store_true", help="Serve only the JSON API")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Launch the Gradio UI with a public Gradio link (API routes are then not served)",
    )
    parser.add_argument("--log-level", default=defaults.log_level)
    args = parser.parse_args(argv)

    return Settings(
        host=args.host,
        port=args.port,
        language=args.language,
        config=args.config,
        checkpoint=args.checkpoint,
        voice=args.voice if args.voice is not None else defaults.voice,
        device=args.device,
        quantize=args.quantize,
        temperature=args.temperature,
        sampler_decode_steps=args.sampler_decode_steps,
        eos_threshold=args.eos_threshold,
        max_tokens=args.max_tokens,
        log_level=args.log_level,
        with_ui=not args.no_ui,
        share=args.share,
    )


def build_app(engine: TTSEngine, settings: Settings) -> FastAPI:
    """The FastAPI app: JSON API plus, when enabled, the Gradio UI mounted at ``/``."""
    api = build_api(engine, settings)
    if not settings.with_ui:
        return api
    demo = build_ui(engine, settings)
    # Mounting at "/" keeps the API routes above: Starlette matches routes in
    # registration order, so /api/* is served by the handlers declared before.
    return gr.mount_gradio_app(api, demo, path="/")


def main(argv: list[str] | None = None) -> None:
    settings = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    engine = TTSEngine(settings)
    logger.info(
        "Model ready on %s (%d Hz, voices: %s)",
        engine.device,
        engine.sample_rate,
        ", ".join(engine.voice_names()[:6]) + ("..." if len(engine.voice_names()) > 6 else ""),
    )

    if settings.share:
        if not settings.with_ui:
            raise SystemExit("--share requires the UI, remove --no-ui")
        logger.warning("--share: only the Gradio UI is served, /api/* routes are not exposed")
        build_ui(engine, settings).launch(
            server_name=settings.host, server_port=settings.port, share=True
        )
        return

    uvicorn.run(
        build_app(engine, settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
