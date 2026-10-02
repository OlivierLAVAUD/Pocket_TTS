<!-- Copyright (c) 2026 oLV - Olivier LAVAUD — SPDX-License-Identifier: MIT -->

# Manual — Gradio app and JSON API for Pocket TTS

A web app (Gradio UI) and HTTP API for [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
served by a **single process** with the **model loaded once** in memory.

| File | Role |
| --- | --- |
| `pocket_tts.py` | The application: Gradio UI + JSON API (single file, repository root) |
| `tests/test_app.py` | 20 tests (stubbed model, nothing to download) |
| `Dockerfile.app` | Docker image of the application |
| `docker-compose.yaml` | `pocket-tts-app` service (next to the existing `pocket-tts` service) |
| `docker-bake.hcl` | `pocket-tts-app` build target |

For a shorter version, see [`README.md`](./README.md).

---

## 1. Quick start (local)

Gradio is an **optional** dependency (not installed by default, so that `uv.lock` stays untouched):

```bash
# from a repository checkout
uv run --with gradio --with soundfile python pocket_tts.py --language french

# if pocket-tts and gradio are already installed in your environment
python pocket_tts.py --host 0.0.0.0 --port 7860
```

Then open <http://localhost:7860>.

> **File name.** `pocket_tts.py` sits next to the `pocket_tts/` package. This is not a conflict: in
> Python, a directory containing `__init__.py` always wins over a module of the same name, so
> `import pocket_tts` and `python -m pocket_tts` still refer to the package.

Start without the web interface (API only):

```bash
python pocket_tts.py --no-ui --port 7860
```

Temporary public link (Gradio only, the `/api/*` routes are not served):

```bash
python pocket_tts.py --share
```

---

## 2. Options and environment variables

Every option has an environment variable equivalent (handy in a container).

| Option | Variable | Default | Description |
| --- | --- | --- | --- |
| `--host` | `POCKET_TTS_HOST` | `0.0.0.0` | Listening interface |
| `--port` | `POCKET_TTS_PORT` | `7860` | Port |
| `--language` | `POCKET_TTS_LANGUAGE` | `english` | Model: `french`, `german_24l`, `english_drifting_26-09`… |
| `--config` | `POCKET_TTS_CONFIG` | – | Local YAML, `https://` URL or `hf://` path (incompatible with `--language`) |
| `--checkpoint` | `POCKET_TTS_CHECKPOINT` | – | Training checkpoint (`.pt`) |
| `--voice` (repeatable) | `POCKET_TTS_VOICE` (separated by `:`) | – | Reference recordings exposed in the UI and the API |
| `--device` | `POCKET_TTS_DEVICE` | `auto` | `cpu`, `cuda` or `auto` |
| `--quantize` | `POCKET_TTS_QUANTIZE=1` | disabled | int8 quantization (CPU only) |
| `--temperature` | – | the model's | Sampling temperature |
| `--sampler-decode-steps` | – | `1` | Flow decoding steps |
| `--eos-threshold` | – | `-4.0` | End-of-sequence threshold |
| `--max-tokens` | – | `50` | Maximum tokens per sentence chunk |
| `--no-ui` | – | interface enabled | Serve only the JSON API |
| `--share` | – | – | Public Gradio link (no API) |
| `--log-level` | `POCKET_TTS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`… |

In the UI: text, voice (predefined or `http(s)://` / `hf://` file URL), reference audio (upload or
microphone, takes priority over the voice), and the "frames after EOS" and "max tokens" sliders.
The available voices are listed by `GET /api/voices` (27 predefined voices).

---

## 3. HTTP API

### `GET /api/health`

```bash
curl http://localhost:7860/api/health
```

```json
{"status": "healthy", "language": "french", "config": null, "checkpoint": null,
 "device": "cpu", "quantize": false, "sample_rate": 24000}
```

### `GET /api/voices`

```json
{"default_voice": "estelle", "voices": ["alba", "anna", "..."], "local_voices": ["my_recording"]}
```

### `POST /api/generate` — JSON, WAV or base64 response

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `string` | *required* | Text to synthesize |
| `voice` | `string` | default voice | Predefined voice, local path, `http(s)://` / `hf://` URL |
| `frames_after_eos` | `int` (0–30) | model | Frames generated after the end of sequence |
| `max_tokens` | `int` (10–200) | `50` | Maximum chunk size |
| `response_format` | `"wav"` or `"json"` | `"wav"` | `json` returns the audio as base64 |

```bash
curl -X POST http://localhost:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world.", "voice": "estelle"}' -o speech.wav
```

```bash
# as JSON (base64)
curl -X POST http://localhost:7860/api/generate -H 'Content-Type: application/json' \
  -d '{"text": "Hello world.", "response_format": "json"}'
```

### `POST /api/tts` — multipart, **streamed** WAV

Same shape as the `/tts` route of `pocket-tts serve`: the WAV is sent as the Mimi decoder produces
the chunks (first bytes after ~200 ms).

```bash
curl -X POST http://localhost:7860/api/tts \
  -F 'text=Hello, this is a test.' -F 'voice=estelle' -o stream.wav

# with a voice to clone from a file
curl -X POST http://localhost:7860/api/tts \
  -F 'text=Hello.' -F 'voice_wav=@my_voice.wav' -o stream.wav
```

The Gradio API is exposed as well (the *Generate* button, `api_name="generate"`).

---

## 4. Docker

```bash
# build + run
docker build -f Dockerfile.app -t pocket-tts-app .
docker run --rm -p 7860:7860 -v hf-cache:/root/.cache/huggingface \
  pocket-tts-app --language french

# with compose (next to the existing `serve` server)
docker compose up pocket-tts-app -d
docker compose logs -f pocket-tts-app

# multi-platform build with docker-bake (pocket-tts-app target)
docker buildx bake pocket-tts-app
```

Mount the model caches (`/root/.cache/huggingface` and `/root/.cache/pocket_tts`, as the compose
file does) to avoid downloading the ~440 MB of weights on every start. Voice cloning requires the
gated weights: accept the terms on
[huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) and pass `HF_TOKEN`
(otherwise only the predefined voices are available).



