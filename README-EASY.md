# Pocket TTS — Easy Start

The fastest way to get a working text-to-speech web app: a **Gradio UI** in your browser plus a
**JSON API**, running on your CPU. No GPU needed.

**In a nutshell:**

```bash
git clone https://github.com/OlivierLAVAUD/Pocket_TTS.git
cd Pocket_TTS
docker compose up -d
```

Then open <http://localhost:7860>.

---

## 1. Pick your mode

| Mode | You need | Command | Disk / notes |
| --- | --- | --- | --- |
| **[A] Docker** *(recommended)* | Docker only | `docker compose up -d` | ~20 GB image (built once), nothing to install |
| **[B] Light (Linux)** | Python 3.10+ | 6 commands below | ~1.2 GB, CPU-only PyTorch |
| **[C] Development** | [uv](https://docs.astral.sh/uv/) | 1 command | mirrors CI, more disk on Linux |
| **[D] GPU** *(optional)* | NVIDIA GPU + driver | `--gpus all --device cuda` | CPU-first model; GPU is only ~2.6x faster |

Whichever mode you choose, **run the command from the repository root** (the folder containing
`pocket_tts/`).

---

## 2. Start the app

### [A] Docker

```bash
git clone https://github.com/OlivierLAVAUD/Pocket_TTS.git
cd Pocket_TTS
docker compose up -d          # builds the image the first time (a few minutes)
docker compose logs -f pocket-tts-app
```

Wait for this line in the logs — it means the model is loaded and the server is up:

```
pocket_tts.app: Model ready on cpu (24000 Hz, voices: alba, anna, azelma, ...)
```

Open <http://localhost:7860>. The first start downloads the model weights (~440 MB, ~20 s); later
starts take about 6 seconds.

Custom run without compose:

```bash
docker build -f Dockerfile.app -t pocket-tts-app .
docker run --rm -p 7860:7860 \
  -v pocket-tts-cache:/root/.cache/pocket_tts \
  -v hf-cache:/root/.cache/huggingface \
  pocket-tts-app --language french
```

Mount both cache folders, otherwise the weights are downloaded again on every start.

### [B] Light install (Linux, smallest footprint)

```bash
git clone https://github.com/OlivierLAVAUD/Pocket_TTS.git
cd Pocket_TTS
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e . gradio soundfile
python pocket_tts.py
```

The CPU index avoids ~3 GB of unused CUDA wheels that `pip install torch` downloads on Linux.

### [C] Development mode (uv)

```bash
git clone https://github.com/OlivierLAVAUD/Pocket_TTS.git
cd Pocket_TTS
uv run --with gradio --with soundfile python pocket_tts.py
```

---

## 3. Use the web UI

Open <http://localhost:7860> and:

1. **Type your text** in the box. Long texts are split into sentences and spoken one chunk at a time.
2. **Pick a voice** from the list (27 voices ship with the model). You can also type a URL or a
   local file path, or record/upload a reference under **Reference audio** to clone a voice.
3. **Generate**. The audio appears with a play button and a download link.
4. **Advanced parameters** (optional): *frames after EOS* (pause length at the end) and
   *max tokens per chunk* (how many tokens before the text is split). Leave them at 0/50 unless you
   know why you change them.
5. **Examples** under the output box fill the text box for you.

> **Voice cloning needs a token.** Uploading a reference recording requires the gated model
> weights: accept the terms on
> [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) and export
> `HF_TOKEN`. Without it, the app works with the 27 built-in voices and shows a clear message if you
> try to upload a reference.

---

## 4. Use the API

Same server, same model, no extra cost. Base URL: `http://localhost:7860`.

**Check it is alive**

```bash
curl http://localhost:7860/api/health
# {"status":"healthy","device":"cpu","quantize":false,"sample_rate":24000,...}
```

**List the voices**

```bash
curl http://localhost:7860/api/voices
```

**Generate speech (WAV file)**

```bash
curl -X POST http://localhost:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world!", "voice": "alba"}' \
  -o speech.wav
```

**Generate speech from Python**

```python
import requests

response = requests.post(
    "http://localhost:7860/api/generate",
    json={"text": "Hello world!", "voice": "alba"},
)
with open("speech.wav", "wb") as audio_file:
    audio_file.write(response.content)
```

**Stream the audio while it is generated** (first bytes after ~200 ms, useful for a live
front-end):

```bash
curl -X POST http://localhost:7860/api/tts \
  -F 'text=Hello world!' -F 'voice=alba' -o speech.wav
```

| Endpoint | What it does |
| --- | --- |
| `GET /api/health` | model, device, sample rate |
| `GET /api/voices` | available voices |
| `POST /api/generate` | JSON body → WAV file (or base64 with `"response_format": "json"`) |
| `POST /api/tts` | multipart form → WAV streamed as it is generated |

JSON body fields of `/api/generate`: `text` (required), `voice`, `frames_after_eos`, `max_tokens`,
`response_format`.
---

## 5. Stop, restart, clean up

```bash
docker compose stop                 # stop, keep everything
docker compose up -d                # start again
docker compose down                 # remove the container
docker compose down -v              # also delete the model cache (re-download next time)
docker system df                    # see what uses disk
docker builder prune                # free space after builds
```

---

## 6. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `port is already allocated` | another program uses 7860: add `--port 8001` |
| Page stays empty although the log says `Model ready` | open the exact port printed in the log |
| "We could not download the weights for the model with voice cloning" | expected: set `HF_TOKEN` (see §3) or use a built-in voice |
| The first start is slow | it downloads ~440 MB of weights; later starts take ~6 s |
| Docker build fails on `lookup ghcr.io … permission denied` | DNS/Docker issue, see [`manuel.md`](./manuel.md) §5 |
| Disk full during `docker build` | the image is large (~20 GB); free space or use mode [B] |
| An error appears in the log | read `docker compose logs pocket-tts-app`, it names the cause |

---

## 7. Good to know

- **One model, one request at a time.** The server keeps a single model in memory and serializes
  the generations, because pocket-tts is not thread-safe. Run one container, not many.
- **No authentication.** Anyone who can reach the port can generate audio. Keep it on localhost
  (`-p 127.0.0.1:7860:7860`) or put a reverse proxy in front if you expose it.
- **CPU is the normal case.** On Linux, install PyTorch from the CPU index to avoid multi-GB CUDA
  downloads.
- **24 kHz mono WAV** in, 24 kHz mono WAV out, generated frame by frame at 12.5 fps.

More documentation: [`README.md`](./README.md) for the library, [`manuel.md`](./manuel.md) (French)
for every option and the full API reference, [`AGENTS.md`](./AGENTS.md) for the code layout.
`--with` adds Gradio without touching `uv.lock` (Gradio is intentionally not a declared dependency).