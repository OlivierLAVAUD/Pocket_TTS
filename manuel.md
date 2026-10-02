<!-- Copyright (c) 2026 oLV - Olivier LAVAUD — SPDX-License-Identifier: MIT -->

# Manuel — application Gradio + API Pocket TTS

Application web (interface Gradio) et API HTTP pour [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
servies par **un seul processus** et un **modèle chargé une fois** en mémoire.

| Fichier | Rôle |
| --- | --- |
| `pocket_tts.py` | L'application : UI Gradio + API JSON (fichier unique, racine du dépôt) |
| `tests/test_app.py` | 20 tests (modèle simulé, aucun téléchargement) |
| `Dockerfile.app` | Image Docker de l'application |
| `docker-compose.yaml` | Service `pocket-tts-app` (à côté du service `pocket-tts` existant) |
| `docker-bake.hcl` | Cible de build `pocket-tts-app` |

---

## 1. Démarrage rapide (local)

Gradio est une dépendance **optionnelle** (non installée par défaut, pour ne pas modifier `uv.lock`) :

```bash
# depuis un checkout du dépôt
uv run --with gradio --with soundfile python pocket_tts.py --language french

# si pocket-tts et gradio sont déjà installés dans votre environnement
python pocket_tts.py --host 0.0.0.0 --port 7860
```

Puis ouvrez <http://localhost:7860>.

> **Nom du fichier.** `pocket_tts.py` est à côté du package `pocket_tts/`. Ce n'est pas un conflit :
> en Python, un dossier contenant `__init__.py` gagne toujours sur un module homonyme, donc
> `import pocket_tts` et `python -m pocket_tts` continuent de désigner le package.

Lancement sans interface (API seule) :

```bash
python pocket_tts.py --no-ui --port 7860
```

Lien public temporaire (Gradio, sans les routes `/api/*`) :

```bash
python pocket_tts.py --share
```

---

## 2. Options et variables d'environnement

Chaque option a un équivalent variable d'environnement (pratique en conteneur).

| Option | Variable | Défaut | Description |
| --- | --- | --- | --- |
| `--host` | `POCKET_TTS_HOST` | `0.0.0.0` | Interface d'écoute |
| `--port` | `POCKET_TTS_PORT` | `7860` | Port |
| `--language` | `POCKET_TTS_LANGUAGE` | `english` | Modèle : `french`, `german_24l`, `english_drifting_26-09`… |
| `--config` | `POCKET_TTS_CONFIG` | – | YAML local, URL `https://` ou `hf://` (incompatible avec `--language`) |
| `--checkpoint` | `POCKET_TTS_CHECKPOINT` | – | Checkpoint d'entraînement `.pt` |
| `--voice` (répétable) | `POCKET_TTS_VOICE` (séparés par `:`) | – | Enregistrements de référence exposés dans l'UI et l'API |
| `--device` | `POCKET_TTS_DEVICE` | `auto` | `cpu`, `cuda` ou `auto` |
| `--quantize` | `POCKET_TTS_QUANTIZE=1` | désactivé | Quantification int8 (CPU uniquement) |
| `--temperature` | – | celui du modèle | Température d'échantillonnage |
| `--sampler-decode-steps` | – | `1` | Pas de décodage du flow |
| `--eos-threshold` | – | `-4.0` | Seuil de fin de séquence |
| `--max-tokens` | – | `50` | Tokens maximum par fragment de phrase |
| `--no-ui` | – | interface activée | Sert uniquement l'API JSON |
| `--share` | – | – | Lien Gradio public (pas d'API) |
| `--log-level` | `POCKET_TTS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`… |

Dans l'UI : texte, voix (prédéfinies ou fichier/URL `http(s)://` / `hf://`), audio de référence
(téléversement ou micro, prioritaire sur la voix), curseurs « frames after EOS » et « max tokens ».
Les voix possibles sont listées par `GET /api/voices` (27 voix prédéfinies).

---

## 3. API HTTP

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
{"default_voice": "estelle", "voices": ["alba", "anna", "..."], "local_voices": ["mon_enregistrement"]}
```

### `POST /api/generate` — JSON, réponse WAV ou base64

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `text` | `string` | *obligatoire* | Texte à synthétiser |
| `voice` | `string` | voix par défaut | Voix prédéfinie, chemin local, URL `http(s)://` / `hf://` |
| `frames_after_eos` | `int` (0–30) | modèle | Frames générées après la fin de séquence |
| `max_tokens` | `int` (10–200) | `50` | Taille maximale d'un fragment |
| `response_format` | `"wav"` ou `"json"` | `"wav"` | `json` renvoie l'audio en base64 |

```bash
curl -X POST http://localhost:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Bonjour le monde.", "voice": "estelle"}' -o speech.wav
```

```bash
# en JSON (base64)
curl -X POST http://localhost:7860/api/generate -H 'Content-Type: application/json' \
  -d '{"text": "Hello world.", "response_format": "json"}'
```

### `POST /api/tts` — multipart, WAV **diffusé**

Même forme que la route `/tts` de `pocket-tts serve` : le WAV est envoyé au fur et à mesure que le
décodeur Mimi produit les fragments (premiers octets en ~200 ms).

```bash
curl -X POST http://localhost:7860/api/tts \
  -F 'text=Bonjour, ceci est un test.' -F 'voice=estelle' -o flux.wav

# avec une voix à cloner depuis un fichier
curl -X POST http://localhost:7860/api/tts \
  -F 'text=Bonjour.' -F 'voice_wav=@ma_voix.wav' -o flux.wav
```

L'API Gradio est également exposée (bouton *Generate*, `api_name="generate"`).

---

## 4. Docker

```bash
# construction + lancement
docker build -f Dockerfile.app -t pocket-tts-app .
docker run --rm -p 7860:7860 -v hf-cache:/root/.cache/huggingface \
  pocket-tts-app --language french

# via compose (à côté du serveur `serve` existant)
docker compose up pocket-tts-app -d
docker compose logs -f pocket-tts-app

# build multi-plateforme avec docker-bake (cible pocket-tts-app)
docker buildx bake pocket-tts-app
```

Points importants :

- **Caches** : montez `/root/.cache/huggingface` et `/root/.cache/pocket_tts` (volumes nommés dans
  `docker-compose.yaml`) pour ne pas retélécharger les poids (~440 Mo) à chaque démarrage.
- **Voix à cloner** : les poids `kyutai/pocket-tts` sont « gated ». Sans `HF_TOKEN`
  (`docker run -e HF_TOKEN=hf_...`, ou `hf auth login` en local), l'application fonctionne mais
  refuse tout fichier de référence avec le message amont « We could not download the weights for
  the model with voice cloning… » (code HTTP 400).
- **Enregistrement de voix** : montez un dossier, puis
  `-e POCKET_TTS_VOICE=/voices/ma_voix.wav` (plusieurs séparés par `:`).
- **Taille de l'image** : `uv.lock` référence 43 paquets `nvidia-*` (torch provient de PyPI avec le
  build CUDA), donc l'image est volumineuse. Pour une image CPU légère, réinstallez torch depuis
  `https://download.pytorch.org/whl/cpu` après le `uv sync`, ou forcez `UV_TORCH_BACKEND=cpu`.
- **Timeouts** : uv limite chaque téléchargement HTTP à 30 s ; pour de gros wheels, ajouter
  `ENV UV_HTTP_TIMEOUT=300` dans `Dockerfile.app`.

---

## 5. Dépannage : `lookup ghcr.io … permission denied`

### Symptôme

`docker compose up -d` (ou `docker build`, ou le simple `docker pull`) échoue avant même de
construire quoi que ce soit :

```
ERROR [pocket-tts-app internal] load metadata for ghcr.io/astral-sh/uv:debian
target pocket-tts: failed to solve: ghcr.io/astral-sh/uv:debian: failed to resolve source
metadata for ghcr.io/astral-sh/uv:debian: failed to do request:
Head "https://ghcr.io/v2/astral-sh/uv/manifests/debian":
dial tcp: lookup ghcr.io on 127.0.0.53:53: write udp 127.0.0.1:53110->127.0.0.53:53:
write: permission denied
```

Les **deux** services sont concernés (`pocket-tts` et `pocket-tts-app`) : ce n'est pas un problème
de Dockerfile.

### Cause

Deux particularités de la machine se combinent :

1. **Docker est installé en snap** (`Docker Root Dir: /var/snap/docker/common/var-lib-docker`).
   Le démon `dockerd` tourne confiné : son propre namespace réseau et des règles AppArmor/seccomp.
2. **La résolution DNS de l'hôte passe par le stub de systemd-resolved** :
   `/etc/resolv.conf` est un lien vers `/run/systemd/resolve/stub-resolv.conf`, qui ne contient
   qu'un `nameserver 127.0.0.53` (le resolver de l'hôte, joignable uniquement depuis l'hôte).

Le démon snap envoie donc sa requête UDP vers `127.0.0.53` **depuis son namespace** : l'écriture est
refusée (`permission denied`). Résultat : le démon ne résout **aucun** nom, donc aucun registre
(`ghcr.io`, `registry-1.docker.io`…) n'est joignable, et la ligne `FROM` échoue dès l'étape
« load metadata ».

### Le diagnostiquer en 30 secondes

```bash
ls -l /etc/resolv.conf                                  # -> ../run/systemd/resolve/stub-resolv.conf
grep nameserver /etc/resolv.conf                        # -> nameserver 127.0.0.53
docker info | grep 'Docker Root Dir'                    # -> /var/snap/docker/... => snap
getent hosts ghcr.io                                   # l'hôte, lui, résout (140.82.121.34)
docker run --rm alpine:latest getent hosts ghcr.io     # le démon, lui, échoue
```

Si la dernière commande renvoie la même erreur `permission denied` vers `127.0.0.53:53`, c'est
bien ce cas.

### Correctif A (recommandé) — pointer `resolv.conf` vers les vrais serveurs DNS

```bash
sudo ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf
sudo snap restart docker
docker pull alpine:latest            # test : doit afficher "Downloaded"
```

`/run/systemd/resolve/resolv.conf` contient les serveurs DNS réels de la machine (ici `192.168.1.1`,
visible via `resolvectl status`). Le démon snap peut alors les joindre.

Réversion :

```bash
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
sudo snap restart docker
```

### Correctif B — DNS explicite pour le démon, sans toucher à l'hôte

```bash
sudo cp /var/snap/docker/current/config/daemon.json /var/snap/docker/current/config/daemon.json.bak
echo '{ "log-level": "error", "dns": ["192.168.1.1", "1.1.1.1"] }' \
  | sudo tee /var/snap/docker/current/config/daemon.json
sudo snap restart docker
docker pull alpine:latest
```

L'option `dns` s'adresse surtout aux conteneurs ; selon la version du snap, le démon peut continuer
à utiliser `127.0.0.53` pour ses propres requêtes. Si le test échoue, passez au correctif A.

### Correctif C — remplacer le snap par le paquet officiel

```bash
sudo snap remove docker
# puis installer docker-ce depuis https://docs.docker.com/engine/install/ubuntu/
```

Docker via apt n'est pas confiné : plus aucun problème de DNS de ce type.

### Alternative ponctuelle : réseau hôte pendant le build

```bash
docker build --network=host -f Dockerfile.app -t pocket-tts-app .
```

Le build se fait alors dans le namespace réseau de l'hôte, où `127.0.0.53` est joignable. Utile
pour un test rapide, mais `docker pull` / `docker run` classiques resteront cassés : préférez A.

### Une fois le DNS réparé

```bash
docker compose up -d
docker compose logs -f pocket-tts-app    # attendre « Model ready on cpu (24000 Hz, …) »
curl http://localhost:7860/api/health

---

## 6. Limites connues

- **Pas de Voice Cloning sans `HF_TOKEN`** : les poids `kyutai/pocket-tts` sont gated. Sans
  jeton, l'app démarre normalement mais tout fichier de référence renvoie HTTP 400 avec le message
  amont. Les voix prédéfinies (états safetensors précalculés) fonctionnent sans jeton.
- **Requêtes sérialisées** : le modèle est partagé et pocket-tts n'est pas thread-safe ; un
  `threading.Lock` sérialise les générations, et la file Gradio est configurée avec
  `concurrency_limit=1`.
- **En-tête du WAV diffusé** : sur `POST /api/tts`, le flux n'est pas *seekable*, donc le nombre de
  frames de l'en-tête est un placeholder — comportement identique à la route `/tts` de
  `pocket-tts serve`. `/api/generate` renvoie un WAV complet, correctement renseigné.
- **Bande passante = 1, batch = 1** : hérité du modèle amont, aucun batching.
- **Textes longs** : le texte est découpé en phrases puis généré fragment par fragment, sans
  teacher forcing (même limitation que la bibliothèque).
- **Pas de reprise sur erreur GPU** : une génération qui échoue remonte une erreur HTTP 500 ; le
  serveur reste vivant.

---

## 7. Tests et vérifications effectuées

```bash
# tests de l'application (20 tests, modèle simulé, aucun téléchargement)
uv run pytest tests/test_app.py -v

# suite complète (télécharge le modèle et les voix)
uv run pytest -n 3 -v

# lint
uvx ruff@0.12.7 check pocket_tts.py tests/test_app.py
uvx ruff@0.12.7 format --check pocket_tts.py tests/test_app.py
```

| Vérification | Résultat |
| --- | --- |
| 20 tests de `tests/test_app.py` | 20 passés |
| `pytest tests/` complet | 124 passés ; 1 échec préexistant (`test_generate_with_custom_voice`, `skipif(CI)` en amont : nécessite les poids gated) |
| Ruff check / format / imports (0.12.7) | propres |
| `uv lock --check` | lock inchangé et cohérent |
| Serveur réel (modèle chargé) | `/`, `/api/health`, `/api/voices`, `/api/generate` (WAV + base64), `/api/tts` (streaming), `/gradio_api/call/generate` : OK |
| Audio produit | 24 kHz mono, WAV valide, ~1,3–1,6× plus rapide que le temps réel sur CPU |
| Mode `--no-ui` | `/` renvoie 404, l'API répond |
| `docker compose config` | valide (le build complet reste à faire une fois le DNS réparé) |
```