# Running Local AI Studio on your own GPU

Two routes. **Docker** is the shorter one and is what this page leads with;
[bare metal](#bare-metal) is below if you would rather manage the Python
environment yourself.

Neither route removes the host requirement: you need an NVIDIA driver new
enough for your card. On Blackwell (RTX 5090, `sm_120`) that means a driver
supporting CUDA 12.8 or newer.

---

## Docker

### 1. Host prerequisites

```bash
nvidia-smi                       # must work, and report your card
docker --version                 # 24+
docker compose version           # v2+
```

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
if you have not, then prove Docker can see the GPU **before** building
anything:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If that fails, nothing below will work. Fix it first.

### 2. Configure

```bash
git clone https://github.com/sekhar5031/open-higgsfield
cd open-higgsfield
git checkout claude/gracious-allen-itquv0
cp .env.docker.example .env
```

Every value has a working default. The two worth deciding now:

- **`AI_DATA`** — where weights and outputs live. The default is a Docker named
  volume, which avoids host permission problems. Set a host path (e.g.
  `AI_DATA=/AI`) if you want to control the location. Allow ~40 GB per model.
- **`PUBLIC_LOCAL_AI_URL`** — see [Reaching it from another machine](#reaching-it-from-another-machine).
  Leave it alone if you will browse from the Docker host itself.

### 3. Build and start

```bash
docker compose up -d --build
```

The backend image is large — around 8–10 GB, most of it the CUDA-enabled torch
wheels — and the first build takes a while. No model weights are in it.

```bash
docker compose ps                      # studio waits for local-ai to be healthy
docker compose exec local-ai python -m server.cli gpu
# device in use: cuda
#   [0] NVIDIA GeForce RTX 5090: 31.4 GB free of 31.8 GB
```

If that says `no CUDA device`, the toolkit is not wired up — back to step 1.

### 4. Install a model

Not baked into the image, so this is a one-off download into the volume:

```bash
docker compose exec local-ai python -m server.cli install flux1-schnell
docker compose exec local-ai python -m server.cli models
```

Start with **schnell**: Apache-2.0, ungated, four steps. `flux1-dev` is gated
and non-commercial — accept its licence on the model card, put `HF_TOKEN` in
`.env`, and `docker compose up -d` again before installing it.

### 5. Generate

Open `http://localhost:3000` and jump to [Your first image](#your-first-image).

### Everyday commands

| | |
| --- | --- |
| `docker compose logs -f local-ai` | watch a generation run |
| `docker compose exec local-ai python -m server.cli models` | what is installed |
| `docker compose run --rm local-ai pytest` | verify the image (75 tests, no GPU needed) |
| `docker compose restart studio` | after changing `.env` |
| `docker compose down` | stop; the volume and its weights survive |
| `docker compose down -v` | stop **and delete the weights** |

---

## Bare metal

### 1. Backend

Install torch **first**, from the CUDA 12.8 index. The `.[gpu]` extra would
otherwise pull a default-index build with no `sm_120` kernels.

```bash
cd local-ai
python3 -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[gpu,dev]'

python -m server.cli gpu          # prove the card is visible before downloading 34 GB
```

### 2. Paths and a model

```bash
cp .env.example .env              # set AI_ROOT to a disk with ~40 GB free
python -m server.cli install flux1-schnell
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

`AI_ROOT` defaults to `~/.local/share/open-higgsfield-local-ai`; nothing is
ever written inside the repository. If you point it at `/AI`, create it first
and make it writable by you.

### 3. Studio

In a second terminal, at the repository root:

```bash
pnpm install
cat > .env.local <<'EOF'
LOCAL_AI_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_LOCAL_AI_URL=http://127.0.0.1:8000
EOF
pnpm dev
```

Local mode needs neither `HF_API_BASE_URL` nor the Blob token. Leave them unset.

---

## Your first image

Open `http://localhost:3000`.

**The key modal opens on first load.** The stored default model is a cloud one,
so the studio asks for a platform key before it knows you want Local. Close it,
then:

1. Click the model name in the composer
2. Switch the segment from **Cloud** to **Local**
3. Pick **FLUX.1 schnell**
4. Type a prompt and press `⌘/Ctrl + Enter`

The first run pays the model load — tens of seconds off disk. After that the
pipeline stays resident, and four steps at 1024² should take a few seconds.
`docker compose logs -f local-ai` (or the uvicorn terminal) shows
`loading … → queued … → completed in Ns`.

---

## Reaching it from another machine

This is the thing most likely to catch you out. The browser fetches generated
images **directly** from the Local AI service — the studio only ever hands it a
URL. So `localhost` in `NEXT_PUBLIC_LOCAL_AI_URL` means *the browser's*
localhost, not the GPU box.

**SSH tunnel** — simplest, and everything stays bound to loopback:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 you@gpu-box
```

Browse `http://localhost:3000` and change no configuration at all.

**Or expose it** — trusted networks only; the API has no authentication:

```bash
# .env  (Docker)
PUBLIC_LOCAL_AI_URL=http://gpu-box.lan:8000
AI_CORS_ORIGINS=http://gpu-box.lan:3000
```

```bash
# bare metal: local-ai/.env
AI_HOST=0.0.0.0
AI_CORS_ORIGINS=http://gpu-box.lan:3000
# bare metal: .env.local at the repo root
LOCAL_AI_API_URL=http://127.0.0.1:8000          # Next → API, same machine
NEXT_PUBLIC_LOCAL_AI_URL=http://gpu-box.lan:8000 # browser → API
```

Then serve with `uvicorn … --host 0.0.0.0` and `pnpm dev -- -H 0.0.0.0`.

`AI_CORS_ORIGINS` matters for **downloads** specifically: `<img>` tags render
without it, but the gallery's download path uses `fetch`, which does not.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `docker run --gpus all …` fails | NVIDIA Container Toolkit is missing or the daemon was not restarted after installing it |
| `no kernel image is available for execution` | torch too old for `sm_120` — rebuild, or reinstall from the cu128 index |
| `server.cli gpu` says no CUDA device | driver/toolkit mismatch; `nvidia-smi` must work on the host first |
| Studio: "The Local AI service is not answering at …" | the API is down, or `LOCAL_AI_API_URL` points at the wrong place. In Docker it must be `http://local-ai:8000`, not `localhost` |
| `FLUX.1 [schnell] is not installed` | the install step has not finished — `python -m server.cli models` |
| Tiles stay grey, images never load | `NEXT_PUBLIC_LOCAL_AI_URL` is not reachable *from the browser* — see above |
| Downloads fail but images render | CORS: add your studio origin to `AI_CORS_ORIGINS` |
| "The GPU ran out of memory" | drop resolution to 768, `numImages` to 1, and check nothing else holds VRAM |
| Model list empty (bare metal) | uvicorn started from the wrong directory — it must be `local-ai/` |
| Install fails with 401/403 | a gated repository: accept its licence on the model card and set `HF_TOKEN` |

Logs are at `$AI_ROOT/logs/local-ai.log` (`/data/logs` inside the container),
and `http://localhost:8000/docs` is the live API browser.
