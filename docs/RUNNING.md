# Running Local AI Studio on your own GPU

Two routes. **Docker** is the shorter one and is what this page leads with;
[bare metal](#bare-metal) is below if you would rather manage the Python
environment yourself.

Neither route removes the host requirement: you need an NVIDIA driver new
enough for your card. On Blackwell (RTX 5090, `sm_120`) that means a driver
supporting CUDA 12.8 or newer. A driver reporting CUDA 13.x is fine — it runs
the CUDA 12.8 wheels.

**On Windows, read [Windows and Docker Desktop](#windows-and-docker-desktop)
first.** The prerequisites differ from Linux and two of the defaults matter
more there.

---

## Docker

### 1. Host prerequisites

```bash
nvidia-smi                       # must work, and report your card
docker --version                 # 24+
docker compose version           # v2+
```

On **Linux**, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and restart the daemon. On **Windows**, do not — Docker Desktop's WSL 2 engine
brings its own; see [below](#windows-and-docker-desktop).

Either way, prove Docker can see the GPU **before** building anything:

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

**Validate the plumbing with the small one first.** `sd15` is ~4 GB, fits
entirely in VRAM with no CPU offload, and answers in seconds — so a mistake
costs a minute rather than a 34 GB download and an evening.

```bash
docker compose exec local-ai python -m server.cli verify sd15   # checks, downloads nothing
docker compose exec local-ai python -m server.cli install sd15
```

Generate once (below) to prove the whole path works, then move up:

```bash
docker compose exec local-ai python -m server.cli verify           # every row
docker compose exec local-ai python -m server.cli install flux1-schnell
docker compose exec local-ai python -m server.cli models
```

`flux1-schnell` is Apache-2.0 and ungated but needs the WSL 2 memory above.
`flux1-dev` is gated and non-commercial — accept its licence on the model
card, put `HF_TOKEN` in `.env`, and `docker compose up -d` again first.

#### `verify` first, always

Registry rows are transcribed from model cards by hand, and model cards move —
repositories get renamed, relicensed or gated. `verify` makes one API call and
reports the real download size, the licence the repository actually states,
and whether it is gated:

```
ok   sd15   stable-diffusion-v1-5/stable-diffusion-v1-5   4.2 GB  creativeml-openrail-m
```

A `FAIL` row is a bug in the registry, not a dead end. Point
`AI_MODEL_REGISTRY` at a JSON file containing a corrected row with the same
`id` and it overrides the built-in one — no need to edit the repository.

### 5. Generate

Open `http://localhost:3000` and jump to [Your first image](#your-first-image).

### Everyday commands

| | |
| --- | --- |
| `docker compose logs -f local-ai` | watch a generation run |
| `docker compose exec local-ai python -m server.cli models` | what is installed |
| `docker compose run --rm local-ai pytest` | verify the image (91 tests, no GPU needed) |
| `docker compose restart studio` | after changing `.env` |
| `docker compose down` | stop; the volume and its weights survive |
| `docker compose down -v` | stop **and delete the weights** |

---

## Windows and Docker Desktop

Everything above applies, with four differences.

**Use the WSL 2 engine.** Docker Desktop → Settings → General → *Use the WSL 2
based engine*. GPU passthrough does not work in Windows-containers mode.
There is no NVIDIA Container Toolkit to install: the WSL 2 engine ships it, and
your Windows driver is what provides the GPU. Confirm with the `--gpus all`
check above before building.

**Leave `AI_DATA` alone.** The named-volume default puts weights on the WSL 2
virtual disk. Pointing it at a Windows path (`AI_DATA=D:/AI`) bind-mounts
across the Windows/Linux filesystem boundary, which is slow enough to be felt
on every 34 GB model load. Keep them in the volume; use
`docker compose cp` if you need a file out.

**Give WSL 2 the RAM.** This is the one most likely to catch you, and it is
not obvious. It does not apply to `sd15`, which fits in VRAM outright — one
more reason to validate with that first. FLUX.1 schnell is ~34 GB of bf16 weights, which does not fit in
32 GB of VRAM, so the engine runs it with model CPU offload: the full pipeline
stays resident in *host* memory and each component is paged onto the card as
it executes. WSL 2 defaults to **half** your system RAM, so even a 64 GB
machine hands the container 32 GB — just under what the pipeline needs, and
the symptom is savage paging or a kill rather than a clear error.

Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB      # >= 40GB for FLUX schnell; leave Windows 12-16GB
swap=16GB        # headroom during the load, not somewhere to run from
```

Then apply it and restart Docker Desktop:

```bash
wsl --shutdown
```

With 32 GB of system RAM you cannot give WSL 40 GB. FLUX schnell will still
load, but it will page against your swap file and be slow — a smaller or
quantised image model is the better fit until one is in the registry.

**Give WSL 2 the disk.** The image is ~8–10 GB and the first model is ~34 GB,
all inside the WSL 2 VHDX, which grows on demand. The WSL 2 backend has no
disk slider — the real limit is free space on whatever drive holds it.
Docker Desktop → Settings → Resources shows the location (typically
`%LOCALAPPDATA%\Docker\wsl`); make sure that drive has ~60 GB free, or use
the same panel to move the disk image to a roomier one.

*Resource Saver* is fine to leave on: it only idles the VM when no containers
are running, and these run with `restart: unless-stopped`.

**Git Bash rewrites arguments that look like paths.** Harmless for everything
in this guide, but if you run a command whose argument starts with `/` —
`docker compose exec local-ai /bin/bash` — MSYS turns it into a Windows path.
Prefix it:

```bash
MSYS_NO_PATHCONV=1 docker compose exec local-ai /bin/bash
```

**Line endings.** The repository now pins LF via `.gitattributes`. If you
cloned before that landed, Git for Windows will have written CRLF into your
working tree; re-normalise once (this discards uncommitted edits):

```bash
git pull
git rm --cached -r . && git reset --hard
```

**Watch your VRAM.** A desktop session holds real memory — browsers, overlays
and Steam were using 1.3 GB on the machine this was written for. FLUX schnell
wants ~24 GB plus headroom, so it fits in 32 GB comfortably, but opening a pile
of GPU-accelerated windows mid-run is a way to meet the OOM path. `nvidia-smi`
before a big run.

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
python -m server.cli verify sd15  # checks the row, downloads nothing
python -m server.cli install sd15 # ~4 GB validation model
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

`local-ai/.env` is read at startup, and real environment variables override
it. `AI_ROOT` defaults to `~/.local/share/open-higgsfield-local-ai`; nothing is
ever written inside the repository. If you point it at `/AI` (or `D:/AI` on
Windows), create it first and make it writable by you.

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
3. Pick **Stable Diffusion 1.5** (Local opens on it)
4. Type a prompt and press `⌘/Ctrl + Enter`

SD 1.5 is old and not very good. That is not the point: it is there to prove
the chain — studio → API → router → engine → GPU → gallery — before anything
large is downloaded. Once a picture lands, switch to FLUX.1 schnell.

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
| Windows: `--gpus all` not supported | Docker Desktop is not on the WSL 2 engine, or needs a restart after the driver update |
| Windows: "no space left on device" mid-download | the WSL 2 virtual disk is full — Docker Desktop → Settings → Resources |
| Windows: model loads are very slow | `AI_DATA` points at a Windows path; move it back to the named volume |
| Windows: the container dies during model load, or the host crawls | WSL 2 has too little RAM for the offloaded pipeline — raise `memory=` in `.wslconfig`, then `wsl --shutdown` |

Logs are at `$AI_ROOT/logs/local-ai.log` (`/data/logs` inside the container),
and `http://localhost:8000/docs` is the live API browser.
