# Local AI — native inference for the OpenHiggsfield studio

A FastAPI service that downloads open-weight models and runs them **directly**
through their own official implementations (Diffusers / Transformers /
PyTorch), on your GPU.

```
OpenHiggsfield Studio  →  Local AI API  →  Model Router  →  Job / GPU Manager
                       →  Native Engines  →  PyTorch / CUDA  →  your GPU
```

**No ComfyUI.** Not as a dependency, a subprocess, a workflow format or a
hidden backend. `engines/image/diffusers_t2i.py` is the whole inference path:
`from_pretrained` on a local snapshot, then call the pipeline.

---

## Docker

The short route — see [../docs/RUNNING.md](../docs/RUNNING.md). Needs the
NVIDIA Container Toolkit on the host.

```bash
cd ..                             # compose lives at the repository root
cp .env.docker.example .env
docker compose up -d --build
docker compose exec local-ai python -m server.cli install flux1-schnell
docker compose run --rm local-ai pytest      # verify the image
```

No weights are baked into the image; they download at runtime into a volume.

## Install (bare metal)

Python 3.10+. Torch is installed separately because the right build depends on
your card — an RTX 5090 (Blackwell, `sm_120`) needs CUDA 12.8 or newer.

```bash
cd local-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[gpu,dev]'
```

Check the card is visible before downloading 34 GB:

```bash
python -m server.cli gpu
# device in use: cuda
#   [0] NVIDIA GeForce RTX 5090: 31.4 GB free of 31.8 GB
```

## Configure

Copy `.env.example` and set at least `AI_ROOT`. Nothing is written inside the
repository; if `AI_ROOT` is unset everything lands under
`~/.local/share/open-higgsfield-local-ai`.

```
AI_ROOT=/AI            → /AI/models  /AI/outputs  /AI/assets  /AI/cache  /AI/logs
```

Every path is separately overridable (`AI_MODEL_DIR`, `AI_OUTPUT_DIR`, …).

## Install a model and run

```bash
python -m server.cli models              # what is registered and what is on disk
python -m server.cli install flux1-schnell
python -m server.cli install flux1-dev   # gated: accept its licence, set HF_TOKEN

uvicorn server.main:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/docs` is the generated OpenAPI browser.

```bash
curl -X POST localhost:8000/v1/generate -H 'content-type: application/json' -d '{
  "model": "flux1-schnell",
  "task": "text-to-image",
  "prompt": "portrait of a beekeeper in a sunlit orchard, medium format film",
  "settings": {"aspectRatio": "4:3", "resolution": "1024", "steps": 4, "seed": 7}
}'
# {"job_id":"job_5f2…","status":"queued","queue_position":0}

curl localhost:8000/v1/jobs/job_5f2…
# {"status":"generating","progress":{"fraction":0.47,"step":2,"total_steps":4,…}}
# {"status":"completed","assets":[{"url":"/v1/assets/job_5f2…-0.png",…}]}
```

## API

| | |
| --- | --- |
| `GET /v1/system/health` | liveness |
| `GET /v1/system/gpu` | device, VRAM total/used/free, utilisation, driver |
| `GET /v1/system/info` | resolved paths, device, model counts, free disk |
| `GET /v1/models` | registry + install state + load state + disk usage. `?type=`, `?task=`, `?installed=` |
| `GET /v1/models/{id}` | one row |
| `POST /v1/models/{id}/install` | 202; resumable background download |
| `POST /v1/models/{id}/unload` | free VRAM, keep the weights |
| `DELETE /v1/models/{id}` | delete the weights |
| `POST /v1/generate` | 202 `{job_id, status}` |
| `GET /v1/jobs` · `GET /v1/jobs/{id}` | `queued → loading_model → generating → encoding → completed \| failed \| cancelled` |
| `POST /v1/jobs/{id}/cancel` | cooperative, at the next step boundary |
| `GET /v1/assets/{file}` | the generated media |

Failures all answer `{"detail": "…"}` with a status code, so the studio can
show the sentence verbatim.

## Adding a model

A row in `server/data/models.json` — no code:

```json
{
  "id": "some-model", "name": "Some Model", "type": "image",
  "tasks": ["text-to-image"], "repository": "ORG/MODEL",
  "engine": "diffusers-t2i", "pipeline": "SomePipeline",
  "size_gb": 20, "vram_gb": 16, "ram_gb": 32,
  "license": "Apache-2.0", "license_url": "https://huggingface.co/ORG/MODEL",
  "capabilities": { "steps": {"min": 10, "max": 50, "default": 28} }
}
```

Point `AI_MODEL_REGISTRY` at a second JSON file to add or override rows without
editing the repository. Verify the licence on the model card first —
`license` and `license_url` are recorded, and `gated: true` makes install
refuse until `HF_TOKEN` is set.

A new **family** needs an engine: subclass `engines/base.py:InferenceEngine`
and register the key in `engines/loader.py:BUILTIN`.

## Design

- **One worker.** Two diffusion jobs sharing 32 GB finish later than the same
  two in sequence, and can OOM each other. The queue is the correct shape for
  one card; one worker per device is the multi-GPU extension point.
- **Admission before allocation.** The GPU manager refuses a load whose
  declared VRAM exceeds what is free (after LRU eviction), naming both numbers.
  A CUDA OOM twelve seconds into sampling tells nobody anything.
- **Offloading is data.** `offload: "model"` keeps one component on the GPU at
  a time, which is what lets a 34 GB pipeline run in 32 GB.
- **Reproducibility is a file.** Every output gets a `.png.json` sidecar with
  the model, repository, prompt, resolved dimensions, steps, guidance and the
  seed that actually produced it.
- **Lazy imports.** torch and diffusers are imported inside the engine, so the
  API, the registry and the tests run anywhere.

## Tests

```bash
pytest          # 75 tests, no GPU and no weights required
```

A fake engine is registered behind the real engine key, so routing, admission,
the LRU, the queue, progress, cancellation, storage and the whole HTTP surface
are exercised for real. Only the two dozen lines that call into Diffusers are
stubbed.

## Roadmap

Image → **video** (Wan, LTX) → **3D** (TRELLIS) → **audio** (Whisper, TTS) →
**LLM**. Each is a new engine behind the same contract. Local media *inputs*
(img2img, start frames) need `POST /v1/assets`, which is the first item of the
video slice.
