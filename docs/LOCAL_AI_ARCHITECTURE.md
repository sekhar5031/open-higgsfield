# Local AI Studio — Architecture Assessment

Assessment of the existing OpenHiggsfield studio and the smallest change that
puts a local, self-hosted inference platform underneath it.

**Hard constraint: ComfyUI is not used anywhere.** No ComfyUI process, API,
workflow graph, node pack, or embedded runtime. Inference runs against the
model authors' own open-source implementations (Diffusers / Transformers /
PyTorch), loaded directly from `safetensors` weights in this repository's own
Python service.

---

## CURRENT ARCHITECTURE

Next.js 16 App Router, React 19, plain CSS, Zustand, pnpm. ~6.5k lines of
TypeScript plus a 3.7k-line stylesheet. One page, one composer, one gallery.

```
src/
  app/                    /  — the only page; /api/blob — upload tokens
  generation/             the generation domain
    catalog/              38 model entries + settings/role declarations
    stores/               active, prompt (x2), media (x2), settings
    actions.ts            "use server" — the ONLY caller of the remote API
    platform.ts           HTTP client for the remote generation API
    to-platform.ts        GenerationPlane -> {path, body} for that API
    plane.ts              stores -> GenerationPlane
    poll.ts               client-side batched status poller
    credentials.ts        api_key cookie encode/decode/validate
    device.ts             device cookie + blob pathname scoping
    upload.ts             client-direct upload to Vercel Blob
  openhiggsfield/         the studio surface (composer, gallery, viewer, …)
  proxy.ts                middleware: mints the device cookie
```

Three layers, cleanly separated already:

| Layer | Files | Knows about the provider? |
| --- | --- | --- |
| Studio UI | `src/openhiggsfield/*` | Only via `submitGeneration` / `watchRequest` and the "platform key" wording |
| Generation domain | `src/generation/*` | Yes — `platform.ts`, `to-platform.ts`, `credentials.ts` |
| Provider | remote HTTP API | — |

The UI never talks to the generation API. Every call crosses a server action.
That is the property that makes this change small.

---

## CURRENT GENERATION FLOW

```
Composer (Generate)
   │
   ▼
assemblePlane()                       src/generation/plane.ts
   reads useActive / usePrompt / useMedia / useSettings
   -> GenerationPlane { model, prompt:{text}, media:{role:[…]}, settings }
   │
   ▼
submitGeneration(plane)               src/generation/actions.ts   "use server"
   getModel(id) -> ModelEntry
   parseSettings(model, raw)          validates against the catalog allow-list
   toPlatform(plane) -> { path, body }
   createPlatformClient({apiKey, baseUrl}).submit(path, body)
   │
   ▼  POST {HF_API_BASE_URL}/{path}   Authorization: Key <id:secret>
   ◄  { request_id, status, status_url, cancel_url }
   │
   ▼
runningRows(requestId, …) -> RunRecord[] with status "running" -> history -> IndexedDB
   │
   ▼
watchRequest(requestId)               src/generation/poll.ts  (client, 4s interval)
   -> getGenerationStatuses({requestIds:[…]})   one server action for ALL in flight
   -> GET /requests/{id}/status
   ◄  { status, request_id, images:[{url}] | video:{url}, error }
   │
   ▼ terminal status ("completed" | "failed" | "nsfw" | "canceled")
terminalRows() -> RunRecord[] with urls -> replaceRequest() -> history -> gallery tile
```

Deadline 10 min (`POLL_DEADLINE_MS`), 3 consecutive failed rounds before the
watches are given up (`MAX_MISSES`). Running rows carry `requestId`, so a page
reload re-attaches a watch to anything still in flight.

**Batch.** `countSetting(model)` finds a native count key (`numImages`,
`batchSize`). If the model has one, one request returns N media. If not, the
studio submits N separate requests, one per result tile.

---

## CURRENT MODEL CATALOG

`src/generation/catalog/` — 38 entries (8 image, 30 video), one file per family,
assembled in `index.ts` as `MODELS`.

```ts
type ModelEntry = {
  id: string;                               // "flux-2", "seedance-2.5"
  surface: "image" | "video";
  label: string;
  roles: Partial<Record<MediaRole, number>>; // start/end/reference/video/audio + per-role cap
  settings: Record<string, SettingField>;    // enum | range | boolean, each with a default
  paths?: PlatformPaths;                     // text / image / firstLast / reference submit paths
};
```

The catalog is genuinely the source of truth: the picker, the settings rail, the
media-role rail, `describeModel()`, the batch control and server-side validation
all read it. Adding an entry needs no studio change.

**But every entry is a remote API path.** `paths` values such as
`"alibaba/wan-3.0/text-to-video"` and `"bytedance/seedance-2.0/fast"` are routes
on the remote provider, not model repositories. Thirteen entries (Soul, Kling,
Seedance) bypass `paths` entirely for bespoke mappers in `to-platform.ts`.

This is the point requirement 16 makes: **the catalog is a list of remote
products, not a list of downloadable weights.** "Wan 3.0" in the catalog is a
hosted endpoint; the open-weight Wan release is a different artifact with
different capabilities and its own license. The two lists must not be conflated.

---

## CURRENT API DEPENDENCIES

| Dependency | Where | Used for |
| --- | --- | --- |
| Remote generation API | `src/generation/platform.ts` | `POST /{model}` submit, `GET /requests/{id}/status` |
| `HF_API_BASE_URL` | `actions.ts:70` | Origin of the above. Server-only |
| `api_key` cookie (`id:secret`) | `credentials.ts` | `Authorization: Key …`. httpOnly, 30 days |
| Vercel Blob | `src/app/api/blob/route.ts`, `src/generation/upload.ts` | Media inputs become public URLs the API can fetch |
| `OPEN_HIGGSFIELD_READ_WRITE_TOKEN` | `api/blob/route.ts:22` | Blob read-write token |
| Result CDN | implicit | Result URLs are the provider's; history outlives them |

**Package dependencies** are clean — `next`, `react`, `zustand`,
`@tanstack/react-virtual`, `@vercel/blob`. No provider SDK.

---

## THE EXACT EXTERNAL API BOUNDARY (Step C)

The remote provider is reached from **exactly two call sites, in one file**:

```
src/generation/actions.ts:41   createPlatformClient(await readCredentials()).submit(path, body)
src/generation/actions.ts:53   client.status(requestId)
```

Everything provider-specific sits behind those two lines:

| File | Role at the boundary |
| --- | --- |
| `src/generation/platform.ts` | HTTP + response shape (`request_id`, `images[]`, `video`) |
| `src/generation/to-platform.ts` | `GenerationPlane` -> that API's field names |
| `src/generation/credentials.ts` | the `id:secret` key that API wants |
| `src/generation/catalog/*.paths` | that API's route strings |

And the shapes crossing the boundary back into the studio:

```ts
QueuedGeneration  = { status, requestId, statusUrl, cancelUrl }
GenerationStatus  = { status, requestId, images?: [{url}], video?: {url}, error? }
```

Those two types are the seam. `poll.ts`, `openhiggsfield-app.tsx`, `history.ts`,
the gallery and the viewer consume **only** these — none of them knows that
`request_id` came from an HTTP body or that the URL is on someone else's CDN.

> **A local backend that produces a `QueuedGeneration` and a `GenerationStatus`
> is indistinguishable from the remote one to every line of UI code.**

Secondary boundary: `src/generation/upload.ts` -> Vercel Blob. Only needed once
local models accept image/video inputs; the first slice is text-to-image.

---

## SMALLEST ARCHITECTURAL CHANGE (Step D)

Introduce a **provider interface** at exactly that seam, and make the provider
an intrinsic property of the model rather than a global mode flag.

```
                        ┌─────────────────────────────────┐
  Composer ──plane────► │ submitGeneration (server action) │
                        │   getModel(plane.model)          │
                        │   .provider === "local" ?        │
                        └──────────┬───────────┬──────────┘
                                   │           │
                     "remote"      │           │  "local"
                                   ▼           ▼
                        toPlatform + platform.ts    local.ts
                                   │           │
                                   ▼           ▼
                        remote generation API   Local AI API (FastAPI, :8000)
                                                    Model Router
                                                    Job / GPU Manager
                                                    Native engines (Diffusers)
                                                    PyTorch / CUDA -> RTX 5090
                                   │           │
                                   └─────┬─────┘
                                         ▼
                        QueuedGeneration / GenerationStatus   (unchanged)
                                         ▼
                        poll.ts -> history -> gallery -> viewer   (untouched)
```

Why provider-per-model rather than a global "Local Mode" switch:

- `submitGeneration` already calls `getModel(plane.model)`. Routing on a field
  of the entry it already has costs one branch and zero new plumbing.
- It makes requirement 17 structural: picking a local model *cannot* reach a
  cloud API, because the route is decided by the model, not by a flag that could
  be stale or spoofed.
- Requirement 16 falls out for free: `MODELS` gains a `provider` field, the
  picker filters on it, and the remote catalog is never mistaken for a list of
  installable weights.

The visible "Local Mode" control is then a **filter in the model picker**, not a
mode the generation path has to thread.

### Files to modify (7)

| File | Change | Size |
| --- | --- | --- |
| `src/generation/catalog/types.ts` | `provider?: "remote" \| "local"` on `ModelEntry`; `task` for local entries | ~6 lines |
| `src/generation/catalog/index.ts` | append `LOCAL_MODELS`; add `modelsFor(provider, surface)` | ~10 lines |
| `src/generation/actions.ts` | route submit/status through `providerFor(model)` | ~20 lines |
| `src/generation/stores/active.ts` | persist the picker's `provider` filter | ~15 lines |
| `src/openhiggsfield/openhiggsfield-app.tsx` | key gate applies to remote models only | ~6 lines |
| `src/openhiggsfield/model-picker.tsx` | Cloud / Local segmented filter | ~30 lines |
| `src/openhiggsfield/openhiggsfield.css` | styles for that segmented control | ~40 lines |

Not modified: `poll.ts`, `history.ts`, `gallery.tsx`, `viewer.tsx`,
`composer.tsx`, `settings.tsx`, `asset-picker.tsx`, `plane.ts`,
`parse-settings.ts`, `to-platform.ts`, `platform.ts`, `credentials.ts`,
`device.ts`, `api/blob/route.ts`. **The frontend is not rewritten.**

### Files to add

```
src/generation/providers/index.ts      provider selection by model
src/generation/providers/remote.ts     the existing path, extracted verbatim
src/generation/providers/local.ts      HTTP client for the Local AI API
src/generation/catalog/local/*.ts      local-compatible model entries

local-ai/                              the Python inference platform
  server/  main.py, config, routes/, services/, schemas/, data/models.json
  engines/ base.py, loader.py, image/diffusers_t2i.py
  tests/
```

---

## LOCAL AI PLATFORM

```
HTTP :8000
  GET  /v1/system/health          liveness
  GET  /v1/system/gpu             torch/NVML device report
  GET  /v1/models                 registry + installed state + disk usage
  GET  /v1/models/{id}
  POST /v1/models/{id}/install    async download job
  DELETE /v1/models/{id}          remove weights from disk
  POST /v1/generate               -> { job_id, status: "queued" }
  GET  /v1/jobs/{id}              -> queued|loading_model|generating|encoding|completed|failed|cancelled
  POST /v1/jobs/{id}/cancel
  GET  /v1/assets/{file}          generated output bytes
```

Single-GPU serial worker: one job executes at a time, the rest queue. That is
the correct model for one RTX 5090 — concurrent diffusion jobs on one device
thrash VRAM and finish later than they would in sequence. The queue is the
extension point for multi-GPU (one worker per device).

**Model lifecycle.** `NOT_LOADED -> LOADING -> READY -> RUNNING -> READY`, with
an LRU resident cache (default 1 model) that evicts before loading a model that
would not fit in free VRAM. The GPU manager refuses a load whose declared
`vram_gb` exceeds free VRAM plus what eviction can reclaim, with a message that
names both numbers rather than letting CUDA OOM mid-step.

**Storage**, all configurable, none inside the repo:

```
AI_ROOT=/AI            AI_MODEL_DIR=$AI_ROOT/models     weights (HF snapshots)
                       AI_OUTPUT_DIR=$AI_ROOT/outputs   generated media
                       AI_ASSET_DIR=$AI_ROOT/assets     uploaded inputs
                       AI_CACHE_DIR=$AI_ROOT/cache      HF hub cache
                       AI_LOG_DIR=$AI_ROOT/logs
```

Defaults land under `~/.local/share/open-higgsfield-local-ai` so a fresh clone
runs without `/AI` existing. Every output is written with a JSON sidecar
carrying model, prompt, resolved settings, seed and timings — reproducibility is
a file, not a log line.

### Initial model set

Registry entries are data (`server/data/models.json`), so a model is added
without touching code. Shipping with image only, per requirement 18.

| id | Repository | License | Weights (bf16) | Status |
| --- | --- | --- | --- | --- |
| `flux1-schnell` | `black-forest-labs/FLUX.1-schnell` | Apache-2.0 | ~24 GB | **default** — permissive, ungated, 4 steps |
| `flux1-dev` | `black-forest-labs/FLUX.1-dev` | FLUX.1 Non-Commercial | ~24 GB | gated on HF; needs `HF_TOKEN` and licence acceptance |
| `qwen-image` | `Qwen/Qwen-Image` | Apache-2.0 | ~41 GB | exceeds 32 GB in bf16 — enabled with CPU offload; quantised variants are the practical path |

Researched but **not implemented yet**, with the licence check that has to pass
before they are:

| Family | Candidate | Licence to verify | Slice |
| --- | --- | --- | --- |
| Video | Wan (Alibaba) | Apache-2.0 on recent releases | 2 |
| Video | LTX-Video (Lightricks) | LTX open weights licence, per-release | 2 |
| 3D | TRELLIS (Microsoft) | MIT on the code; weights licence is per-release | 3 |
| Speech | `faster-whisper` / `openai/whisper-large-v3` | MIT / Apache-2.0 | 4 |
| TTS | Qwen TTS or equivalent open-weight model | per-release | 4 |
| LLM | Qwen / Llama class via `transformers` or `llama.cpp` | per-release | 5 |

> Model availability, exact repository ids, sizes and licence terms move. Each
> row is verified against the model card at install time — the registry records
> `license` and `license_url`, and `POST /v1/models/{id}/install` refuses a
> gated repository until the user has accepted its terms on Hugging Face.

### Engine contract

```python
class InferenceEngine(ABC):
    def load_model(self, model: ModelSpec) -> None
    def unload_model(self) -> None
    def generate(self, request: GenerationRequest, report: ProgressFn) -> EngineResult
    def get_progress(self) -> Progress
    def cancel(self, job_id: str) -> None
```

`engines/image/diffusers_t2i.py` implements it against `diffusers` directly:
`FluxPipeline` / `DiffusionPipeline.from_pretrained(local_dir, torch_dtype=bfloat16)`,
`pipe(prompt, …, callback_on_step_end=…)` for progress and cooperative cancel.
Nothing generates a graph, launches a subprocess, or speaks to a node server.

---

## DELIVERY ORDER

1. **This slice** — Studio -> Local API -> FLUX -> RTX 5090 -> gallery, end to end.
2. Video (Wan, LTX) — the same engine contract, a frame encoder, longer jobs.
3. 3D (TRELLIS) — a new surface in the catalog and a mesh viewer.
4. Audio — Whisper in, TTS out.
5. LLM — prompt assistance.

Local media inputs (img2img, start frames) need a local replacement for the
Vercel Blob upload path — `POST /v1/assets` writing to `AI_ASSET_DIR`. That is
the first item of slice 2, not of this one.
