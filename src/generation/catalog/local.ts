import type { ModelEntry } from "./types";

/* Models the Local AI service can run on this machine's own GPU.

   These are NOT the hosted catalog with a flag flipped. A hosted "Wan 3.0" is
   an endpoint; the open-weight release of the same family is a different
   artifact with its own license, resolution ceiling and VRAM cost. Each entry
   here mirrors one row of local-ai/server/data/models.json by id, and nothing
   appears in both lists by accident.

   Slice 1 is text-to-image only: no roles are declared, so the studio never
   offers a media input a local model cannot yet accept. */

const LOCAL_ASPECT = ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"] as const;
/** The base edge, in pixels. Output area lands near base², whatever the ratio,
    so changing shape does not change how long a render takes. */
const LOCAL_RESOLUTION = ["512", "768", "1024", "1280", "1536"] as const;

function localImage(spec: {
  id: string;
  label: string;
  note: string;
  steps: { min: number; max: number; default: number };
  guidance: { min: number; max: number; default: number; step?: number };
  maxImages: number;
  /** A model whose ceiling is lower than the shared list declares its own,
      so the rail never offers a size the service will silently clamp. */
  resolutions?: readonly string[];
  resolution?: string;
}): ModelEntry {
  return {
    id: spec.id,
    provider: "local",
    surface: "image",
    label: spec.label,
    task: "text-to-image",
    note: spec.note,
    roles: {},
    settings: {
      aspectRatio: { type: "enum", values: LOCAL_ASPECT, default: "1:1" },
      resolution: {
        type: "enum",
        values: spec.resolutions ?? LOCAL_RESOLUTION,
        default: spec.resolution ?? "1024",
      },
      steps: { type: "range", ...spec.steps },
      guidance: { type: "range", step: spec.guidance.step ?? 0.5, ...spec.guidance },
      /* A count key, so the studio's batch control drives it natively and one
         request comes back with that many images. */
      numImages: { type: "range", min: 1, max: spec.maxImages, default: 1 },
    },
  };
}

/* First in the list and the one Local lands on: ~4 GB, fits in VRAM with no
   CPU offload, and proves the whole path in seconds. Old and not very good —
   that is fine, its job is to tell you the plumbing works before you spend
   34 GB and an evening finding out it does not. */
export const sd15Local = localImage({
  id: "sd15",
  label: "Stable Diffusion 1.5",
  note: "Validation model · OpenRAIL-M · ~4 GB · fits in VRAM, no offload",
  steps: { min: 10, max: 50, default: 25 },
  guidance: { min: 1, max: 12, default: 7.5 },
  maxImages: 4,
  resolutions: ["512", "640", "768", "1024"],
  resolution: "512",
});

export const fluxSchnellLocal = localImage({
  id: "flux1-schnell",
  label: "FLUX.1 schnell",
  note: "Apache-2.0 · ~34 GB on disk · ~24 GB VRAM",
  steps: { min: 1, max: 12, default: 4 },
  guidance: { min: 0, max: 5, default: 0 },
  maxImages: 4,
});

export const fluxDevLocal = localImage({
  id: "flux1-dev",
  label: "FLUX.1 dev",
  note: "Non-commercial licence · gated · ~34 GB on disk · ~24 GB VRAM",
  steps: { min: 8, max: 50, default: 28 },
  guidance: { min: 1, max: 10, default: 3.5 },
  maxImages: 4,
});

export const qwenImageLocal = localImage({
  id: "qwen-image",
  label: "Qwen-Image",
  note: "Apache-2.0 · ~57 GB on disk · runs with CPU offload, slowly",
  steps: { min: 10, max: 50, default: 30 },
  guidance: { min: 1, max: 10, default: 4 },
  maxImages: 2,
});

export const LOCAL_MODELS: readonly ModelEntry[] = [
  sd15Local,
  fluxSchnellLocal,
  fluxDevLocal,
  qwenImageLocal,
];
