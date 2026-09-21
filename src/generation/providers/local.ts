import type { GenerationPlane } from "../catalog/types";
import { PlatformError } from "../platform";
import type { GenerationStatus, QueuedGeneration } from "../platform";
import type { GenerationProvider } from "./types";

/** Local job ids are namespaced so the batched status poll can route a mixed
    set of in-flight requests without the studio having to tell it which
    backend each one came from. The run log stores the namespaced id, so a
    reload resumes a local job as readily as a hosted one. */
export const LOCAL_PREFIX = "local:";

export function isLocalRequestId(requestId: string): boolean {
  return requestId.startsWith(LOCAL_PREFIX);
}

export type LocalConfig = {
  /** Where this Next server reaches the API. */
  baseUrl: string;
  /** Where the browser reaches it — the two differ under Docker or a tunnel,
      and the asset URLs written into history must be the browser's. */
  publicUrl: string;
  fetch?: typeof fetch;
};

/** Read once per call rather than at module scope: a server action is not a
    build step, and the operator may change the port without a rebuild. */
export function localConfig(): LocalConfig {
  const baseUrl = trim(process.env.LOCAL_AI_API_URL) ?? "http://127.0.0.1:8000";
  return {
    baseUrl,
    publicUrl: trim(process.env.NEXT_PUBLIC_LOCAL_AI_URL) ?? baseUrl,
  };
}

type LocalAsset = {
  url?: unknown;
  kind?: unknown;
  width?: unknown;
  height?: unknown;
};

type LocalJob = {
  job_id?: unknown;
  status?: unknown;
  assets?: unknown;
  error?: unknown;
  progress?: unknown;
};

export function createLocalProvider(config: LocalConfig = localConfig()): GenerationProvider {
  const baseUrl = config.baseUrl.replace(/\/$/, "");
  const publicUrl = config.publicUrl.replace(/\/$/, "");
  const fetchImpl = config.fetch ?? fetch;

  async function send(method: "GET" | "POST", path: string, body?: unknown): Promise<unknown> {
    const url = `${baseUrl}${path}`;
    let response: Response;
    try {
      response = await fetchImpl(url, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        ...(body ? { body: JSON.stringify(body) } : {}),
        cache: "no-store",
      });
    } catch (caught) {
      /* A refused connection is the single most likely failure here, and
         "fetch failed" tells the visitor nothing they can act on. */
      throw new PlatformError(503, {
        detail:
          `The Local AI service is not answering at ${baseUrl}. Start it with ` +
          "`cd local-ai && uvicorn server.main:app --port 8000`, or point " +
          `LOCAL_AI_API_URL somewhere else. (${describe(caught)})`,
      });
    }

    const payload = await readJson(response);
    if (!response.ok) throw new PlatformError(response.status, payload);
    return payload;
  }

  return {
    async submit(plane): Promise<QueuedGeneration> {
      const payload = asRecord(await send("POST", "/v1/generate", toLocal(plane)));
      const jobId = typeof payload.job_id === "string" ? payload.job_id : "";
      if (!jobId) throw new PlatformError(502, { detail: "Local AI response carried no job_id" });
      const requestId = `${LOCAL_PREFIX}${jobId}`;
      return {
        status: typeof payload.status === "string" ? payload.status : "queued",
        requestId,
        statusUrl: `${publicUrl}/v1/jobs/${jobId}`,
        cancelUrl: `${publicUrl}/v1/jobs/${jobId}/cancel`,
      };
    },

    async status(requestId): Promise<GenerationStatus> {
      const jobId = requestId.startsWith(LOCAL_PREFIX)
        ? requestId.slice(LOCAL_PREFIX.length)
        : requestId;
      if (!jobId) throw new PlatformError(400, { detail: "Missing job id" });
      const job = asRecord(await send("GET", `/v1/jobs/${encodeURIComponent(jobId)}`)) as LocalJob;
      return toStatus(job, requestId, publicUrl);
    },
  };
}

/** The studio's plane, in the Local AI API's own words. Only the settings the
    catalog declared are sent; the service clamps them to what the model can
    actually do and answers with the resolved values in the job's sidecar. */
export function toLocal(plane: GenerationPlane): Record<string, unknown> {
  const settings = plane.settings;
  const task = plane.model && taskFor(plane);
  return {
    model: plane.model,
    task,
    prompt: plane.prompt.text.trim(),
    settings: {
      ...(typeof settings.aspectRatio === "string" ? { aspectRatio: settings.aspectRatio } : {}),
      ...(typeof settings.resolution === "string" ? { resolution: settings.resolution } : {}),
      ...(typeof settings.steps === "number" ? { steps: settings.steps } : {}),
      ...(typeof settings.guidance === "number" ? { guidance: settings.guidance } : {}),
      ...(typeof settings.numImages === "number" ? { numImages: settings.numImages } : {}),
      ...(typeof settings.seed === "number" ? { seed: settings.seed } : {}),
    },
  };
}

function taskFor(plane: GenerationPlane): string {
  const hasStart = (plane.media.start ?? []).length > 0;
  if (hasStart) return "image-to-image";
  return "text-to-image";
}

/** Local job states, in the vocabulary `poll.ts` already terminates on.
    Anything unrecognised is passed through as non-terminal, so a new stage
    added to the service keeps the run spinning rather than failing it. */
export function toStatus(job: LocalJob, requestId: string, publicUrl: string): GenerationStatus {
  const raw = typeof job.status === "string" ? job.status : "unknown";
  const status = raw === "cancelled" ? "canceled" : raw;
  const assets = Array.isArray(job.assets) ? (job.assets as LocalAsset[]) : [];

  const images = assets
    .filter((asset) => asset.kind !== "video")
    .flatMap((asset) => (typeof asset.url === "string" ? [{ url: absolute(asset.url, publicUrl) }] : []));
  const video = assets.find((asset) => asset.kind === "video");
  const videoUrl = typeof video?.url === "string" ? absolute(video.url, publicUrl) : undefined;

  return {
    status,
    requestId,
    ...(images.length ? { images } : {}),
    ...(videoUrl ? { video: { url: videoUrl } } : {}),
    ...(typeof job.error === "string" && job.error ? { error: job.error } : {}),
  };
}

function absolute(url: string, publicUrl: string): string {
  return /^https?:\/\//i.test(url) ? url : `${publicUrl}${url.startsWith("/") ? "" : "/"}${url}`;
}

function trim(value: string | undefined): string | undefined {
  const text = value?.trim();
  return text ? text.replace(/\/$/, "") : undefined;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function describe(caught: unknown): string {
  if (caught instanceof Error) {
    const cause = (caught as { cause?: { code?: unknown } }).cause;
    const code = cause && typeof cause.code === "string" ? cause.code : null;
    return code ?? caught.message;
  }
  return String(caught);
}
