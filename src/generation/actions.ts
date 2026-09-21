"use server";

import { cookies } from "next/headers";

import { getModel, parseSettings, providerOf } from "./catalog";
import type { GenerationPlane, ModelEntry } from "./catalog/types";
import {
  MissingCredentialsError,
  PLATFORM_KEY_COOKIE,
  PLATFORM_KEY_COOKIE_OPTIONS,
  decodeCredentials,
  encodeCredentials,
  parseCredentialInput,
} from "./credentials";
import type { QueuedGeneration, StatusResult } from "./platform";
import { createLocalProvider, isLocalRequestId } from "./providers/local";
import { createRemoteProvider } from "./providers/remote";
import type { GenerationProvider } from "./providers/types";

/* Every failure below crosses back as a value, never as a throw.

   Next redacts the message of an error thrown out of a server action in a
   production build — the client receives a digest and a generic sentence. All
   the work these messages do ("install it first", "the Local AI service is not
   answering at …, start it with …") is therefore lost in exactly the build
   people run, and visible only in `pnpm dev`. Returning them keeps the words. */
export type ActionResult<T> = { ok: true; value: T } | { ok: false; error: string };

function failed(caught: unknown): { ok: false; error: string } {
  return { ok: false, error: caught instanceof Error ? caught.message : String(caught) };
}

export async function savePlatformCredentials(data: unknown): Promise<ActionResult<null>> {
  try {
    const { apiKey } = parseCredentialInput(data);
    const jar = await cookies();
    jar.set(PLATFORM_KEY_COOKIE, encodeCredentials(apiKey), PLATFORM_KEY_COOKIE_OPTIONS);
    return { ok: true, value: null };
  } catch (caught) {
    return failed(caught);
  }
}

export async function clearPlatformCredentials() {
  const jar = await cookies();
  jar.set(PLATFORM_KEY_COOKIE, "", { ...PLATFORM_KEY_COOKIE_OPTIONS, maxAge: 0 });
}

export async function hasPlatformCredentials() {
  return (await readStoredCredentials()) !== null;
}

export async function submitGeneration(
  plane: GenerationPlane,
): Promise<ActionResult<QueuedGeneration>> {
  try {
    const model = getModel(plane.model);
    const parsed: GenerationPlane = {
      ...plane,
      settings: parseSettings(model, plane.settings),
    };
    return { ok: true, value: await (await providerFor(model)).submit(parsed) };
  } catch (caught) {
    return failed(caught);
  }
}

/** Every request in flight, answered in one round trip. Next dispatches server
    actions one at a time per client, so a poll per run would queue ahead of the
    next submit — the fan-out belongs on this side of the call, where it is
    genuinely parallel.

    A local run and a hosted one can be in flight together, so the ids are
    split by namespace first. The hosted client — and the key it needs — is
    built only if some id actually belongs to it: generating locally must not
    require a platform key. */
export async function getGenerationStatuses(data: unknown): Promise<StatusResult[]> {
  const requestIds = parseRequestIds(data);
  const local = requestIds.filter(isLocalRequestId);
  const remote = requestIds.filter((requestId) => !isLocalRequestId(requestId));

  const localProvider = local.length > 0 ? createLocalProvider() : null;
  /* A missing key must fail only the hosted requests. Reading it outside the
     per-request boundary would throw the whole round — and take every local
     run's poll down with it. */
  let remoteProvider: GenerationProvider | null = null;
  let remoteError: string | null = null;
  if (remote.length > 0) {
    try {
      remoteProvider = createRemoteProvider(await readCredentials());
    } catch (caught) {
      remoteError = caught instanceof Error ? caught.message : String(caught);
    }
  }

  return Promise.all(
    requestIds.map(async (requestId): Promise<StatusResult> => {
      const provider = isLocalRequestId(requestId) ? localProvider : remoteProvider;
      try {
        if (!provider) throw new Error(remoteError ?? "No provider for this request");
        return { requestId, status: await provider.status(requestId) };
      } catch (caught) {
        return { requestId, error: caught instanceof Error ? caught.message : String(caught) };
      }
    }),
  );
}

/** The model decides the backend. Nothing else does — there is no mode flag a
    local run could be routed around. */
async function providerFor(model: ModelEntry): Promise<GenerationProvider> {
  if (providerOf(model) === "local") return createLocalProvider();
  return createRemoteProvider(await readCredentials());
}

async function readStoredCredentials() {
  const jar = await cookies();
  return decodeCredentials(jar.get(PLATFORM_KEY_COOKIE)?.value);
}

async function readCredentials() {
  const stored = await readStoredCredentials();
  if (!stored) throw new MissingCredentialsError();
  const baseUrl = process.env.HF_API_BASE_URL;
  if (!baseUrl) throw new Error("Missing HF_API_BASE_URL");
  return { ...stored, baseUrl };
}

function parseRequestIds(data: unknown): string[] {
  const payload = asObject(data, "Invalid status payload");
  const requestIds = payload.requestIds;
  if (!Array.isArray(requestIds) || requestIds.length === 0) {
    throw new Error("Invalid request ids");
  }
  return requestIds.map((requestId) => {
    if (typeof requestId !== "string" || !requestId) throw new Error("Invalid request id");
    return requestId;
  });
}

function asObject(data: unknown, message: string): Record<string, unknown> {
  if (data === null || typeof data !== "object" || Array.isArray(data)) throw new Error(message);
  return data as Record<string, unknown>;
}
