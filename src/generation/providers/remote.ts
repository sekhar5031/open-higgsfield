import { createPlatformClient } from "../platform";
import { toPlatform } from "../to-platform";
import type { GenerationProvider } from "./types";

/** The hosted generation API, exactly as before — the mapping and the HTTP
    client are unchanged, only the call site moved behind the interface. */
export function createRemoteProvider(credentials: {
  apiKey: string;
  baseUrl: string;
}): GenerationProvider {
  const client = createPlatformClient(credentials);
  return {
    async submit(plane) {
      const { path, body } = toPlatform(plane);
      return client.submit(path, body);
    },
    status(requestId) {
      return client.status(requestId);
    },
  };
}
