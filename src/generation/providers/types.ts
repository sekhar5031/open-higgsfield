import type { GenerationPlane } from "../catalog/types";
import type { GenerationStatus, QueuedGeneration } from "../platform";

/** The seam the studio was already written against.

    Everything above this — the poller, the run log, the gallery, the viewer —
    consumes only `QueuedGeneration` and `GenerationStatus`. A backend that
    produces those two shapes is indistinguishable from any other, which is why
    a local GPU can be dropped in without touching a line of UI. */
export type GenerationProvider = {
  submit(plane: GenerationPlane): Promise<QueuedGeneration>;
  status(requestId: string): Promise<GenerationStatus>;
};
