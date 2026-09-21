import { create } from "zustand";
import { persist } from "zustand/middleware";

import { getModel, modelsFor, providerOf } from "../catalog";
import type { Provider, Surface } from "../catalog/types";
import { browserStorage } from "./browser-storage";

/** Results one press of Generate produces. Models that do not carry a count of
    their own are submitted once per result — every unit is a real platform
    request — so the ceiling is deliberately small. */
export const MAX_BATCH = 4;

const FALLBACK: Record<Provider, string> = {
  remote: "seedance-2.5",
  local: "flux1-schnell",
};

type ActiveState = {
  surface: Surface;
  /** Which backend the picker is listing. Purely a filter: what actually
      routes a run is the chosen model's own `provider`. */
  provider: Provider;
  model: string;
  batch: number;
  setModel: (id: string) => void;
  setProvider: (provider: Provider) => void;
  setBatch: (count: number) => void;
};

export const useActive = create<ActiveState>()(
  persist(
    (set) => ({
      surface: "video",
      provider: "remote",
      model: "seedance-2.5",
      batch: 1,
      setModel: (id) => {
        const model = getModel(id);
        const provider = providerOf(model);
        set((state) =>
          state.model === model.id && state.surface === model.surface && state.provider === provider
            ? state
            : { model: model.id, surface: model.surface, provider },
        );
      },
      /* Switching backends keeps the surface and moves to that backend's first
         model for it, or to its first model of any surface — the local
         catalog has no video entry yet, and landing on an empty list would
         leave the composer pointing at nothing. */
      setProvider: (provider) =>
        set((state) => {
          if (state.provider === provider) return state;
          const sameSurface = modelsFor(provider, state.surface)[0];
          const next =
            sameSurface ??
            modelsFor(provider, state.surface === "image" ? "video" : "image")[0];
          if (!next) return state;
          return { provider, model: next.id, surface: next.surface };
        }),
      setBatch: (count) =>
        set((state) => {
          const batch = Math.min(MAX_BATCH, Math.max(1, Math.round(count)));
          return state.batch === batch ? state : { batch };
        }),
    }),
    {
      name: "openhiggsfield.active.v3",
      storage: browserStorage(),
      partialize: (state) => ({
        surface: state.surface,
        provider: state.provider,
        model: state.model,
        batch: state.batch,
      }),
      /* A stored model that has since left the catalog, or one whose provider
         no longer matches what was stored, must not leave the studio pointing
         at a model it cannot submit. */
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        try {
          state.setModel(getModel(state.model).id);
        } catch {
          state.setModel(FALLBACK[state.provider] ?? FALLBACK.remote);
        }
      },
    },
  ),
);
