export type Surface = "image" | "video";

/** Who executes a model. "remote" is the hosted generation API; "local" is the
    Local AI service on this machine. The field is on the entry rather than in
    a global mode flag so a local model structurally cannot reach a cloud API —
    the route is decided by what was picked, not by a switch that could be
    stale. */
export type Provider = "remote" | "local";
export type MediaRole = "start" | "end" | "reference" | "video" | "audio";

export type MediaItem = {
  id: string;
  url: string;
  role: MediaRole;
};

export type SettingField =
  | { type: "enum"; values: readonly string[]; default: string }
  | { type: "range"; min: number; max: number; default: number; step?: number }
  | { type: "boolean"; default: boolean };

export type PlatformPaths = {
  text?: string;
  image?: string;
  firstLast?: string;
  reference?: string;
};

export type ModelEntry = {
  id: string;
  surface: Surface;
  /** Absent means "remote": every pre-existing catalog entry is hosted. */
  provider?: Provider;
  label: string;
  roles: Partial<Record<MediaRole, number>>;
  settings: Record<string, SettingField>;
  /** Submit paths when the shared mapper is enough. Soul, Kling 3, and Seedance keep custom maps. */
  paths?: PlatformPaths;
  /** Local models only: the task the Local AI API routes on, and the note the
      picker shows about licence and footprint. */
  task?: string;
  note?: string;
};

export type GenerationPlane = {
  model: string;
  prompt: { text: string };
  media: Partial<Record<MediaRole, MediaItem[]>>;
  settings: Record<string, unknown>;
};
