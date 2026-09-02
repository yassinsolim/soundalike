export type CatalogTrack = {
  row: number;
  title: string;
  artist: string;
};

export type Seed = {
  title: string;
  artist: string;
  artworkUrl?: string;
};

export type Recommendation = {
  position: number;
  title: string;
  artist: string;
  deezerId?: number;
  bpm?: number;
  spotifyUrl?: string;
};

export type Vibe = {
  tempo?: string;
  tone?: string;
  dynamics?: string;
  lowEnd?: string;
};

export type RecommendationSet = {
  seed: Seed;
  vibe: Vibe;
  results: Recommendation[];
  method: string;
  indexVersion: string;
  librarySize: number;
};

export type SharedTrack = {
  trackId: string;
  title?: string;
  artist?: string;
  artworkUrl?: string;
};

export type FeedbackSelection = "good" | "mixed" | "off";

export type FeedbackReason =
  | "style"
  | "mood_energy"
  | "tempo"
  | "vocals_language"
  | "instruments_timbre";
