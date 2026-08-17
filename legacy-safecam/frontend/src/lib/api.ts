export type Camera = {
  id: string;
  name: string;
  location: string;
  stream_url?: string | null;
  description?: string | null;
  created_at: string;
};

export type Video = {
  id: string;
  camera_id: string;
  camera_name?: string | null;
  location?: string | null;
  original_filename: string;
  status: "uploaded" | "processing" | "processed" | "failed" | string;
  duration_seconds?: number | null;
  recorded_at?: string | null;
  created_at: string;
  processed_at?: string | null;
  error_message?: string | null;
  video_url?: string | null;
  clip_count: number;
};

export type Clip = {
  id: string;
  video_id: string;
  camera_id: string;
  camera_name?: string | null;
  location?: string | null;
  start_second: number;
  end_second: number;
  start_time?: string | null;
  end_time?: string | null;
  caption: string;
  objects: string[];
  attributes: Record<string, unknown>;
  confidence: number;
  thumbnail_url?: string | null;
  clip_url?: string | null;
  source_video_url?: string | null;
};

export type SearchResult = Clip & {
  score: number;
  reason: string;
};

export type SearchResponse = {
  query: string;
  answer: string;
  results: SearchResult[];
  latency_ms: number;
};

export type Stats = {
  cameras: number;
  videos: number;
  clips: number;
  processed_videos: number;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function assetUrl(path?: string | null): string | undefined {
  if (!path) return undefined;
  if (path.startsWith("http")) return path;
  return `${API_BASE_URL}${path}`;
}

export const api = {
  cameras: () => request<Camera[]>("/api/cameras"),
  videos: () => request<Video[]>("/api/videos"),
  clips: (videoId?: string) => request<Clip[]>(videoId ? `/api/clips?video_id=${videoId}` : "/api/clips"),
  stats: () => request<Stats>("/api/stats"),
  search: (query: string, limit = 8) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, limit })
    }),
  uploadVideo: (form: FormData) =>
    request<Video>("/api/videos/upload", {
      method: "POST",
      body: form
    }),
  processVideo: (id: string) =>
    request<Video>(`/api/videos/${id}/process`, {
      method: "POST"
    })
};

