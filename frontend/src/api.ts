const BASE = "/api";

export interface Example {
  game_id: number;
  platform_game_id: string;
  played_at: string | null;
  ply: number;
  fen: string | null;
  san: string;
  best_move: string | null;
  best_move_san: string | null;
  delta_win_prob: number;
  clock_remaining_s: number | null;
  narration: string;
  narration_source: string;
}

export interface Puzzle {
  id: string;
  rating: number;
  popularity: number;
  url: string;
}

export interface WeaknessRow {
  motif: string;
  phase: string | null;
  time_bucket: string | null;
  player_rate: number;
  cohort_rate: number;
  ratio: number;
  p_value: number;
  q_value: number | null;
  significant: boolean;
  total_winprob_lost: number;
  n_observations: number;
  examples: Example[];
  puzzles: Puzzle[];
}

export interface Report {
  player: {
    platform: string;
    username: string;
    estimated_rating: number;
    games: number;
    last_synced_at: string | null;
  };
  severity_counts: Record<string, number>;
  weaknesses: WeaknessRow[];
}

export async function syncPlayer(platform: string, username: string) {
  const r = await fetch(`${BASE}/players/${platform}/${username}/sync`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function startAnalysis(platform: string, username: string) {
  const r = await fetch(`${BASE}/players/${platform}/${username}/analyze`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ job_id: string }>;
}

export function jobEvents(jobId: string): EventSource {
  return new EventSource(`${BASE}/jobs/${jobId}/events`);
}

export async function fetchReport(platform: string, username: string) {
  const r = await fetch(`${BASE}/players/${platform}/${username}/report`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<Report>;
}
