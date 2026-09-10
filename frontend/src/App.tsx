import { useState } from "react";
import {
  fetchReport,
  jobEvents,
  Report,
  startAnalysis,
  syncPlayer,
  WeaknessRow,
} from "./api";
import Board from "./Board";

const MOTIF_LABELS: Record<string, string> = {
  __any_error__: "Overall error rate",
  hanging_piece: "Hanging pieces",
  fork: "Missed forks",
  pin: "Missed pins",
  skewer: "Missed skewers",
  back_rank: "Back-rank tactics",
  discovered_attack: "Discovered attacks",
  missed_mate: "Missed mates",
};

const BUCKET_LABELS: Record<string, string> = {
  normal: "normal time",
  lt60s: "under 60s on clock",
  scramble: "time scramble",
  unknown: "no clock data",
};

function pct(x: number, digits = 1) {
  return `${(100 * x).toFixed(digits)}%`;
}

export default function App() {
  const [platform, setPlatform] = useState("chesscom");
  const [username, setUsername] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number; status: string } | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadReport() {
    setError(null);
    try {
      setReport(await fetchReport(platform, username));
    } catch (e) {
      setError(String(e));
    }
  }

  async function onSync() {
    setBusy("Syncing games…");
    setError(null);
    try {
      const r = await syncPlayer(platform, username);
      setBusy(null);
      alert(`Fetched ${r.fetched} games, ${r.inserted} new`);
    } catch (e) {
      setBusy(null);
      setError(String(e));
    }
  }

  async function onAnalyze() {
    setError(null);
    try {
      const { job_id } = await startAnalysis(platform, username);
      setBusy("Analyzing…");
      const es = jobEvents(job_id);
      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        setProgress(data);
        if (data.status === "done") {
          es.close();
          setBusy(null);
          setProgress(null);
          loadReport();
        } else if (data.status === "error") {
          es.close();
          setBusy(null);
          setProgress(null);
          setError(data.error);
        }
      };
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="page">
      <header>
        <h1>
          Blunderless<span className="tagline">find the mistakes you keep repeating</span>
        </h1>
        <div className="controls">
          <select value={platform} onChange={(e) => setPlatform(e.target.value)}>
            <option value="chesscom">Chess.com</option>
            <option value="lichess">Lichess</option>
          </select>
          <input
            placeholder="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadReport()}
          />
          <button onClick={onSync} disabled={!username || !!busy}>
            Sync
          </button>
          <button onClick={onAnalyze} disabled={!username || !!busy}>
            Analyze
          </button>
          <button onClick={loadReport} disabled={!username || !!busy}>
            Report
          </button>
        </div>
      </header>

      {busy && (
        <div className="progress">
          <span>{busy}</span>
          {progress && progress.total > 0 && (
            <>
              <div className="bar">
                <div
                  className="fill"
                  style={{ width: `${(100 * progress.done) / progress.total}%` }}
                />
              </div>
              <span>
                {progress.status} — {progress.done}/{progress.total} games
              </span>
            </>
          )}
        </div>
      )}
      {error && <div className="error">{error}</div>}

      {report && (
        <>
          <section className="summary">
            <div className="stat">
              <b>{report.player.games}</b>
              <span>games analyzed</span>
            </div>
            <div className="stat">
              <b>~{report.player.estimated_rating}</b>
              <span>estimated rating</span>
            </div>
            {(["blunder", "mistake", "inaccuracy"] as const).map((s) => (
              <div className="stat" key={s}>
                <b>{report.severity_counts[s] ?? 0}</b>
                <span>{s}s</span>
              </div>
            ))}
          </section>

          <section className="weaknesses">
            <h2>Your weaknesses, ranked by what they cost you</h2>
            {report.weaknesses.length === 0 && (
              <p>No weaknesses computed yet — run Analyze first (and build the cohort baseline).</p>
            )}
            {report.weaknesses.map((w, i) => (
              <WeaknessCard key={i} w={w} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}

function WeaknessCard({ w }: { w: WeaknessRow }) {
  const [open, setOpen] = useState(false);
  const label = MOTIF_LABELS[w.motif] ?? w.motif;
  return (
    <div className={`card ${w.significant ? "significant" : ""}`}>
      <div className="card-head" onClick={() => setOpen(!open)}>
        <div>
          <b>{label}</b>
          <span className="context">
            {w.phase} · {BUCKET_LABELS[w.time_bucket ?? "unknown"] ?? w.time_bucket}
          </span>
        </div>
        <div className="numbers">
          <span className="ratio">{w.ratio.toFixed(1)}×</span>
          <span className="rates">
            you {pct(w.player_rate)} vs cohort {pct(w.cohort_rate)}
          </span>
          {w.significant ? (
            <span className="badge sig">q={w.q_value?.toFixed(3)}</span>
          ) : (
            <span className="badge">
              n={w.n_observations}{w.q_value != null ? `, q=${w.q_value.toFixed(2)}` : ""}
            </span>
          )}
          <span className="cost">
            ≈{w.total_winprob_lost.toFixed(1)} games' worth of win prob lost
          </span>
        </div>
      </div>
      {open && (
        <div className="card-body">
          <div className="examples">
            {w.examples.map((ex, j) => (
              <div className="example" key={j}>
                {ex.fen && <Board fen={ex.fen} bestMove={ex.best_move} />}
                <div className="example-info">
                  <p className="narration">{ex.narration}</p>
                  <p className="meta">
                    You played <b>{ex.san}</b>
                    {ex.best_move_san && (
                      <>
                        {" "}— better was <b>{ex.best_move_san}</b>
                      </>
                    )}
                    {" "}(−{pct(ex.delta_win_prob)})
                    {ex.clock_remaining_s != null && ` · ${Math.round(ex.clock_remaining_s)}s on clock`}
                  </p>
                </div>
              </div>
            ))}
          </div>
          {w.puzzles.length > 0 && (
            <div className="puzzles">
              <b>Train it:</b>
              {w.puzzles.map((p) => (
                <a key={p.id} href={p.url} target="_blank" rel="noreferrer">
                  puzzle {p.rating}
                </a>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
