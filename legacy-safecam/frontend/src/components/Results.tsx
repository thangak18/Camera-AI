import { Clock, MapPin, ShieldCheck } from "lucide-react";

import { assetUrl, type SearchResult } from "../lib/api";

type Props = {
  answer?: string;
  results: SearchResult[];
  selectedId?: string | null;
  onSelect: (result: SearchResult) => void;
};

export function Results({ answer, results, selectedId, onSelect }: Props) {
  return (
    <section className="results-shell">
      <div className="result-answer">
        <ShieldCheck size={20} />
        <p>{answer || "Upload video, chờ pipeline tạo clip, rồi nhập truy vấn để xem bằng chứng."}</p>
      </div>
      <div className="results-grid">
        {results.length === 0 && <p className="empty wide">Chưa có kết quả truy vấn.</p>}
        {results.map((result) => (
          <button
            className={`result-card ${result.id === selectedId ? "active" : ""}`}
            key={result.id}
            type="button"
            onClick={() => onSelect(result)}
          >
            <img alt={result.caption} src={assetUrl(result.thumbnail_url) ?? ""} />
            <span className="result-body">
              <strong>{result.caption}</strong>
              <small>
                <MapPin size={14} />
                {result.camera_name ?? result.camera_id} | {result.location ?? "unknown"}
              </small>
              <small>
                <Clock size={14} />
                {result.start_second.toFixed(0)}s - {result.end_second.toFixed(0)}s
              </small>
              <em>{result.reason}</em>
            </span>
            <span className="score">{Math.round(result.score * 100)}%</span>
          </button>
        ))}
      </div>
    </section>
  );
}

