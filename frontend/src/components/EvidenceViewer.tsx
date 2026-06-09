import { FileText, Film, Tags } from "lucide-react";

import { assetUrl, type Clip, type SearchResult, type Video } from "../lib/api";

type Props = {
  selected?: SearchResult | Clip | null;
  selectedVideo?: Video | null;
};

export function EvidenceViewer({ selected, selectedVideo }: Props) {
  const clipUrl = selected ? assetUrl(selected.clip_url) : undefined;
  const videoUrl = selected ? assetUrl(selected.source_video_url) : selectedVideo ? assetUrl(selectedVideo.video_url) : undefined;
  const src = clipUrl ?? videoUrl;

  return (
    <section className="viewer">
      <div className="viewer-header">
        <span>
          <Film size={18} />
          <h2>Bằng chứng</h2>
        </span>
        {selected ? (
          <strong>
            {selected.start_second.toFixed(0)}s - {selected.end_second.toFixed(0)}s
          </strong>
        ) : null}
      </div>
      <div className="video-frame">
        {src ? (
          <video key={src} controls src={src} />
        ) : (
          <div className="video-placeholder">Chọn một kết quả để xem clip.</div>
        )}
      </div>
      <div className="evidence-detail">
        <div>
          <FileText size={17} />
          <p>{selected?.caption ?? selectedVideo?.original_filename ?? "Chưa chọn bằng chứng."}</p>
        </div>
        <div>
          <Tags size={17} />
          <p>{selected?.objects?.join(", ") || "Không có tag."}</p>
        </div>
      </div>
    </section>
  );
}

