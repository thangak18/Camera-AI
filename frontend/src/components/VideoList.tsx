import { RefreshCw, RotateCcw, Video as VideoIcon } from "lucide-react";

import type { Video } from "../lib/api";

type Props = {
  videos: Video[];
  activeVideoId?: string | null;
  onSelect: (video: Video) => void;
  onProcess: (video: Video) => void;
  onRefresh: () => void;
};

export function VideoList({ videos, activeVideoId, onSelect, onProcess, onRefresh }: Props) {
  return (
    <section className="panel video-list">
      <div className="panel-title between">
        <span>
          <VideoIcon size={18} />
          <h2>Video đã nạp</h2>
        </span>
        <button className="icon-button" type="button" onClick={onRefresh} title="Làm mới">
          <RefreshCw size={17} />
        </button>
      </div>
      <div className="video-rows">
        {videos.length === 0 && <p className="empty">Chưa có video nào.</p>}
        {videos.map((video) => (
          <div
            className={`video-row ${video.id === activeVideoId ? "active" : ""}`}
            key={video.id}
            onClick={() => onSelect(video)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") onSelect(video);
            }}
            role="button"
            tabIndex={0}
          >
            <span>
              <strong>{video.original_filename}</strong>
              <small>
                {video.camera_name ?? video.camera_id} | {video.location ?? "unknown"} |{" "}
                {video.clip_count} clip
              </small>
            </span>
            <em data-status={video.status}>{video.status}</em>
            {video.status === "uploaded" || video.status === "failed" ? (
              <button
                className="tiny-button"
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onProcess(video);
                }}
                title="Xử lý lại"
              >
                <RotateCcw size={14} />
              </button>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
}
