import { Activity, Camera, Database, Film, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { EvidenceViewer } from "./components/EvidenceViewer";
import { Metric } from "./components/Metric";
import { Results } from "./components/Results";
import { SearchPanel } from "./components/SearchPanel";
import { UploadPanel } from "./components/UploadPanel";
import { VideoList } from "./components/VideoList";
import { api, type Camera as CameraType, type SearchResult, type Stats, type Video } from "./lib/api";

function App() {
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [stats, setStats] = useState<Stats>({ cameras: 0, videos: 0, clips: 0, processed_videos: 0 });
  const [answer, setAnswer] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const activeVideoId = useMemo(() => selected?.video_id ?? selectedVideo?.id ?? null, [selected, selectedVideo]);

  async function refresh() {
    try {
      const [cameraData, videoData, statData] = await Promise.all([api.cameras(), api.videos(), api.stats()]);
      setCameras(cameraData);
      setVideos(videoData);
      setStats(statData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không tải được dữ liệu.");
    }
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, []);

  async function handleSearch(query: string) {
    setLoadingSearch(true);
    setError("");
    try {
      const response = await api.search(query, 8);
      setAnswer(`${response.answer} (${response.latency_ms}ms)`);
      setResults(response.results);
      setSelected(response.results[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tìm kiếm thất bại.");
    } finally {
      setLoadingSearch(false);
    }
  }

  async function handleUpload(form: FormData) {
    setUploading(true);
    setError("");
    try {
      const video = await api.uploadVideo(form);
      setSelectedVideo(video);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload thất bại.");
    } finally {
      setUploading(false);
    }
  }

  async function handleProcess(video: Video) {
    setError("");
    try {
      await api.processVideo(video.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Xử lý video thất bại.");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <span className="brand-icon">
            <Camera size={24} />
          </span>
          <div>
            <h1>SafeCam AI</h1>
            <p>Trợ lý truy xuất sự kiện camera</p>
          </div>
        </div>

        <div className="metrics">
          <Metric icon={<Camera size={18} />} label="Camera" value={stats.cameras} />
          <Metric icon={<Film size={18} />} label="Video" value={stats.videos} />
          <Metric icon={<Database size={18} />} label="Clip" value={stats.clips} />
          <Metric icon={<Activity size={18} />} label="Đã xử lý" value={stats.processed_videos} />
        </div>

        <SearchPanel loading={loadingSearch} onSearch={handleSearch} />
        <UploadPanel cameras={cameras} uploading={uploading} onUpload={handleUpload} />
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>Dashboard phân tích video</h2>
            <p>Upload, tách clip, tạo embedding và hỏi bằng tiếng Việt.</p>
          </div>
          <button className="secondary-button compact" type="button" onClick={refresh}>
            <RefreshCw size={17} />
            Làm mới
          </button>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="main-grid">
          <div className="left-column">
            <Results answer={answer} results={results} selectedId={selected?.id} onSelect={setSelected} />
            <VideoList
              activeVideoId={activeVideoId}
              videos={videos}
              onSelect={(video) => {
                setSelectedVideo(video);
                setSelected(null);
              }}
              onProcess={handleProcess}
              onRefresh={refresh}
            />
          </div>
          <EvidenceViewer selected={selected} selectedVideo={selectedVideo} />
        </div>
      </section>
    </main>
  );
}

export default App;

