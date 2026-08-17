import { Camera, PlayCircle, UploadCloud } from "lucide-react";
import { FormEvent, useState } from "react";

import type { Camera as CameraType } from "../lib/api";

type Props = {
  cameras: CameraType[];
  uploading: boolean;
  onUpload: (form: FormData) => void;
};

export function UploadPanel({ cameras, uploading, onUpload }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState("");
  const [location, setLocation] = useState("Cong chinh");
  const [recordedAt, setRecordedAt] = useState("");
  const [processNow, setProcessNow] = useState(false);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;

    const form = new FormData();
    form.set("file", file);
    if (cameraId) form.set("camera_id", cameraId);
    if (location) form.set("location", location);
    if (recordedAt) form.set("recorded_at", recordedAt);
    form.set("process_now", String(processNow));
    onUpload(form);
  }

  return (
    <section className="panel upload-panel">
      <div className="panel-title">
        <UploadCloud size={18} />
        <h2>Upload video</h2>
      </div>
      <form onSubmit={submit}>
        <label className="file-input">
          <input
            accept="video/*"
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <UploadCloud size={20} />
          <span>{file ? file.name : "Chọn file camera"}</span>
        </label>

        <label>
          <span>Camera</span>
          <select value={cameraId} onChange={(event) => setCameraId(event.target.value)}>
            <option value="">Camera mặc định</option>
            {cameras.map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.name} - {camera.location}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Vị trí</span>
          <input value={location} onChange={(event) => setLocation(event.target.value)} />
        </label>

        <label>
          <span>Thời gian ghi hình</span>
          <input
            type="datetime-local"
            value={recordedAt}
            onChange={(event) => setRecordedAt(event.target.value)}
          />
        </label>

        <label className="toggle-row">
          <input
            checked={processNow}
            type="checkbox"
            onChange={(event) => setProcessNow(event.target.checked)}
          />
          <span>Xử lý đồng bộ</span>
        </label>

        <button className="secondary-button" disabled={uploading || !file} type="submit">
          {processNow ? <PlayCircle size={18} /> : <Camera size={18} />}
          {uploading ? "Đang upload" : "Đưa vào pipeline"}
        </button>
      </form>
    </section>
  );
}

