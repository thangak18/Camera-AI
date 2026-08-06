# AIC 2026 Video Retrieval System

Hệ thống AI tìm kiếm và truy xuất sự kiện trong video, được thiết kế và tối ưu riêng cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM năm 2026 (AIC 2026)**.

## Mục tiêu giải quyết theo đề thi AIC 2026

Hệ thống được thiết kế để giải quyết 3 dạng truy vấn chính của vòng sơ tuyển:

1. **Truy vấn dạng 1 (Textual KIS)**: Tìm kiếm sự kiện chính xác theo văn bản.
   - Đầu vào: Mô tả văn bản.
   - Đầu ra: `video_id`, `frame_id`.
2. **Truy vấn dạng 2 (Video Q&A)**: Tìm kiếm sự kiện và trả lời câu hỏi.
   - Đầu vào: Mô tả sự kiện và câu hỏi cụ thể.
   - Đầu ra: `video_id`, `frame_id`, `answer`.
3. **Truy vấn dạng 3 (TRAKE)**: Truy xuất và căn chỉnh sự kiện video theo thời gian.
   - Đầu vào: Chuỗi sự kiện có cấu trúc.
   - Đầu ra: `video_id` và danh sách các `frame_id` (semantic keyframes) tương ứng.

## Stack Công Nghệ

- **Backend**: FastAPI, SQLAlchemy, FFmpeg.
- **AI Models**: 
  - **Vector Search**: Sử dụng trực tiếp CLIP features (clip-ViT-B-32) do BTC cung cấp.
  - **Vision-Language Model (VLM)**: Xử lý dạng Q&A và hỗ trợ nhận diện chuỗi sự kiện TRAKE.
- **Frontend**: React, Vite, TypeScript, TailwindCSS.
- **Database**: PostgreSQL 16 + pgvector (Lưu trữ và query CLIP embeddings, objects, metadata).
- **Deploy**: Docker Compose.

## Sơ đồ kiến trúc (Theo chuẩn AIC 2026)

```mermaid
flowchart TB
    U["Người dùng / Đội thi"] --> W["React Dashboard<br/>Nhập truy vấn KIS, Q&A, TRAKE"]
    W -->|REST API /api| A["FastAPI Backend"]

    subgraph Data_Ingestion["Dữ liệu cung cấp (AIC 2026)"]
        V["Videos (.mp4)"]
        K["Keyframes (.jpg)"]
        O["Objects (Faster R-CNN .json)"]
        C["CLIP Features (.npy)"]
        M["Metadata (YouTube .json)"]
    end

    Data_Ingestion --> I["Data Ingestion Pipeline<br/>Đồng bộ & Chuẩn hóa"]
    I --> DB
    
    subgraph Backend["Backend & AI Services"]
        A --> Q["Query Engine<br/>Phân loại & Chuẩn hóa (KIS/Q&A/TRAKE)"]
        Q --> S["Search Service<br/>pgvector cosine search"]
        Q --> VLM["VLM Service<br/>Sinh câu trả lời & Căn chỉnh (TRAKE)"]
        S --> R["Result Formatter<br/>Tính R-Score mô phỏng"]
        VLM --> R
    end

    subgraph Data["Database Layer"]
        DB[("PostgreSQL + pgvector<br/>videos, keyframes, embeddings")]
    end

    R --> A
    A --> W
```

## Luồng xử lý Truy Vấn

```mermaid
sequenceDiagram
    participant User as User
    participant Web as React Web
    participant API as FastAPI
    participant Search as pgvector DB
    participant VLM as VLM Engine

    User->>Web: Nhập truy vấn (KIS / Q&A / TRAKE)
    Web->>API: POST /api/search
    API->>Search: Vector Search với CLIP features
    Search-->>API: Top-K Keyframes & Video candidates
    
    alt Nếu là Q&A
        API->>VLM: Đưa Keyframes + Câu hỏi vào VLM
        VLM-->>API: Trả về Answer
    else Nếu là TRAKE
        API->>VLM: Phân tích các phân đoạn video
        VLM-->>API: Danh sách các semantic keyframes
    end
    
    API-->>Web: Kết quả format chuẩn (video_id, frame_id, answer)
```

## Database (PostgreSQL + pgvector) hay Supabase?

Mặc định repo sử dụng PostgreSQL local trong Docker Compose với extension `pgvector`.
Có thể chuyển đổi dễ dàng sang Supabase bằng cách thay đổi `DATABASE_URL` trong file `.env`:

```bash
DATABASE_URL=postgresql+psycopg://postgres:<password>@<supabase-host>:5432/postgres
```

## Chạy nhanh bằng Docker

```bash
cp .env.example .env
# Cấu hình các đường dẫn đến dataset của BTC trong .env nếu cần
docker compose up --build
```

- Mở dashboard tại [http://localhost:8080](http://localhost:8080)
- Kiểm tra API tại [http://localhost:8000/api/health](http://localhost:8000/api/health)

## Chạy thủ công từng service

### Database:
```bash
docker compose up -d db
```

### Backend:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend:
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Ghi chú về Tích hợp Dữ Liệu AIC

- Thay vì tự tách frame và trích xuất đặc trưng bằng FFmpeg/Model từ đầu như các hệ thống thông thường, backend sẽ ưu tiên đọc trực tiếp từ các file `.npy` (CLIP features) và `.json` (Objects, Metadata) do BTC cung cấp để tiết kiệm thời gian và đảm bảo độ chính xác khi chấm điểm.
- API sẽ có luồng riêng để đánh giá (evaluate) theo công thức tính **R-Score** (Top-1, 5, 20, 50, 100) của BTC.

## GitHub

Repo đã được tạo trên account `thangak18`: [https://github.com/thangak18/Camera-AI](https://github.com/thangak18/Camera-AI).
