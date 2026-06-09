.PHONY: docker-up docker-down docker-logs backend-dev frontend-dev test-backend build-frontend

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api web db

backend-dev:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev:
	cd frontend && npm install && VITE_API_BASE_URL=http://localhost:8000 npm run dev

test-backend:
	cd backend && PYTHONPATH=. pytest

build-frontend:
	cd frontend && npm install && npm run build

