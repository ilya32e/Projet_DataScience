.PHONY: help docker-build docker-up docker-down docker-logs docker-clean docker-rebuild docker-shell

help:
	@echo "Docker Commands:"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start all services (build and run)"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - View service logs"
	@echo "  make docker-clean     - Remove containers, volumes and images"
	@echo "  make docker-rebuild   - Rebuild images and restart services"
	@echo "  make docker-shell     - Open shell in streamlit container"

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "✓ Services started:"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - FastAPI: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-logs-streamlit:
	docker-compose logs -f streamlit

docker-logs-api:
	docker-compose logs -f api

docker-clean:
	docker-compose down -v
	docker system prune -f

docker-rebuild:
	docker-compose down
	docker-compose up --build -d
	@echo "✓ Services rebuilt and restarted"

docker-shell:
	docker exec -it retention_streamlit /bin/bash

docker-shell-api:
	docker exec -it retention_api /bin/bash

docker-test:
	docker-compose run --rm streamlit pytest

ps:
	docker-compose ps
