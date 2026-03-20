COMPOSE=src/deploy/docker-compose.yml

up:
	docker compose -f $(COMPOSE) up -d --build

down:
	docker compose -f $(COMPOSE) down

ps:
	docker compose -f $(COMPOSE) ps

logs:
	docker compose -f $(COMPOSE) logs -f

test:
	@echo "Run service tests individually, e.g. cd src/gateway && uv run pytest"

lock:
	@for d in src/edge src/reid_worker src/inference_engine src/streaming src/gateway src/query_service; do \
		echo "Locking $$d"; \
		(cd $$d && uv lock); \
	done
