# Deploy Notes

- `reid_worker` is a background consumer and should not be exposed behind a public `Service` by default.
- The worker depends on Kafka, MongoDB, Qdrant, Redis, MinIO, and the inference service being reachable before startup.
- Use `src/deploy/docker-compose.yml` for local stack orchestration and `src/deploy/k8s/reid-worker.yaml` for Kubernetes deployment.
