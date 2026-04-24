"""Shared dependencies initialised during app lifespan."""
from __future__ import annotations

from src.db.mongo_client import MongoQueryClient
from src.db.qdrant_client import QdrantQueryClient
from src.db.redis_client import RedisQueryCache
from src.services.nl_parser import NLQueryParser
from src.services.query_executor import QueryExecutor
from src.services.minio_urls import MinIOURLBuilder

# Set during lifespan
_mongo: MongoQueryClient | None = None
_qdrant: QdrantQueryClient | None = None
_redis: RedisQueryCache | None = None
_executor: QueryExecutor | None = None
_nl_parser: NLQueryParser | None = None
_minio_urls: MinIOURLBuilder | None = None


def init(mongo: MongoQueryClient, qdrant: QdrantQueryClient, redis: RedisQueryCache,
         executor: QueryExecutor, nl_parser: NLQueryParser, minio_urls: MinIOURLBuilder) -> None:
    global _mongo, _qdrant, _redis, _executor, _nl_parser, _minio_urls
    _mongo = mongo
    _qdrant = qdrant
    _redis = redis
    _executor = executor
    _nl_parser = nl_parser
    _minio_urls = minio_urls


def get_mongo() -> MongoQueryClient:
    assert _mongo is not None
    return _mongo

def get_qdrant() -> QdrantQueryClient:
    assert _qdrant is not None
    return _qdrant

def get_redis() -> RedisQueryCache:
    assert _redis is not None
    return _redis

def get_executor() -> QueryExecutor:
    assert _executor is not None
    return _executor

def get_nl_parser() -> NLQueryParser:
    assert _nl_parser is not None
    return _nl_parser

def get_minio_urls() -> MinIOURLBuilder:
    assert _minio_urls is not None
    return _minio_urls

