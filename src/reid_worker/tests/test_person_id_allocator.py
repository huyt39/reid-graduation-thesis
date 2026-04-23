import types

import redis

from src.persistence.redis_cache import RedisPersonIdAllocator


def test_redis_person_id_allocator_allocate_increments(monkeypatch):
    state = {"n": 0}

    class DummyRedis:
        def incr(self, key):
            assert key == "reid:seq:person_id"
            state["n"] += 1
            return state["n"]

        def close(self):
            return None

    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: DummyRedis())

    alloc = RedisPersonIdAllocator(url="redis://localhost:6379/0")
    assert alloc.allocate() == 1
    assert alloc.allocate() == 2
    assert alloc.allocate() == 3


def test_redis_person_id_allocator_uses_custom_key(monkeypatch):
    seen = {"key": None}

    class DummyRedis:
        def incr(self, key):
            seen["key"] = key
            return 10

        def close(self):
            return None

    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: DummyRedis())

    alloc = RedisPersonIdAllocator(url="redis://localhost:6379/0", key="custom:seq:key")
    assert alloc.allocate() == 10
    assert seen["key"] == "custom:seq:key"


def test_redis_person_id_allocator_close_calls_client_close(monkeypatch):
    closed = {"value": False}

    class DummyRedis:
        def incr(self, key):
            return 1

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: DummyRedis())

    alloc = RedisPersonIdAllocator(url="redis://localhost:6379/0")
    alloc.close()
    assert closed["value"] is True
