from server.services.inference_router import EnginePool


def test_a_loaded_model_is_reused_rather_than_reloaded(container, install, fake_engine):
    path = install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    first = container.router.pool.acquire(spec, path)
    second = container.router.pool.acquire(spec, path)
    assert first is second
    assert fake_engine.loads == ["flux1-schnell"]


def test_load_state_is_reported_for_the_model_library(container, install, fake_engine):
    path = install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    assert container.router.pool.load_state(spec.id) == "not_loaded"
    container.router.pool.acquire(spec, path)
    assert container.router.pool.load_state(spec.id) == "ready"


def test_the_lru_evicts_before_admitting_the_next_model(container, install, fake_engine):
    pool = EnginePool(container.settings, container.gpu, max_resident=1)
    first = container.registry.get("flux1-schnell")
    second = container.registry.get("flux1-dev")
    pool.acquire(first, install("flux1-schnell"))
    pool.acquire(second, install("flux1-dev"))
    assert fake_engine.unloads == ["flux1-schnell"]
    assert pool.load_state("flux1-schnell") == "not_loaded"
    assert pool.load_state("flux1-dev") == "ready"


def test_two_can_stay_resident_when_the_budget_allows(container, install, fake_engine):
    pool = EnginePool(container.settings, container.gpu, max_resident=2)
    pool.acquire(container.registry.get("flux1-schnell"), install("flux1-schnell"))
    pool.acquire(container.registry.get("flux1-dev"), install("flux1-dev"))
    assert fake_engine.unloads == []
    assert pool.states() == {"flux1-schnell": "ready", "flux1-dev": "ready"}


def test_unload_all_releases_everything(container, install, fake_engine):
    pool = EnginePool(container.settings, container.gpu, max_resident=2)
    pool.acquire(container.registry.get("flux1-schnell"), install("flux1-schnell"))
    pool.acquire(container.registry.get("flux1-dev"), install("flux1-dev"))
    pool.unload_all()
    assert pool.states() == {}
    assert sorted(fake_engine.unloads) == ["flux1-dev", "flux1-schnell"]


def test_a_failed_load_does_not_leave_the_model_stuck_in_loading(container, install):
    from engines.loader import register_engine, unregister_engine

    def explode(device, dtype):
        raise RuntimeError("weights are corrupt")

    register_engine("diffusers-t2i", explode)
    try:
        spec = container.registry.get("flux1-schnell")
        try:
            container.router.pool.acquire(spec, install("flux1-schnell"))
        except RuntimeError:
            pass
        assert container.router.pool.load_state(spec.id) == "not_loaded"
    finally:
        unregister_engine("diffusers-t2i")
