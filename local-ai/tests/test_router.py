import pytest

from server.errors import ModelNotInstalled, UnsupportedTask
from server.schemas.generation import GenerationRequest, GenerationSettings
from server.schemas.models import Capabilities
from server.services.inference_router import resolve_dimensions


def test_square_ratio_gives_the_base_edge():
    assert resolve_dimensions("1:1", 1024, Capabilities()) == (1024, 1024)


def test_area_stays_near_base_squared_across_ratios():
    caps = Capabilities()
    for ratio in ("1:1", "4:3", "3:4", "16:9", "9:16"):
        width, height = resolve_dimensions(ratio, 1024, caps)
        # Switching shape must not quietly change how long a render takes.
        assert 0.85 <= (width * height) / 1024**2 <= 1.15, ratio


def test_edges_snap_to_the_vae_stride():
    width, height = resolve_dimensions("16:9", 1024, Capabilities(size_multiple=16))
    assert width % 16 == 0 and height % 16 == 0


def test_edges_are_clamped_to_what_the_model_declares():
    caps = Capabilities(max_width=768, max_height=768)
    width, height = resolve_dimensions("1:1", 4096, caps)
    assert width <= 768 and height <= 768


def test_auto_and_nonsense_ratios_fall_back_to_square():
    assert resolve_dimensions("auto", 512, Capabilities()) == (512, 512)
    assert resolve_dimensions("banana", 512, Capabilities()) == (512, 512)
    assert resolve_dimensions("0:0", 512, Capabilities()) == (512, 512)


def test_route_refuses_a_model_whose_weights_are_absent(container):
    with pytest.raises(ModelNotInstalled, match="not installed"):
        container.router.route(GenerationRequest(model="flux1-schnell", prompt="a cat"))


def test_route_refuses_a_task_the_model_does_not_declare(container, install):
    install("flux1-schnell")
    with pytest.raises(UnsupportedTask, match="text-to-image"):
        container.router.route(
            GenerationRequest(model="flux1-schnell", task="image-to-video", prompt="a cat")
        )


def test_prepare_clamps_steps_and_guidance_to_the_models_own_bounds(container, install):
    install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    request = GenerationRequest(
        model="flux1-schnell",
        prompt="a cat",
        settings=GenerationSettings(steps=200, guidance=40, numImages=8),
    )
    prepared = container.router.prepare(spec, request, "job_test")
    assert prepared.steps == spec.capabilities.steps.max
    assert prepared.guidance == spec.capabilities.guidance.max
    assert prepared.num_images == spec.capabilities.max_images


def test_prepare_uses_the_models_defaults_when_the_studio_sends_nothing(container, install):
    install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    prepared = container.router.prepare(
        spec, GenerationRequest(model="flux1-schnell", prompt="a cat"), "job_test"
    )
    assert prepared.steps == spec.capabilities.steps.default
    assert prepared.guidance == spec.capabilities.guidance.default


def test_an_omitted_seed_is_recorded_not_left_none(container, install):
    install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    prepared = container.router.prepare(
        spec, GenerationRequest(model="flux1-schnell", prompt="a cat"), "job_test"
    )
    # Reproducibility: every run has a seed, even one the caller never chose.
    assert isinstance(prepared.seed, int) and prepared.seed >= 0


def test_explicit_width_and_height_win_over_the_aspect_ratio(container, install):
    install("flux1-schnell")
    spec = container.registry.get("flux1-schnell")
    request = GenerationRequest(
        model="flux1-schnell",
        prompt="a cat",
        settings=GenerationSettings(aspectRatio="16:9", width=640, height=640),
    )
    prepared = container.router.prepare(spec, request, "job_test")
    assert (prepared.width, prepared.height) == (640, 640)
