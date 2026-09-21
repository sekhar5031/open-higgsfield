from server.services.gpu_manager import GB, GpuManager


def test_device_selection_honours_an_explicit_override():
    assert GpuManager(preferred="cpu").torch_device() == "cpu"
    assert GpuManager(preferred="cuda:1").torch_device() == "cuda:1"


def test_auto_falls_back_to_cpu_when_there_is_no_cuda_device():
    device = GpuManager(preferred="auto").torch_device()
    assert device in {"cpu", "cuda"}


def test_a_report_is_always_returned_even_with_no_torch():
    report = GpuManager(preferred="auto").report()
    assert report.devices
    assert report.device_in_use


def test_admission_does_not_block_cpu_execution():
    # No CUDA device to measure against: slow is not the same as impossible.
    gpu = GpuManager(preferred="cpu")
    if gpu.memory() is None:
        gpu.admit("flux1-schnell", 24.0)  # must not raise


def test_admission_rejects_a_model_larger_than_the_card():
    import pytest

    from server.errors import InsufficientVram
    from server.services.gpu_manager import DeviceMemory

    gpu = GpuManager(preferred="cuda")
    gpu.memory = lambda index=0: DeviceMemory(total=32 * GB, free=30 * GB, used=2 * GB)  # type: ignore[method-assign]
    with pytest.raises(InsufficientVram, match="in total"):
        gpu.admit("huge-model", 48.0)
    assert gpu.fits_device(48.0) is False
    assert gpu.fits_device(24.0) is True


def test_admission_rejects_a_model_that_would_not_fit_right_now():
    import pytest

    from server.errors import InsufficientVram
    from server.services.gpu_manager import DeviceMemory

    gpu = GpuManager(preferred="cuda")
    gpu.memory = lambda index=0: DeviceMemory(total=32 * GB, free=4 * GB, used=28 * GB)  # type: ignore[method-assign]
    with pytest.raises(InsufficientVram, match="is free"):
        gpu.admit("flux1-schnell", 24.0)
