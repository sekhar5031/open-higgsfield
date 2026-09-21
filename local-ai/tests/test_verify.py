"""Verification of registry rows against the live repository.

The registry is hand-written from model cards, so these tests pin the checks
that catch a row having drifted — the API itself is stubbed, because the point
is the comparison logic, not huggingface_hub.
"""

from dataclasses import dataclass

import pytest

from server.services.model_manager import wanted_files


@dataclass
class Sibling:
    rfilename: str
    size: int = 0


def info_for(files, gated=False, license_id="creativeml-openrail-m"):
    @dataclass
    class Info:
        siblings: list
        gated: object
        card_data: dict

    return lambda repo, **kwargs: Info(
        siblings=files, gated=gated, card_data={"license": license_id}
    )


GB = 1024**3


def test_wanted_files_drops_the_formats_the_engine_never_reads():
    names = ["unet/model.safetensors", "unet/model.bin", "v1-5-pruned.ckpt", "model_index.json"]
    kept = wanted_files(names, None, ("*.bin", "*.ckpt"))
    assert kept == ["unet/model.safetensors", "model_index.json"]


def test_wanted_files_honours_an_allow_list():
    names = ["a.safetensors", "docs/readme.md"]
    assert wanted_files(names, ("*.safetensors",), ()) == ["a.safetensors"]


def test_a_healthy_row_verifies_clean(container):
    spec = container.registry.get("sd15")
    files = [Sibling("model_index.json", 500), Sibling("unet/diffusion_pytorch_model.safetensors", int(4.2 * GB))]
    result = container.models.verify(spec, model_info=info_for(files))
    assert result.ok, result.problems
    assert result.exists
    assert result.file_count == 2
    assert 4.0 < result.download_bytes / GB < 4.4


def test_ckpt_duplicates_are_excluded_from_the_reported_size(container):
    spec = container.registry.get("sd15")
    files = [
        Sibling("unet/diffusion_pytorch_model.safetensors", int(4.2 * GB)),
        Sibling("v1-5-pruned.ckpt", int(7.0 * GB)),
    ]
    result = container.models.verify(spec, model_info=info_for(files))
    # The .ckpt would otherwise more than double the download for no benefit.
    assert result.download_bytes / GB == pytest.approx(4.2, abs=0.1)


def test_a_repository_that_became_gated_is_reported(container):
    spec = container.registry.get("sd15")  # registry says gated: false
    files = [Sibling("unet/model.safetensors", int(4.2 * GB))]
    result = container.models.verify(spec, model_info=info_for(files, gated=True))
    assert not result.ok
    assert any("gated" in problem for problem in result.problems)


def test_a_missing_repository_fails_rather_than_raising(container):
    spec = container.registry.get("sd15")

    def explode(repo, **kwargs):
        raise OSError("404 Client Error: Repository Not Found")

    result = container.models.verify(spec, model_info=explode)
    assert result.exists is False
    assert not result.ok
    assert "could not read the repository" in result.problems[0]


def test_a_download_set_with_no_weights_is_a_problem(container):
    spec = container.registry.get("sd15")
    result = container.models.verify(spec, model_info=info_for([Sibling("README.md", 100)]))
    assert not result.ok
    assert any("safetensors" in problem for problem in result.problems)


def test_a_licence_change_is_a_note_not_a_failure(container):
    spec = container.registry.get("sd15")
    files = [Sibling("unet/model.safetensors", int(4.2 * GB))]
    result = container.models.verify(spec, model_info=info_for(files, license_id="apache-2.0"))
    # Worth telling the user; not something to block an install over.
    assert result.ok
    assert any("licence" in note for note in result.notes)


def test_a_size_that_has_drifted_is_noted(container):
    spec = container.registry.get("sd15")  # declares 4.3 GB
    files = [Sibling("unet/model.safetensors", int(12.0 * GB))]
    result = container.models.verify(spec, model_info=info_for(files))
    assert any("12.0 GB" in note for note in result.notes)
