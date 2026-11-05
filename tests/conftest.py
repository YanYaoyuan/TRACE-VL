import pytest
import torch

from trace_vl.data.synthetic import make_synthetic


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(1)


@pytest.fixture
def fixture_manifest(tmp_path):
    return make_synthetic(tmp_path / "clips", train=4, val=2, test=2, video=True)
