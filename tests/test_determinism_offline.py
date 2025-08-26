import torch

from llm_ripper.cli import set_global_seeds


def test_seed_determinism_same_seed_same_randoms():
    set_global_seeds(123)
    a = torch.rand(5)
    set_global_seeds(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)
