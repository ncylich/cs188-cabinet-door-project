import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def dataset_path():
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    from diffusion_policy.data import get_dataset_path
    return get_dataset_path()


@pytest.fixture(scope="session")
def default_config(dataset_path):
    from diffusion_policy.config import DiffusionConfig
    return DiffusionConfig(dataset_path=dataset_path)


@pytest.fixture(scope="session")
def dataset(default_config):
    from diffusion_policy.data import DiffusionPolicyDataset
    return DiffusionPolicyDataset(default_config)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
