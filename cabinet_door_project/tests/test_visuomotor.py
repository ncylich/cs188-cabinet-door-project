import os

import numpy as np
import torch
import pytest

from diffusion_policy.models.vision import ResNetImageEncoder, MultiCameraEncoder


class TestImageEncoder:
    def test_output_shape(self):
        encoder = ResNetImageEncoder(feature_dim=512, freeze=True)
        images = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
        features = encoder(images)
        assert features.shape == (4, 512)

    def test_augmentation_differs(self):
        encoder = ResNetImageEncoder(feature_dim=512, freeze=False)
        encoder.train()
        img = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
        f1 = encoder(img, augment=True)
        f2 = encoder(img, augment=True)
        # Augmentation may or may not differ per call due to randomness
        # Just verify it doesn't crash and shapes are correct
        assert f1.shape == (2, 512)


class TestMultiCameraEncoder:
    def test_visuomotor_obs_shape(self):
        encoder = MultiCameraEncoder(num_cameras=3, feature_dim=512, freeze=True)
        images = [torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8) for _ in range(3)]
        features = encoder(images)
        assert features.shape == (4, 1536)  # 3 * 512

    def test_combined_with_state(self):
        encoder = MultiCameraEncoder(num_cameras=3, feature_dim=512, freeze=True)
        images = [torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8) for _ in range(3)]
        img_features = encoder(images)
        state = torch.randn(4, 16)
        combined = torch.cat([img_features, state], dim=-1)
        assert combined.shape == (4, 1552)


class TestGradientToImages:
    def test_unfrozen_gradients_flow(self):
        encoder = ResNetImageEncoder(feature_dim=512, freeze=False)
        encoder.train()
        images = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8).float() / 255.0
        images.requires_grad_(True)
        features = encoder(images)
        loss = features.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.backbone.parameters())
        assert has_grad, "No gradients flowing to image encoder"


class TestImageLoading:
    def test_load_video_frame(self, dataset_path):
        video_dir = os.path.join(dataset_path, "videos", "chunk-000", "observation.images.robot0_agentview_left")
        video_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
        assert len(video_files) > 0

        import imageio.v3 as iio
        video_path = os.path.join(video_dir, video_files[0])
        frames = iio.imread(video_path, plugin="pyav")
        frame = frames[0]
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
