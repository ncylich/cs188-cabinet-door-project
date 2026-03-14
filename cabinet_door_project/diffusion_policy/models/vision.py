import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class SpatialResNetEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.n_keypoints = 32
        self.keypoint_conv = nn.Conv2d(512, self.n_keypoints, 1)
        self.proj = nn.Linear(self.n_keypoints * 2, feature_dim)

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.eval_resize = T.Resize(224, antialias=True)

    def spatial_softmax(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = feature_map.shape
        pos_x = torch.linspace(-1, 1, w, device=feature_map.device)
        pos_y = torch.linspace(-1, 1, h, device=feature_map.device)
        flat = feature_map.reshape(batch, channels, -1)
        weights = torch.softmax(flat, dim=-1).reshape(batch, channels, h, w)
        exp_x = (weights.sum(dim=2) * pos_x).sum(dim=-1)
        exp_y = (weights.sum(dim=3) * pos_y).sum(dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = self.normalize(images)
        images = self.eval_resize(images)
        feature_map = self.backbone(images)
        keypoint_map = self.keypoint_conv(feature_map)
        keypoints = self.spatial_softmax(keypoint_map)
        return self.proj(keypoints)


class R3MEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, model_size: str = "resnet18", freeze: bool = True):
        super().__init__()
        from r3m import load_r3m
        self.r3m = load_r3m(model_size)
        self.r3m_output_dim = 512 if model_size == "resnet18" else 2048
        self.proj = nn.Linear(self.r3m_output_dim, feature_dim)

        if freeze:
            for param in self.r3m.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float()
        elif images.max() <= 1.0:
            images = images * 255.0
        # R3M expects (B, C, H, W) float tensors in [0, 255]
        with torch.no_grad() if not any(p.requires_grad for p in self.r3m.parameters()) else torch.enable_grad():
            features = self.r3m(images)
        return self.proj(features)


class MultiCameraEncoder(nn.Module):
    def __init__(
        self,
        num_cameras: int = 3,
        feature_dim: int = 512,
        freeze: bool = True,
        encoder_type: str = "spatial_resnet",
        r3m_model_size: str = "resnet18",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "r3m":
            self.encoders = nn.ModuleList([
                R3MEncoder(feature_dim, r3m_model_size, freeze) for _ in range(num_cameras)
            ])
        else:
            self.encoders = nn.ModuleList([
                SpatialResNetEncoder(feature_dim, freeze_backbone=freeze) for _ in range(num_cameras)
            ])

    def forward(self, images: list[torch.Tensor], augment: bool = False) -> torch.Tensor:
        features = [enc(img) for enc, img in zip(self.encoders, images)]
        return torch.cat(features, dim=-1)


ResNetImageEncoder = SpatialResNetEncoder
