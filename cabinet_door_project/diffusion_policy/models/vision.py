import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class ResNetImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, freeze: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, feature_dim)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.augment = T.Compose([
            T.RandomCrop(224),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
        self.eval_resize = T.Resize(224, antialias=True)

    def forward(self, images: torch.Tensor, augment: bool = False) -> torch.Tensor:
        # images: (batch, C, H, W) uint8 or float [0,1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        images = self.normalize(images)

        if augment and self.training:
            images = self.augment(images)
        else:
            images = self.eval_resize(images)

        features = self.backbone(images).squeeze(-1).squeeze(-1)
        return self.proj(features)


class MultiCameraEncoder(nn.Module):
    def __init__(self, num_cameras: int = 3, feature_dim: int = 512, freeze: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList([
            ResNetImageEncoder(feature_dim, freeze) for _ in range(num_cameras)
        ])

    def forward(self, images: list[torch.Tensor], augment: bool = False) -> torch.Tensor:
        features = [enc(img, augment) for enc, img in zip(self.encoders, images)]
        return torch.cat(features, dim=-1)
