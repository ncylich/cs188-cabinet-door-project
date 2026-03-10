from dataclasses import dataclass, field


@dataclass
class DiffusionConfig:
    state_dim: int = 16
    action_dim: int = 12
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    num_diffusion_steps: int = 100
    num_inference_steps: int = 16
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    backbone: str = "mlp"
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256

    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    num_epochs: int = 3000
    warmup_steps: int = 500
    ema_decay: float = 0.9999

    dataset_path: str = ""
    checkpoint_dir: str = "/tmp/diffusion_policy_checkpoints"

    use_amp: bool = True
    compile_model: bool = False
    num_workers: int = 4

    # Visuomotor (V4)
    use_images: bool = False
    image_keys: list = field(default_factory=lambda: [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ])
    image_encoder_feature_dim: int = 512
    freeze_image_encoder: bool = True
