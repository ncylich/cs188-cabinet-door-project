import logging
import os
import sys
import argparse
from dataclasses import replace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import get_dataset_path, DiffusionPolicyDataset
from diffusion_policy.training import train
from diffusion_policy.inference import DiffusionPolicyInference
from diffusion_policy.evaluation import run_rollouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mlp")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--beta_schedule", default="cosine")
    parser.add_argument("--checkpoint_dir", default="/tmp/diffusion_policy_checkpoints")
    parser.add_argument("--eval_rollouts", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_split", default="pretrain")
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataset_path = get_dataset_path()

    config = DiffusionConfig(
        dataset_path=dataset_path,
        backbone=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_diffusion_steps=args.num_diffusion_steps,
        num_inference_steps=args.num_inference_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        beta_schedule=args.beta_schedule,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=bool(args.use_amp),
        num_workers=args.num_workers,
    )

    checkpoint_path = args.checkpoint
    if not args.skip_train:
        checkpoint_path = train(config)

    if checkpoint_path:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = DiffusionPolicyInference.from_checkpoint(checkpoint_path, device)

        results = run_rollouts(
            pipeline,
            num_rollouts=args.eval_rollouts,
            max_steps=args.eval_steps,
            split=args.eval_split,
            video_path=args.video_path,
        )

        success_rate = sum(results["successes"]) / args.eval_rollouts * 100
        logging.info("Final: %.1f%% success (%d/%d)", success_rate, sum(results["successes"]), args.eval_rollouts)


if __name__ == "__main__":
    main()
