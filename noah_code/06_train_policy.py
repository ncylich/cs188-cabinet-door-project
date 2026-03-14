"""
Step 6: Train a Low-Dim Behavior Cloning Transformer
====================================================
Trains a sequence model on low-dimensional OpenCabinet state from the existing
LeRobot dataset. By default this uses replayed handle-relative features from
`extras/states.npz`, while keeping simpler door-relative and proprio-only paths
available as fallbacks.

Usage:
    python 06_train_policy.py
    python 06_train_policy.py --epochs 150 --seq_len 16 --batch_size 128
    python 06_train_policy.py --max_episodes 32 --epochs 20
    python 06_train_policy.py --use_diffusion_policy
"""

import argparse
import os
import random
import sys

import numpy as np
import yaml

from policy_utils import (
    EpisodeSequenceDataset,
    TemporalBCTransformer,
    compute_normalization_stats,
    get_dataset_path,
    load_lowdim_episodes,
    load_precomputed_lowdim_episodes,
    print_section,
    split_episodes,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def weighted_smooth_l1(pred_actions, target_actions, action_weights):
    import torch.nn.functional as F

    per_dim = F.smooth_l1_loss(pred_actions, target_actions, reduction="none")
    return (per_dim * action_weights.view(1, -1)).mean()


def evaluate_loss(model, dataloader, action_weights, device):
    if len(dataloader.dataset) == 0:
        return None

    total_loss = 0.0
    num_batches = 0
    model.eval()

    import torch

    with torch.no_grad():
        for states, padding_mask, actions in dataloader:
            states = states.to(device)
            padding_mask = padding_mask.to(device)
            actions = actions.to(device)
            pred_actions = model(states, padding_mask)
            loss = weighted_smooth_l1(pred_actions, actions, action_weights)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def print_diffusion_policy_instructions():
    print_section("Official Diffusion Policy Training")
    print(
        "If you want a vision policy or a stronger non-lowdim baseline, use the\n"
        "official repos instead of this lightweight sequence model.\n"
        "\n"
        "Option A: Diffusion Policy\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "Option B: pi-0 / OpenPi\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "\n"
        "Option C: GR00T N1.5\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
    )


def train_temporal_bc(config):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        sys.exit(1)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_section("Loading Dataset")
    dataset_path = get_dataset_path()
    state_mode = config.get("state_mode", "handle_relative")
    feature_cache_dir = config.get("feature_cache_dir")
    extra_dataset_path = config.get("extra_dataset_path")

    episodes, lerobot_root, state_keys, cache_dir = load_lowdim_episodes(
        dataset_path,
        max_episodes=config.get("max_episodes"),
        state_mode=state_mode,
        feature_cache_dir=feature_cache_dir,
    )
    num_base_episodes = len(episodes)

    if extra_dataset_path:
        episode_id_offset = max(ep["episode_id"] for ep in episodes) + 1 if episodes else 0
        extra_episodes = load_precomputed_lowdim_episodes(
            extra_dataset_path,
            expected_state_dim=episodes[0]["states"].shape[-1] if episodes else None,
            expected_action_dim=episodes[0]["actions"].shape[-1] if episodes else None,
            episode_id_offset=episode_id_offset,
        )
        episodes.extend(extra_episodes)
    else:
        extra_episodes = []

    train_episodes, val_episodes = split_episodes(
        episodes,
        val_fraction=config["val_fraction"],
        seed=seed,
    )

    stats = compute_normalization_stats(train_episodes)
    state_dim = train_episodes[0]["states"].shape[-1]
    action_dim = train_episodes[0]["actions"].shape[-1]

    train_dataset = EpisodeSequenceDataset(
        train_episodes,
        seq_len=config["seq_len"],
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
    )
    val_dataset = EpisodeSequenceDataset(
        val_episodes,
        seq_len=config["seq_len"],
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    print(f"Dataset root:      {lerobot_root}")
    print(f"Base episodes:     {num_base_episodes}")
    if extra_episodes:
        print(f"DAgger episodes:   {len(extra_episodes)} from {extra_dataset_path}")
    print(f"Episodes loaded:   {len(episodes)}")
    print(f"Train / val eps:   {len(train_episodes)} / {len(val_episodes)}")
    print(f"Train windows:     {len(train_dataset)}")
    print(f"Val windows:       {len(val_dataset)}")
    print(f"State dim:         {state_dim}")
    print(f"Action dim:        {action_dim}")
    print(f"State mode:        {state_mode}")
    print(f"State keys:        {state_keys}")
    if cache_dir is not None:
        print(f"Feature cache:     {cache_dir}")
    print(f"Sequence length:   {config['seq_len']}")
    print(f"Static action dims: {np.where(stats['static_action_mask'])[0].tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:            {device}")

    model_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "seq_len": config["seq_len"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
    }
    model = TemporalBCTransformer(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config["epochs"], 1)
    )

    action_weights = torch.ones(action_dim, dtype=torch.float32, device=device)
    action_weights[6] = config["gripper_loss_weight"]

    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_policy.pt")
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")

    print_section("Training")
    print(f"Epochs:            {config['epochs']}")
    print(f"Batch size:        {config['batch_size']}")
    print(f"Learning rate:     {config['learning_rate']}")
    print(f"Weight decay:      {config['weight_decay']}")
    print(f"Patience:          {config['patience']}")

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    last_train_loss = float("inf")
    last_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for states, padding_mask, actions in train_loader:
            states = states.to(device)
            padding_mask = padding_mask.to(device)
            actions = actions.to(device)

            pred_actions = model(states, padding_mask)
            loss = weighted_smooth_l1(pred_actions, actions, action_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip_norm"]
            )
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        last_train_loss = epoch_loss / max(num_batches, 1)
        val_loss = evaluate_loss(model, val_loader, action_weights, device)
        last_val_loss = last_train_loss if val_loss is None else val_loss

        print(
            f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
            f"train={last_train_loss:.5f}  val={last_val_loss:.5f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            best_train_loss = last_train_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "policy_type": "temporal_bc_transformer",
                    "model_kwargs": model_kwargs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": best_train_loss,
                    "val_loss": best_val_loss,
                    "loss": best_val_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "state_keys": state_keys,
                    "seq_len": config["seq_len"],
                    "state_mean": stats["state_mean"],
                    "state_std": stats["state_std"],
                    "action_mean": stats["action_mean"],
                    "action_std": stats["action_std"],
                    "static_action_mask": stats["static_action_mask"],
                    "static_action_values": stats["static_action_values"],
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(
                f"\nEarly stopping after {epoch + 1} epochs "
                f"(best val loss {best_val_loss:.5f})."
            )
            break

    torch.save(
        {
            "policy_type": "temporal_bc_transformer",
            "model_kwargs": model_kwargs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": last_train_loss,
            "val_loss": last_val_loss,
            "loss": last_val_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_keys": state_keys,
            "seq_len": config["seq_len"],
            "state_mean": stats["state_mean"],
            "state_std": stats["state_std"],
            "action_mean": stats["action_mean"],
            "action_std": stats["action_std"],
            "static_action_mask": stats["static_action_mask"],
            "static_action_values": stats["static_action_values"],
        },
        final_path,
    )

    print_section("Training Complete")
    print(f"Best val loss:     {best_val_loss:.5f}")
    print(f"Best train loss:   {best_train_loss:.5f}")
    print(f"Best checkpoint:   {best_path}")
    print(f"Final checkpoint:  {final_path}")
    print(
        "\nEvaluate with:\n"
        f"  python 07_evaluate_policy.py --checkpoint {best_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument(
        "--seq_len", type=int, default=16, help="Number of past states per sample"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Transformer hidden size"
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Transformer attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Transformer encoder layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of episodes reserved for validation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience measured in epochs",
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--gripper_loss_weight",
        type=float,
        default=2.0,
        help="Extra weight on the gripper action dimension",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--state_mode",
        type=str,
        default="handle_relative",
        choices=["handle_relative", "door_relative", "proprio"],
        help="Low-dimensional state representation to train on",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on the number of parquet episodes to load",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for replayed low-dimensional features",
    )
    parser.add_argument(
        "--extra_dataset_path",
        type=str,
        default=None,
        help="Optional directory of precomputed DAgger parquet episodes to merge into training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI hyperparameters)",
    )
    parser.add_argument(
        "--use_diffusion_policy",
        action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "val_fraction": args.val_fraction,
            "patience": args.patience,
            "grad_clip_norm": args.grad_clip_norm,
            "gripper_loss_weight": args.gripper_loss_weight,
            "seed": args.seed,
            "state_mode": args.state_mode,
            "max_episodes": args.max_episodes,
            "feature_cache_dir": args.feature_cache_dir,
            "extra_dataset_path": args.extra_dataset_path,
            "checkpoint_dir": args.checkpoint_dir,
        }

    train_temporal_bc(config)


if __name__ == "__main__":
    main()
