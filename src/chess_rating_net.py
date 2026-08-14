"""
Parametrized RatingNet baseline with attention and anomaly extensions.

This file extends the original ``chess_rating_net.py`` from
``AstroBoy1/RatingNet`` (MIT) with:

* argparse + YAML configuration for all major training/inference knobs
* periodic checkpointing (every-epoch + latest/resume helper)
* optional Bahdanau/self-attention module wired into the model
* optional anomaly-detection branch

Training remains off by default so the frozen ``model_55.pth`` checkpoint can
be loaded for inference without GPU time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow running both as `python src/chess_rating_net.py` and via `from src.chess_rating_net import ...`.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from attention import BahdanauAttention, SelfAttention
from anomaly import AnomalyDetector
from format_data import board_to_array, categorize_time_control, time_to_seconds

logger = logging.getLogger(__name__)


class ChessGamesDataset(Dataset):
    """Dataset loader for preprocessed pickle games."""

    def __init__(
        self,
        filenames: list[str],
        max_moves: int = 100,
        ratings_mean: float = 1514,
        ratings_std: float = 366,
        clocks_mean: float = 273,
        clocks_std: float = 380,
    ):
        self.filenames = filenames
        self.max_moves = max_moves
        self.ratings_mean = ratings_mean
        self.ratings_std = ratings_std
        self.clocks_mean = clocks_mean
        self.clocks_std = clocks_std

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with open(self.filenames[idx], "rb") as f:
            game_info = pickle.load(f)
        clocks = [time_to_seconds(c) for c in game_info.get("Clocks", [])]
        clocks = [(c - self.clocks_mean) / self.clocks_std for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[: self.max_moves]
        white = game_info.get("white", False)
        last_rating = game_info.get("rating_after_last_game")
        if last_rating is not None:
            last_rating = (last_rating - self.ratings_mean) / self.ratings_std
            last_rating = torch.tensor(last_rating, dtype=torch.float)
        positions = torch.stack(game_info["Positions"])[: self.max_moves]
        white_elo, black_elo = float(game_info["WhiteElo"]), float(game_info["BlackElo"])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float)
        targets = (targets - self.ratings_mean) / self.ratings_std

        length = len(positions)
        initial_time, increment = map(int, game_info["Time"].split("+"))
        estimated_duration = initial_time + 40 * increment
        time_control = categorize_time_control(estimated_duration)
        result = game_info.get("Result")

        return {
            "positions": positions,
            "clocks": clocks,
            "targets": targets,
            "length": length,
            "time_control": time_control,
            "white": white,
            "last_rating": last_rating,
            "result": result,
            "game_id": Path(self.filenames[idx]).stem,
            "white_elo": white_elo,
            "black_elo": black_elo,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sequences for batching."""
    positions = pad_sequence([item["positions"] for item in batch], batch_first=True)
    clocks = pad_sequence([item["clocks"] for item in batch], batch_first=True)
    targets = torch.stack([item["targets"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.int)
    time_controls = [item["time_control"] for item in batch]
    white = torch.tensor([item["white"] for item in batch])
    game_ids = [item["game_id"] for item in batch]
    white_elos = torch.tensor([item["white_elo"] for item in batch], dtype=torch.float)
    black_elos = torch.tensor([item["black_elo"] for item in batch], dtype=torch.float)
    last_rating = None
    if batch[0]["last_rating"] is not None:
        last_rating = torch.stack([item["last_rating"] for item in batch])
    if batch[0]["result"] is not None:
        results = [item["result"] for item in batch]
        return {
            "positions": positions,
            "clocks": clocks,
            "targets": targets,
            "lengths": lengths,
            "time_controls": time_controls,
            "white": white,
            "last_rating": last_rating,
            "results": results,
            "game_ids": game_ids,
            "white_elos": white_elos,
            "black_elos": black_elos,
        }
    return {
        "positions": positions,
        "clocks": clocks,
        "targets": targets,
        "lengths": lengths,
        "time_controls": time_controls,
        "white": white,
        "last_rating": last_rating,
        "game_ids": game_ids,
        "white_elos": white_elos,
        "black_elos": black_elos,
    }


class ChessEloPredictor(nn.Module):
    """CNN-BiLSTM rating estimator with optional attention and anomaly branch.

    The base architecture matches the released RatingNet checkpoint. When
    ``use_attention`` is False the forward pass is identical to the baseline,
    so the frozen ``model_55.pth`` weights load cleanly. When True, an
    attention module is attached; its weights are *not* present in the frozen
    checkpoint and must be trained on HPC (outside this prototype scope).

    Args:
        conv_filters: Number of filters in the first conv layer.
        lstm_layers: Number of LSTM layers.
        dropout_rate: Dropout probability.
        lstm_h: LSTM hidden size.
        fc1_h: First fully-connected layer size.
        bidirectional: Whether the LSTM is bidirectional.
        use_attention: If True, attach an attention module.
        attention_type: ``bahdanau`` or ``self``.
        attention_dim: Projection size for Bahdanau attention.
        use_anomaly: If True, expose an anomaly detector branch.
    """

    def __init__(
        self,
        conv_filters: int = 32,
        lstm_layers: int = 3,
        dropout_rate: float = 0.5,
        lstm_h: int = 64,
        fc1_h: int = 32,
        bidirectional: bool = True,
        use_attention: bool = False,
        attention_type: str = "bahdanau",
        attention_dim: int = 64,
        use_anomaly: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_anomaly = use_anomaly
        self.bidirectional = bidirectional
        self.lstm_h = lstm_h

        # CNN trunk (identical to baseline)
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)
        self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters * 8)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # BiLSTM (identical to baseline)
        lstm_input_size = conv_filters * 8 + 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_h,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_h * 2 if bidirectional else lstm_h

        # Optional attention module (does not change baseline head dims)
        self.attention: nn.Module | None = None
        if use_attention:
            if attention_type == "bahdanau":
                self.attention = BahdanauAttention(lstm_output_dim, attention_dim=attention_dim)
            elif attention_type == "self":
                self.attention = SelfAttention(lstm_output_dim)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")

        # Rating head (identical to baseline)
        self.fc1 = nn.Linear(lstm_output_dim, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 2)

        # Optional anomaly branch
        self.anomaly_detector: nn.Module | None = None
        if use_anomaly:
            self.anomaly_detector = AnomalyDetector()

    def forward(
        self,
        positions: torch.Tensor,
        clocks: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
        baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        Returns:
            By default (return_attention=False): ``(per_move_preds, last_step_preds)``
                identical to the baseline.
            If return_attention=True: a dictionary with predictions, attention
                weights, anomaly scores, and intermediate values.
        """
        batch_size = positions.size(0)
        sequence_length = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)

        x = F.leaky_relu(self.bn1(self.conv1(positions)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        x = x.view(batch_size, sequence_length, -1)

        clocks = clocks.unsqueeze(2)
        lstm_input = torch.cat((x, clocks), dim=2)
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        y = F.leaky_relu(self.fc1(lstm_output))
        y = self.dropout1(y)
        per_move_preds = self.fc2(y)

        idx = torch.arange(batch_size, device=positions.device)
        last_time_step_output = per_move_preds[idx, lengths - 1, :]

        if not return_attention:
            return per_move_preds, last_time_step_output

        # Build mask based on actual lengths
        mask = torch.arange(sequence_length, device=positions.device).unsqueeze(0) < lengths.unsqueeze(1)

        attention_weights = None
        if self.attention is not None:
            _, attention_weights = self.attention(lstm_output, mask=mask)

        anomaly_out = None
        if self.anomaly_detector is not None and baseline is not None:
            anomaly_out = self.anomaly_detector(per_move_preds, baseline, attention_weights)

        return {
            "per_move_preds": per_move_preds,
            "last_step_preds": last_time_step_output,
            "attention_weights": attention_weights,
            "anomaly": anomaly_out,
        }

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
        """Load a baseline checkpoint, ignoring keys that belong to attention/anomaly.

        Warns when the checkpoint lacks attention/anomaly parameters that this
        model actually uses, since those modules would stay randomly
        initialized and silently corrupt inference.
        """
        result = self.load_state_dict(state_dict, strict=strict)
        if not strict and result.missing_keys:
            served_missing = [
                k
                for k in result.missing_keys
                if (self.use_attention and k.startswith("attention."))
                or (self.use_anomaly and k.startswith("anomaly_detector."))
            ]
            if served_missing:
                logger.warning(
                    "Checkpoint is missing %d attention/anomaly parameter(s) the served model "
                    "uses; they remain randomly initialized: %s",
                    len(served_missing),
                    served_missing if len(served_missing) <= 8 else served_missing[:8] + ["..."],
                )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
) -> float:
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        positions = batch["positions"].to(device)
        clocks = batch["clocks"].to(device)
        targets = batch["targets"].to(device)
        lengths = batch["lengths"]
        optimizer.zero_grad()
        _, outputs = model(positions, clocks, lengths)
        loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
) -> float:
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch["positions"].to(device)
            clocks = batch["clocks"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"]
            _, outputs = model(positions, clocks, lengths)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def mae_per_item(outputs: torch.Tensor, targets: torch.Tensor, ratings_mean: float, ratings_std: float) -> torch.Tensor:
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    return torch.abs(outputs_rescaled - targets_rescaled).mean(dim=1)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
    per_game_csv: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate on the test set.

    When ``per_game_csv`` is given, also dumps one row per game
    (``game_id, white_err, black_err, time_control, white_elo, black_elo``)
    with errors in Elo points, enabling paired-bootstrap, rating-bracket,
    calibration, and move-index analyses downstream.
    """
    model.eval()
    total_test_loss = 0.0
    loss_by_time_control = {tc: 0.0 for tc in ["ultrabullet", "bullet", "blitz", "rapid", "classical"]}
    count_by_time_control = {tc: 0 for tc in ["ultrabullet", "bullet", "blitz", "rapid", "classical"]}

    csv_file = None
    csv_writer = None
    if per_game_csv is not None:
        csv_file = open(per_game_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["game_id", "white_err", "black_err", "time_control", "white_elo", "black_elo"])

    try:
        with torch.no_grad():
            for batch in test_loader:
                positions = batch["positions"].to(device)
                clocks = batch["clocks"].to(device)
                targets = batch["targets"].to(device)
                lengths = batch["lengths"]
                time_controls = batch["time_controls"]

                _, outputs = model(positions, clocks, lengths)
                loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
                total_test_loss += loss.item()

                if isinstance(criterion, nn.L1Loss):
                    mae = mae_per_item(outputs, targets, ratings_mean, ratings_std)
                    for idx, time_control in enumerate(time_controls):
                        loss_by_time_control[time_control] += mae[idx].item()
                        count_by_time_control[time_control] += 1

                if csv_writer is not None:
                    abs_errors = torch.abs(
                        outputs * ratings_std + ratings_mean - targets * ratings_std - ratings_mean
                    )  # (batch, 2) Elo points
                    for i, time_control in enumerate(time_controls):
                        csv_writer.writerow(
                            [
                                batch["game_ids"][i],
                                round(abs_errors[i, 0].item(), 4),
                                round(abs_errors[i, 1].item(), 4),
                                time_control,
                                batch["white_elos"][i].item(),
                                batch["black_elos"][i].item(),
                            ]
                        )
    finally:
        if csv_file is not None:
            csv_file.close()

    for key in loss_by_time_control:
        if count_by_time_control[key] > 0:
            loss_by_time_control[key] /= count_by_time_control[key]

    return total_test_loss / len(test_loader), loss_by_time_control


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    params: dict[str, Any],
    best_val_loss: float | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    best_epoch: int | None = None,
) -> None:
    """Save optimizer/scheduler state, epoch counters, and hyperparameters alongside weights."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "params": params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    scheduler: ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    """Resume from a checkpoint written by ``save_checkpoint``."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def load_or_create_split(
    data_dir: str,
    all_files: list[str],
    split_seed: int,
) -> tuple[list[str], list[str], list[str], str]:
    """Deterministic, auditable train/val/test split persisted to ``split_manifest.json``.

    On first run, splits the sorted basenames with ``split_seed`` and writes the
    manifest (three file lists + SHA-256 over the sorted basenames). On every
    subsequent run the manifest is loaded and verified instead of recomputed.

    Returns ``(train, val, test)`` basename lists and the manifest hash.
    """
    manifest_path = Path(data_dir) / "split_manifest.json"
    digest = hashlib.sha256("\n".join(all_files).encode("utf-8")).hexdigest()

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("sha256") != digest:
            raise ValueError(
                f"split_manifest.json hash mismatch (manifest={manifest.get('sha256')}, "
                f"current={digest}): the data directory changed since the split was frozen. "
                f"Delete {manifest_path} to recompute the split."
            )
        train_names = manifest["train_files"]
        val_names = manifest["val_files"]
        test_names = manifest["test_files"]
        if sorted(train_names + val_names + test_names) != sorted(all_files):
            raise ValueError(
                f"split_manifest.json file lists do not match the contents of {data_dir}. "
                f"Delete {manifest_path} to recompute the split."
            )
        print(f"Loaded split manifest {manifest_path} (verified)")
    else:
        train_val_names, test_names = train_test_split(all_files, test_size=0.1, random_state=split_seed)
        train_names, val_names = train_test_split(train_val_names, test_size=0.2, random_state=split_seed)
        train_names, val_names, test_names = sorted(train_names), sorted(val_names), sorted(test_names)
        manifest = {
            "sha256": digest,
            "split_seed": split_seed,
            "train_files": train_names,
            "val_files": val_names,
            "test_files": test_names,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote split manifest {manifest_path}")

    return train_names, val_names, test_names, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RatingNet prototype trainer / inference runner")
    parser.add_argument("--data_dir", default="data/processed_games", help="Directory containing .pkl preprocessed games")
    parser.add_argument("--experiment", default="cnn_bilstm_clocks_all", help="Experiment name (used for model/log dirs)")
    parser.add_argument("--train", action="store_true", default=False, help="Run training (default: inference-only)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--model_dir", default="models", help="Root directory for checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate in inference mode (default: discover model_55.pth)")
    parser.add_argument("--config", default=None, help="Optional YAML config file to override defaults")
    parser.add_argument("--use_attention", action="store_true", help="Attach attention module to the model")
    parser.add_argument("--use_anomaly", action="store_true", help="Attach anomaly-detection branch")
    parser.add_argument("--val_batch_size", type=int, default=512, help="Validation/test batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for torch/numpy/random and training shuffle (vary across seed-ablation runs)")
    parser.add_argument("--split_seed", type=int, default=42, help="Train/val/test split seed; NEVER vary (preserves comparability with the paper's 182 MAE)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=5, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--conv_filters", type=int, default=32, help="CNN base filter count")
    parser.add_argument("--lstm_layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--lstm_h", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--fc1_h", type=int, default=32, help="FC hidden size")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True, help="Use bidirectional LSTM")
    parser.add_argument("--attention_type", default="bahdanau", choices=["bahdanau", "self"], help="Attention variant")
    parser.add_argument("--attention_dim", type=int, default=64, help="Bahdanau attention projection size")
    return parser


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge optional YAML config into argparse namespace."""
    if args.config is None:
        return args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> int:
    parser = build_parser()
    args = load_config(parser.parse_args())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Hyperparameters: CLI > YAML > checkpoint > defaults.
    params: dict[str, Any] = {
        "train_batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "patience": args.patience,
        "lr_factor": args.lr_factor,
        "conv_filters": args.conv_filters,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
        "dropout_rate": args.dropout_rate,
        "lstm_h": args.lstm_h,
        "fc1_h": args.fc1_h,
        "use_attention": args.use_attention,
        "attention_type": args.attention_type,
        "attention_dim": args.attention_dim,
        "use_anomaly": args.use_anomaly,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }

    data_dir = args.data_dir
    experiment_name = args.experiment
    model_dir = Path(args.model_dir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    if not all_files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    train_names, val_names, test_names, manifest_hash = load_or_create_split(data_dir, all_files, args.split_seed)
    train_files = [os.path.join(data_dir, f) for f in train_names]
    val_files = [os.path.join(data_dir, f) for f in val_names]
    test_files = [os.path.join(data_dir, f) for f in test_names]

    print(f"Environment: data_dir={data_dir} files={len(all_files)} seed={args.seed} split_seed={args.split_seed}")
    print(f"Split: train={len(train_files)} val={len(val_files)} test={len(test_files)} manifest_sha256={manifest_hash}")
    print(f"Resolved hyperparameters: {params}")

    train_dataset = ChessGamesDataset(train_files)
    val_dataset = ChessGamesDataset(val_files)
    test_dataset = ChessGamesDataset(test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["val_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["val_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ChessEloPredictor(
        conv_filters=params["conv_filters"],
        lstm_layers=params["lstm_layers"],
        dropout_rate=params["dropout_rate"],
        lstm_h=params["lstm_h"],
        fc1_h=params["fc1_h"],
        bidirectional=params["bidirectional"],
        use_attention=params["use_attention"],
        attention_type=params["attention_type"],
        attention_dim=params["attention_dim"],
        use_anomaly=params["use_anomaly"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=params["patience"], factor=params["lr_factor"])
    criterion = nn.L1Loss()

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    latest_path = model_dir / "latest.pth"

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, device, scheduler=scheduler)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)

    if not args.train:
        # Inference-only path: load a checkpoint for evaluation.
        # Prefer an explicit --checkpoint; otherwise discover the frozen baseline
        # (experiment-specific path, then the root models/ directory, e.g. models/model_55.pth).
        if args.checkpoint:
            best_path = Path(args.checkpoint)
            if not best_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        else:
            candidate_paths = [model_dir / "model_55.pth", Path(args.model_dir) / "model_55.pth"]
            best_path = next((p for p in candidate_paths if p.exists()), None)
            if best_path is None:
                raise FileNotFoundError(
                    f"Frozen checkpoint not found. Place model_55.pth at {model_dir / 'model_55.pth'} "
                    f"or {Path(args.model_dir) / 'model_55.pth'}, or pass --checkpoint, then retry."
                )
        print(f"Loading checkpoint from {best_path}")
        saved_model = torch.load(best_path, map_location=device)
        model.load_base_state_dict(saved_model["model_state_dict"], strict=False)
        per_game_csv = model_dir / "per_game_errors.csv"
        test_loss, loss_by_tc = test(model, test_loader, device, criterion, per_game_csv=str(per_game_csv))
        print("Test Loss:", test_loss)
        print("Loss by time control:", loss_by_tc)
        print(f"Wrote per-game errors to {per_game_csv}")
        return 0

    writer = SummaryWriter(log_dir=str(log_dir))
    print("Training model")
    start = time.time()

    for epoch in range(start_epoch, params["epochs"]):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        val_loss = validate(model, val_loader, device, criterion)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        epoch_duration = (time.time() - epoch_start) / 60
        writer.add_scalar("Timing/Epoch Duration", epoch_duration, epoch)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt = model_dir / f"best_model.pth"
            save_checkpoint(
                best_ckpt, model, optimizer, epoch + 1, params,
                best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
            )
            print("Saved new best model")

        # Periodic every-epoch checkpoint. Written after the best-loss update so
        # it carries the post-update best (resume never sees a stale value).
        epoch_ckpt = model_dir / f"model_{epoch + 1}.pth"
        save_checkpoint(
            epoch_ckpt, model, optimizer, epoch + 1, params,
            best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
        )
        save_checkpoint(
            latest_path, model, optimizer, epoch + 1, params,
            best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
        )
        print(f"Saved epoch checkpoint {epoch_ckpt}")

    end = time.time()
    print("Training duration (min):", (end - start) / 60)
    print("best val loss:", best_val_loss)
    print("best val epoch:", best_epoch)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
