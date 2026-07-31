"""Render the static charts used by the final training README.

The script is intentionally offline: it reads only the verified final JSON and
CSV artifacts committed under ``results/`` and writes PNGs to
``results/graphs/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRAPHS = RESULTS / "graphs"

INK = "#172033"
MUTED = "#64748B"
GRID = "#E2E8F0"
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
GOLD = "#D97706"
OLIVE = "#4D7C0F"
PANEL = "#F8FAFC"


def load_inputs() -> tuple[dict, dict, list[dict[str, str]]]:
    metrics = json.loads((RESULTS / "final_metrics.json").read_text())
    checkpoint = json.loads((RESULTS / "final_checkpoint.json").read_text())
    with (RESULTS / "training_history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    return metrics, checkpoint, history


def numeric_rows(
    history: Iterable[dict[str, str]], event: str, fields: tuple[str, ...]
) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for row in history:
        if row.get("event") != event or any(not row.get(field) for field in fields):
            continue
        rows.append(tuple(float(row[field]) for field in fields))
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    averaged = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), averaged))


def base_axes(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.suptitle(title, x=0.075, y=0.97, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.075, 0.915, subtitle, ha="left", fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.84)
    return fig, ax


def save(fig: plt.Figure, filename: str) -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_loss(history: list[dict[str, str]]) -> None:
    rows = numeric_rows(history, "train", ("train_step", "train_loss"))
    steps, losses = (np.array(column) for column in zip(*rows, strict=True))
    smooth = moving_average(losses, 20)
    fig, ax = base_axes(
        "Training loss",
        f"Logged every 10 optimizer steps · available continuation segment {int(steps[0]):,}–{int(steps[-1]):,}",
    )
    ax.plot(steps, losses, color=BLUE_LIGHT, linewidth=1.0, alpha=0.65, label="Logged loss")
    ax.plot(steps, smooth, color=BLUE, linewidth=2.2, label="20-record moving average")
    ax.set_xlabel("Optimizer step", color=MUTED)
    ax.set_ylabel("Cross-entropy loss", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "training_loss.png")


def plot_validation_loss(history: list[dict[str, str]]) -> None:
    rows = numeric_rows(history, "validation", ("train_step", "validation_loss"))
    steps, losses = (np.array(column) for column in zip(*rows, strict=True))
    fig, ax = base_axes(
        "Validation loss",
        f"20 deterministic validation batches per evaluation · {len(steps)} recorded evaluations",
    )
    ax.plot(steps, losses, color=GOLD, linewidth=2.0, marker="o", markersize=4, label="Validation loss")
    ax.scatter([steps[-1]], [losses[-1]], s=55, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=5)
    ax.annotate(
        f"Final {losses[-1]:.4f}",
        (steps[-1], losses[-1]),
        xytext=(-8, 14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("Optimizer step", color=MUTED)
    ax.set_ylabel("Cross-entropy loss", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    save(fig, "validation_loss.png")


def plot_hellaswag(history: list[dict[str, str]]) -> None:
    rows = numeric_rows(history, "hellaswag", ("train_step", "hellaswag_accuracy"))
    steps, accuracy = (np.array(column) for column in zip(*rows, strict=True))
    accuracy *= 100
    fig, ax = base_axes(
        "HellaSwag accuracy",
        "Full 10,042-example evaluation · two recorded checkpoints in the available history",
    )
    positions = np.arange(len(steps))
    bars = ax.bar(positions, accuracy, width=0.55, color=BLUE, edgecolor=INK, linewidth=0.7)
    ax.axhline(25, color=MUTED, linewidth=1.2, linestyle="--", label="Random-choice baseline (25%)")
    ax.set_xticks(positions, [f"Step {int(step):,}" for step in steps])
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    ax.set_ylim(0, max(35, float(accuracy.max()) + 5))
    for bar, value in zip(bars, accuracy, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.2f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "hellaswag_accuracy.png")


def plot_learning_rate(history: list[dict[str, str]]) -> None:
    rows = numeric_rows(history, "train", ("train_step", "learning_rate"))
    steps, rates = (np.array(column) for column in zip(*rows, strict=True))
    fig, ax = base_axes(
        "Learning rate",
        f"Cosine-decay schedule · available continuation segment {int(steps[0]):,}–{int(steps[-1]):,}",
    )
    ax.plot(steps, rates, color=OLIVE, linewidth=2.2)
    ax.set_xlabel("Optimizer step", color=MUTED)
    ax.set_ylabel("Learning rate", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
    save(fig, "learning_rate.png")


def plot_throughput(history: list[dict[str, str]]) -> None:
    rows = numeric_rows(history, "train", ("train_step", "tokens_per_second"))
    steps, throughput = (np.array(column) for column in zip(*rows, strict=True))
    smooth = moving_average(throughput, 20)
    mean = float(np.mean(throughput))
    fig, ax = base_axes(
        "Training throughput",
        "One NVIDIA H100 · logged optimizer steps · transient setup/evaluation time excluded",
    )
    ax.plot(steps, throughput / 1000, color=BLUE_LIGHT, linewidth=1.0, alpha=0.6, label="Logged throughput")
    ax.plot(steps, smooth / 1000, color=BLUE, linewidth=2.1, label="20-record moving average")
    ax.axhline(mean / 1000, color=GOLD, linewidth=1.2, linestyle="--", label=f"Logged mean {mean / 1000:.0f}k tok/s")
    ax.set_xlabel("Optimizer step", color=MUTED)
    ax.set_ylabel("Thousand tokens per second", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "tokens_per_second.png")


def plot_summary(metrics: dict, checkpoint: dict) -> None:
    fig = plt.figure(figsize=(11, 6.2), dpi=160, facecolor="white")
    fig.suptitle("GPT-2 124M pretraining summary", x=0.055, y=0.95, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.055, 0.895, "FineWeb-Edu sample-10BT · completed and independently verified", ha="left", fontsize=10.5, color=MUTED)

    cards = [
        ("FINAL STEP", f"{metrics['final_step']:,}", "optimizer steps"),
        ("TRAINING TOKENS", f"{metrics['total_tokens_processed'] / 1e9:.3f}B", "processed"),
        ("TRAIN LOSS", f"{metrics['train_loss']:.4f}", "final checkpoint"),
        ("VALIDATION", f"{metrics['validation_loss']:.4f}", f"perplexity {metrics['validation_perplexity']:.2f}"),
        ("HELLASWAG", f"{metrics['hellaswag_results']['accuracy_percent']:.2f}%", f"{metrics['hellaswag_results']['correct']:,} / {metrics['hellaswag_results']['total']:,}"),
        ("THROUGHPUT", f"{metrics['throughput_tokens_per_second'] / 1000:.0f}k", "aggregate tokens/s"),
    ]
    for index, (label, value, note) in enumerate(cards):
        row, column = divmod(index, 3)
        left = 0.055 + column * 0.31
        bottom = 0.53 - row * 0.27
        ax = fig.add_axes((left, bottom, 0.275, 0.205))
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.07, 0.76, label, transform=ax.transAxes, color=MUTED, fontsize=8.5, fontweight="bold")
        ax.text(0.07, 0.42, value, transform=ax.transAxes, color=INK, fontsize=22, fontweight="bold")
        ax.text(0.07, 0.14, note, transform=ax.transAxes, color=MUTED, fontsize=9)

    fig.text(0.055, 0.09, f"Final checkpoint · {checkpoint['size_bytes'] / (1024 ** 3):.2f} GiB · SHA-256 verified · CPU torch.load passed", color=INK, fontsize=9.5)
    fig.text(0.055, 0.05, f"Training time {metrics['training_time_hours']:.2f} h · estimated H100 compute ${metrics['estimated_cost_usd']:.2f} · W&B history step {metrics['wandb_last_history_step']:,}", color=MUTED, fontsize=9)
    save(fig, "training_summary.png")


def main() -> None:
    metrics, checkpoint, history = load_inputs()
    plot_training_loss(history)
    plot_validation_loss(history)
    plot_hellaswag(history)
    plot_learning_rate(history)
    plot_throughput(history)
    plot_summary(metrics, checkpoint)
    print(f"wrote 6 graphs to {GRAPHS}")


if __name__ == "__main__":
    main()
