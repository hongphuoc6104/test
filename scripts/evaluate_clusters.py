import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict

import pandas as pd


def _load_predictions(characters_json: str) -> Dict[str, str]:
    """Load frame -> character_id predictions from ``characters.json``.

    The JSON file can be in one of two formats:

    1. ``{"frame.jpg": "0", ...}``
    2. ``{"0": {"frames": ["frame.jpg", ...]}, ...}``
    """
    with open(characters_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: frame -> character_id mapping
    if all(isinstance(v, (int, str)) for v in data.values()):
        return {str(k): str(v) for k, v in data.items()}

    # Case 2: character_id -> info with frames list
    preds: Dict[str, str] = {}
    for cid, info in data.items():
        frames = info.get("frames") if isinstance(info, dict) else None
        if not frames:
            continue
        for frame in frames:
            preds[str(frame)] = str(cid)
    return preds


def evaluate(characters_json: str, labels_csv: str, out_path: str) -> str:
    """Evaluate clustering results against human labels."""
    preds = _load_predictions(characters_json)
    df = pd.read_csv(labels_csv)

    metrics: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for _, row in df.iterrows():
        frame = str(row["frame"])
        true_id = str(row["character_id"])
        pred_id = preds.get(frame)
        if pred_id == true_id:
            metrics[true_id]["tp"] += 1
        else:
            metrics[true_id]["fn"] += 1
            if pred_id is not None:
                metrics[pred_id]["fp"] += 1

    rows = []
    total_tp = total_fp = total_fn = 0
    for cid, m in sorted(metrics.items()):
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "character_id": cid,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall
        else 0.0
    )
    rows.append({
        "character_id": "OVERALL",
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate character clustering")
    parser.add_argument("characters", help="Path to characters.json")
    parser.add_argument(
        "labels", help="CSV with human labels (columns: frame, character_id)"
    )
    parser.add_argument(
        "--out", default="reports/eval_metrics.csv", help="Where to write metrics CSV"
    )
    args = parser.parse_args()
    evaluate(args.characters, args.labels, args.out)


if __name__ == "__main__":
    main()