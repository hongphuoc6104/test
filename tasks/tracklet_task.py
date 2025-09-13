from __future__ import annotations

import os
import numpy as np
import pandas as pd
from prefect import task


def _frame_to_int(frame_name: str) -> int:
    """Convert frame filename to an integer index."""
    base = os.path.splitext(str(frame_name))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return 0


def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute Intersection over Union for two bboxes [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
    return float(np.dot(v1, v2) / denom)


def link_tracklets(
    df: pd.DataFrame, iou_threshold: float = 0.3, cos_threshold: float = 0.6
) -> pd.DataFrame:
    """Link consecutive detections into tracklets using IoU and cosine similarity."""
    if df.empty:
        return df

    df = df.copy()
    df["_frame_idx"] = df["frame"].apply(_frame_to_int)
    df.sort_values("_frame_idx", inplace=True)
    df.reset_index(drop=True, inplace=True)

    active_tracks: list[dict] = []
    next_id = 0
    assigned_ids: list[int] = []

    for _, row in df.iterrows():
        frame_idx = row["_frame_idx"]
        bbox = np.array(row["bbox"])
        emb = np.array(row["emb"])

        # prune tracks that are older than the previous frame to keep lookups fast
        active_tracks = [
            t for t in active_tracks if t["last_frame"] >= frame_idx - 1
        ]

        matched_track = None
        for track in active_tracks:
            # only consider tracks from previous frame
            if track["last_frame"] != frame_idx - 1:
                continue
            if (
                _iou(bbox, track["bbox"]) >= iou_threshold
                and _cosine_sim(emb, track["emb"]) >= cos_threshold
            ):
                matched_track = track
                break

        if matched_track is None:
            next_id += 1
            matched_track = {
                "track_id": next_id,
                "bbox": bbox,
                "emb": emb,
                "last_frame": frame_idx,
            }
            active_tracks.append(matched_track)
        else:
            matched_track["bbox"] = bbox
            matched_track["emb"] = emb
            matched_track["last_frame"] = frame_idx

        assigned_ids.append(matched_track["track_id"])

    df["track_id"] = assigned_ids
    df.drop(columns=["_frame_idx"], inplace=True)
    return df


@task(name="Tracklet Task")
def tracklet_task(
    df: pd.DataFrame, iou_threshold: float = 0.3, cos_threshold: float = 0.6
) -> pd.DataFrame:
    """Prefect wrapper around :func:`link_tracklets`."""
    return link_tracklets(df, iou_threshold, cos_threshold)
