from typing import List, Dict
from prefect import task

@task
def filter_clusters(clusters: List[Dict]) -> List[Dict]:
    """
    Remove clusters that fail basic quality thresholds.

    Criteria:
    - fewer than 3 tracklets
    - average detection score below 0.6
    - total frames across tracklets fewer than 5
    """
    cleaned = []
    for cluster in clusters:
        tracklets = cluster.get("tracklets", [])
        if len(tracklets) < 3:
            continue

        det_scores = [t.get("det_score", 0.0) for t in tracklets]
        if det_scores and (sum(det_scores) / len(det_scores)) < 0.6:
            continue

        total_frames = sum(t.get("frame_count", 0) for t in tracklets)
        if total_frames < 5:
            continue

        cleaned.append(cluster)

    return cleaned

filter_clusters_task = filter_clusters
