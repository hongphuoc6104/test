# Cluster Evaluation

This script compares predicted character clusters with human labels.

## Labeling Format

Create a CSV file with two columns:

```
frame,character_id
movie1/frame_0001.jpg,0
movie1/frame_0002.jpg,1
```

Each row specifies the correct `character_id` for a given frame.

## Evaluation

Run the evaluation script:

```bash
python scripts/evaluate_clusters.py warehouse/characters.json labels.csv --out reports/eval_metrics.csv
```

The script reads `characters.json` to determine the predicted character for each
frame and compares it to the labeled CSV. It reports precision, recall and F1
for every character and an overall summary in `reports/eval_metrics.csv`.