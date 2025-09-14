import os
import time
from flows.pipeline import main_pipeline
from warehouse.update_pipeline import update_pipeline


def benchmark(new_parquet: str, out_file: str = "docs/perf.md") -> None:
    """Benchmark full vs incremental pipeline and write results to markdown."""
    start = time.perf_counter()
    main_pipeline()
    full_time = time.perf_counter() - start

    start = time.perf_counter()
    update_pipeline(new_parquet)
    inc_time = time.perf_counter() - start

    lines = [
        "# Pipeline Performance",
        "",
        "| Pipeline | Time (s) |",
        "|----------|---------|",
        f"| Full | {full_time:.2f} |",
        f"| Incremental | {inc_time:.2f} |",
        "",
    ]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark full vs incremental pipeline")
    parser.add_argument("parquet", help="Path to new parquet for incremental run")
    parser.add_argument("--out", default="docs/perf.md", help="Where to write markdown output")
    args = parser.parse_args()
    benchmark(args.parquet, args.out)
