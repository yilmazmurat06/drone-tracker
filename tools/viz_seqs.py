#!/usr/bin/env python3
"""Create a combined visualization MP4 for specific trajectory sequences only."""
import argparse, csv, os
from collections import defaultdict
import subprocess, tempfile, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--seqs", required=True, help="comma-separated seq_ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_seqs = {int(s) for s in args.seqs.split(",")}
    path = args.video if os.path.exists(args.video) else args.video + ".mp4"

    # Create temp CSV with only target sequences
    tmp_csv = tempfile.mktemp(suffix=".csv")
    with open(args.csv) as fin, open(tmp_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["seq_id", "flight", "frame_idx", "x", "y", "w", "h", "conf"])
        for r in reader:
            if int(r["seq_id"]) in target_seqs:
                writer.writerow([r["seq_id"], r.get("flight", ""), r["frame_idx"],
                                 r["x"], r["y"], r["w"], r["h"], r["conf"]])

    # Run viz_trajectories.py on the temp CSV
    script = os.path.join(os.path.dirname(__file__), "viz_trajectories.py")
    subprocess.run([sys.executable, script, path, tmp_csv, "--out", args.out], check=True)
    os.unlink(tmp_csv)


if __name__ == "__main__":
    main()
