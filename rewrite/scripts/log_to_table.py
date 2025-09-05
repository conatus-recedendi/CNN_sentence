#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_to_table.py
여러 .log 파일에서 (dataset, model, final test performance)를 추출해
논문 표 형태(MR, SST-1, SST-2, Subj, TREC, CR, MPQA)의 CSV로 저장합니다.
"""

import re
import sys
import glob
import argparse
from collections import defaultdict, OrderedDict
import pandas as pd

DATASET_MAP = {
    "mr": "MR",
    "sst-1": "SST-1",
    "sst1": "SST-1",
    "sst-2": "SST-2",
    "sst2": "SST-2",
    "subj": "Subj",
    "trec": "TREC",
    "cr": "CR",
    "mpqa": "MPQA",
}
COLUMNS = ["MR", "SST-1", "SST-2", "Subj", "TREC", "CR", "MPQA"]
ROW_ORDER = ["CNN-rand", "CNN-static", "CNN-nonstatic", "CNN-multichannel"]


# ---------- 파서 ----------
def detect_dataset(text: str):
    # 1) "Processing <name> dataset"
    m = re.search(r"Processing\s+([A-Za-z0-9\-]+)\s+dataset", text, re.IGNORECASE)
    if m:
        return DATASET_MAP.get(m.group(1).lower())
    # 2) "./data/<name>/test.csv"
    m = re.search(r"\./data/([a-z0-9\-]+)/test\.csv", text, re.IGNORECASE)
    if m:
        return DATASET_MAP.get(m.group(1).lower())
    return None


def detect_model(text: str):
    # 기본 구조
    arch = None
    if re.search(r"Model architecture:\s*CNN-static", text, re.IGNORECASE):
        arch = "CNN-static"
    elif re.search(r"Model architecture:\s*CNN-?non-?static", text, re.IGNORECASE):
        arch = "CNN-non-static"
    elif re.search(r"Model architecture:\s*CNN-multichannel", text, re.IGNORECASE):
        arch = "CNN-multichannel"

    using_random = bool(re.search(r"Using:\s*random vectors", text, re.IGNORECASE))

    # 매핑 규칙
    if arch == "CNN-non-static" and using_random:
        return "CNN-rand"
    if arch == "CNN-non-static" and not using_random:
        return "CNN-nonstatic"
    if arch == "CNN-static":
        return "CNN-static"
    if arch == "CNN-multichannel":
        return "CNN-multichannel"
    return None


def detect_score(text: str):
    # 'Final test performance' 우선, 없으면 'Test accuracy'
    m = re.search(r"Final test performance:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Test accuracy:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not m:
        return None
    return round(float(m.group(1)) * 100.0, 1)  # %로 환산, 소수 1자리


def parse_runs(text: str):
    """
    하나의 로그 텍스트에서 여러 번의 실행 블록을 추출.
    각 블록은 'Final test performance'로 종료된다고 가정.
    """
    runs = []
    cur = []
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if "Final test performance:" in line:
            runs.append("".join(cur))
            cur = []
    return runs


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="읽을 .log 경로(글롭 허용: logs/*.log)")
    ap.add_argument(
        "-o", "--output", default="cnn_results_table.csv", help="출력 CSV 파일명"
    )
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    if not files:
        print("No log files found.")
        sys.exit(1)

    # 누적: model -> {dataset: score}
    acc = defaultdict(dict)

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for block in parse_runs(txt):
            ds = detect_dataset(block)
            model = detect_model(block)
            score = detect_score(block)
            if ds and model and score is not None:
                # 같은 (model, ds)가 여러 번 나오면 **가장 마지막** 값을 남김
                acc[model][ds] = score

    # 표 구성
    rows = OrderedDict()
    for model in ROW_ORDER:
        if model in acc:
            rows[model] = [acc[model].get(col, "") for col in COLUMNS]

    if not rows:
        print("No (dataset, model, score) triples were found in the logs.")
        sys.exit(2)

    df = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
    df.index.name = "Model"
    df.to_csv(args.output)
    print(f"Saved: {args.output}")
    print(df.to_markdown())


if __name__ == "__main__":
    main()
