# 확장: 표를 만들 때, "논문 표(ground truth)" 값과 로그 결과를 비교하여 차이를 함께 표시
# 논문 값은 이미지에 나와 있는 표를 직접 코드에 하드코딩하여 기준으로 사용한다.

import pandas as pd

# 기준값 (이미지 표에서 추출)
ground_truth = {
    "CNN-rand": {
        "MR": 76.1,
        "SST-1": 45.0,
        "SST-2": 82.7,
        "Subj": 89.6,
        "TREC": 91.2,
        "CR": 79.8,
        "MPQA": 83.4,
    },
    "CNN-static": {
        "MR": 81.0,
        "SST-1": 45.5,
        "SST-2": 86.8,
        "Subj": 93.0,
        "TREC": 92.8,
        "CR": 84.7,
        "MPQA": 89.6,
    },
    "CNN-nonstatic": {
        "MR": 81.5,
        "SST-1": 48.0,
        "SST-2": 87.2,
        "Subj": 93.4,
        "TREC": 93.6,
        "CR": 84.3,
        "MPQA": 89.5,
    },
    "CNN-multichannel": {
        "MR": 81.1,
        "SST-1": 47.4,
        "SST-2": 88.1,
        "Subj": 93.2,
        "TREC": 92.2,
        "CR": 85.0,
        "MPQA": 89.4,
    },
}

# 예시: 앞서 파싱한 결과 (df)
df_results = df.copy()

# 기준값 DataFrame
df_truth = pd.DataFrame.from_dict(ground_truth, orient="index")[df_results.columns]

# 차이 계산
df_diff = df_results.astype(float).sub(df_truth, fill_value=float("nan"))

# 결과 합치기: "실험값 (차이)" 형태로 출력
df_combined = df_results.copy()
for col in df_results.columns:
    for idx in df_results.index:
        val = df_results.at[idx, col]
        if val == "":  # 값이 없는 경우 pass
            continue
        truth_val = df_truth.at[idx, col]
        if pd.notna(truth_val):
            diff = val - truth_val
            # 부호 표시
            diff_str = f"{diff:+.1f}"
            df_combined.at[idx, col] = f"{val:.1f} ({diff_str})"
        else:
            df_combined.at[idx, col] = f"{val:.1f} (NA)"

import os

csv_path_diff = "/mnt/data/cnn_results_with_diff.csv"
df_combined.to_csv(csv_path_diff)

import caas_jupyter_tools

caas_jupyter_tools.display_dataframe_to_user(
    "CNN Results with Differences", df_combined
)

csv_path_diff
