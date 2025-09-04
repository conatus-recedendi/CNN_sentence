import os
import re
import urllib.request
import zipfile
from typing import List, Tuple

# [ +3 ] 또는 [ -2 ] 추출 (부호 필수)
LABEL_RE = re.compile(r"\[\s*([+-])\s*(\d+)\s*\]")


def parse_grouped_file(path: str, encoding: str = "utf-8") -> List[Tuple[int, str]]:
    """
    파일을 [t] 제목 기준으로 블록화하여 파싱.
    각 블록의 설명 라인에 나오는 라벨 숫자(부호 포함)를 모두 합산해
    블록 라벨(>0 -> 1, <=0 -> 0)을 결정하고,
    블록의 모든 설명 라인에 해당 라벨을 붙여 (label, sentence)로 반환.
    """
    results: List[Tuple[int, str]] = []

    # 현재 블록 누적
    current_sentences: List[str] = []
    current_sum: int = 0
    in_block: bool = False  # [t]를 한 번이라도 본 이후인지

    def flush_block():
        nonlocal results, current_sentences, current_sum, in_block
        if not in_block or not current_sentences:
            # [t] 이전에 떠도는 설명 라인이 있더라도 수집하지 않음
            current_sentences = []
            current_sum = 0
            return
        label = 1 if current_sum > 0 else 0  # 0도 0으로 처리
        for s in current_sentences:
            results.append((label, s))
        # 블록 초기화
        current_sentences = []
        current_sum = 0

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("[t]"):
                # 이전 블록 마감 후 새 블록 시작
                flush_block()
                in_block = True
                continue

            # 설명 라인 후보: " ... ]##문장 " 형태만 수집
            # print(line)
            # enter wait
            # input()
            if "##" not in line:
                # print("skip: no ##")
                continue

            left, right = line.split("##", 1)
            right = right.strip()
            if not right:
                # print("skip: no right")
                continue

            # 설명 라인 왼쪽에서 라벨(복수 가능) 모두합
            labels = LABEL_RE.findall(left)
            # 라벨이 없는 설명 라인은 스킵 (원칙에 맞춤)
            # if not labels:
            #     continue

            # 부호*숫자를 누적
            line_sum = 0
            for sign, num in labels:
                val = int(num)
                line_sum += val if sign == "+" else -val
            # print("line_sum =", line_sum)
            # print("right  =", right)
            current_sentences.append(right)
            current_sum += line_sum

    # 파일 끝에서 마지막 블록 마감
    flush_block()
    return results


if __name__ == "__main__":
    url = "http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip"
    local_path = "CustomerReviewData.zip"
    out_dir = "./CustomerReviewData"

    print("Downloading and extracting the CR dataset...")

    try:
        # 1) 다운로드
        urllib.request.urlretrieve(url, local_path)
        print("Download completed!")

        # 2) ZIP 해제 (표준 라이브러리 사용)
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(out_dir)
        print("Extraction completed!")

        # 3) 모든 .txt 파싱 (예: CustomerReviewData/customer review data/*.txt)
        base_dir = os.path.join(out_dir, "customer review data")
        all_pairs: List[Tuple[int, str]] = []
        for root, _, files in os.walk(base_dir):
            for name in files:
                if name.lower().endswith(".txt"):
                    fp = os.path.join(root, name)
                    print(f"[parse] {fp}")
                    all_pairs.extend(parse_grouped_file(fp, encoding="utf-8"))

        # 4) train/test 분할 (임의: 9:1)
        train, test = [], []
        for i, (y, s) in enumerate(all_pairs):
            (test if i % 10 == 0 else train).append((y, s))

        # 5) 저장
        import pandas as pd

        pd.DataFrame(train, columns=["label", "sentence"]).to_csv(
            "train.csv", index=False, header=False
        )
        pd.DataFrame(test, columns=["label", "sentence"]).to_csv(
            "test.csv", index=False, header=False
        )

        print(f"Done. train={len(train)}, test={len(test)}")

    except Exception as e:
        print(f"[ERROR] Failed: {e}")
    finally:
        print("Process finished.")
        if os.path.exists(local_path):
            os.remove(local_path)
