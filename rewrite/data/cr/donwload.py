import os
import urllib.request

# import rarfile  # pip install rarfile
import patoolib

import re
from typing import List, Tuple, Iterable

# [ +3 ] 또는 [ -2 ] 같이 부호 필수, 숫자 1+ 자리
LABEL_RE = re.compile(r"\[\s*([+-])\s*(\d+)\s*\]")


def parse_line(line: str) -> List[Tuple[int, str]]:
    """
    한 줄을 규칙에 따라 (label, sentence) 쌍으로 변환.
    - label: + -> 1, - -> 0
    - sentence: 첫 번째 '##' 이후의 모든 텍스트 (strip 처리)
    조건 미충족 시 빈 리스트 반환.
    """
    if "##" not in line:
        return []
    left, right = line.split("##", 1)  # 첫 번째 ## 기준 분리
    labels = LABEL_RE.findall(left)
    if not labels:
        return []
        # return [(1, right.strip())]  # 라벨이 없으면 긍정으로 간주

    sentence = right.strip()
    # 부호만 중요(+ => 1, - => 0). 숫자 크기는 무시.
    out = []
    lables = lables[:1]
    for sign, _num in labels:
        y = 1 if sign == "+" else 0
        out.append((y, sentence))
    return out


def parse_file(path: str, encoding: str = "utf-8") -> List[Tuple[int, str]]:
    """
    파일 전체를 순회하여 (label, sentence) 리스트 반환.
    """
    results: List[Tuple[int, str]] = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            results.extend(parse_line(line))
    return results


# # ---- 사용 예시 ----
# if __name__ == "__main__":
#     # 데모 라인
#     demo = [
#         "[t]excellent phone , excellent service .",  # 수집 X (## 없음)
#         "##i am a business user who heavily depend on mobile service .",  # 수집 X (라벨 없음)
#         "phone[+3], work[+2]##there is much which has been said in other reviews about the features...",  # 수집 O (2개)
#         "bad[-1]  camera[-2]  ##terrible experience overall",  # 수집 O (2개)
#         "neutral[0]##meh",  # 수집 X (부호가 없음 -> 라벨로 취급하지 않음)
#         "ok [+1]##works fine",  # 수집 O (공백 허용)
#     ]
#     for s in demo:
#         print(s, "->", parse_line(s))


if __name__ == "__main__":
    url = "http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip"
    local_path = "CustomerReviewData.zip"

    print("Downloading and extracting the CR dataset...")

    try:
        # 다운로드
        urllib.request.urlretrieve(url, local_path)
        print("Download completed!")

        # 압축 해제
        patoolib.extract_archive(
            os.path.join(local_path), outdir="./CustomerReviewData"
        )
        print("Extraction completed!")

        # path.join(local_path, "customer review data") 안에 있는 모든 .txt 파일을 열고
        # with open()
        sentences = []
        labels = []

        with os.scandir(
            os.path.join("CustomerReviewData", "customer review data")
        ) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".txt"):
                    file_path = os.path.join(
                        "CustomerReviewData", "customer review data", entry.name
                    )
                    print(f"Parsing file: {file_path}")
                    parsed = parse_file(file_path, encoding="utf-8")
                    for label, sentence in parsed:
                        labels.append(label)
                        sentences.append(sentence)

        test = []
        train = []

        for i in range(len(sentences)):
            if i % 10 == 0:
                test.append((labels[i], sentences[i]))
            else:
                train.append((labels[i], sentences[i]))

        import pandas as pd

        df_train = pd.DataFrame(train, columns=["label", "sentence"])
        df_test = pd.DataFrame(test, columns=["label", "sentence"])
        df_train.to_csv("train.csv", index=False, header=False)
        df_test.to_csv("test.csv", index=False, header=False)

    except Exception as e:
        print(f"[ERROR] Failed to download or extract: {e}")

    finally:
        print("Process finished.")
        # 임시 파일 제거
        if os.path.exists(local_path):
            os.remove(local_path)
