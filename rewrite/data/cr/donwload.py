# empty


# read rt-polarity.neg
# read rt-polarity.pos

# data,label 꼴의 csv로 만들어야 함.
# 또한 단일 파일로 저장하되, train/test로 나누어야 함

import pandas as pd


if __name__ == "__main__":
    print("Downloading and extracting the CR dataset...")

    # read from ./rt-poloarity.neg and ./rt-polarity.pos

    # save to ./cr.csv
    pos_file = "rt-polarity.pos"
    neg_file = "rt-polarity.neg"
    data = []
    labels = []
    with open(pos_file, "r", encoding="cp1252") as f:
        for line in f:
            data.append(line.strip())
            labels.append(1)
    with open(neg_file, "r", encoding="cp1252") as f:
        for line in f:
            data.append(line.strip())
            labels.append(0)
    df = pd.DataFrame({"sentence": data, "label": labels})
    df.to_csv("cr.csv", index=False)
