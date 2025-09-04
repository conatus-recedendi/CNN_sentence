# empty


# read rt-polarity.neg
# read rt-polarity.pos

# data,label 꼴의 csv로 만들어야 함.
# 또한 단일 파일로 저장하되, train/test로 나누어야 함

import pandas as pd
import random

if __name__ == "__main__":
    print("Downloading and extracting the MR dataset...")

    # read from ./rt-polarity.neg and ./rt-polarity.pos

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

    test = []
    train = []
    valid = []

    for i in range(len(data)):
        ran = random.randint(0, 9)
        if ran == 0:
            test.append((labels[i], data[i]))
        else:
            ran = random.randint(0, 9)
            if ran == 0:
                valid.append((labels[i], data[i]))
            else:
                train.append((labels[i], data[i]))

    df_train = pd.DataFrame(train, columns=["label", "text"])
    df_test = pd.DataFrame(test, columns=["label", "text"])
    df_valid = pd.DataFrame(valid, columns=["label", "text"])

    df_train.to_csv("train.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
    df_valid.to_csv("validation.csv", index=False, header=False)
