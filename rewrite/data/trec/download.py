from datasets import load_dataset

ds = load_dataset("SetFit/TREC-QC")

import pandas as pd
import random

if __name__ == "__main__":
    test = []
    train = []
    valid = []

    for example in ds["test"]:
        test.append((example["label_coarse"], example["text"]))

    for example in ds["train"]:
        ran = random.randint(0, 9)
        if ran == 0:
            valid.append((example["label_coarse"], example["text"]))
        else:
            train.append((example["label_coarse"], example["text"]))

    # for example in ds["validation"]:
    #     valid.append((example["label"], example["text"]))

    df_train = pd.DataFrame(train, columns=["label", "text"])
    df_test = pd.DataFrame(test, columns=["label", "text"])
    df_valid = pd.DataFrame(valid, columns=["label", "text"])

    df_train.to_csv("train.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
    df_valid.to_csv("validation.csv", index=False, header=False)
