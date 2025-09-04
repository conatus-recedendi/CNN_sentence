from datasets import load_dataset
import pandas as pd

ds = load_dataset("SetFit/sst2")

if __name__ == "__main__":
    test = []
    train = []
    valid = []

    for example in ds["test"]:
        test.append((example["label"], example["text"]))

    for example in ds["train"]:
        train.append((example["label"], example["text"]))

    for example in ds["validation"]:
        valid.append((example["label"], example["text"]))

    df_train = pd.DataFrame(train, columns=["label", "text"])
    df_test = pd.DataFrame(test, columns=["label", "text"])
    df_valid = pd.DataFrame(ds["validation"], columns=["label", "text"])

    df_train.to_csv("train.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
    df_valid.to_csv("valid.csv", index=False, header=False)
