from datasets import load_dataset
import pandas as pd

ds = load_dataset("jxm/mpqa")

if __name__ == "__main__":
    test = []
    train = []

    for example in ds["test"]:
        test.append((example["label"], example["text"]))

    for example in ds["train"]:
        train.append((example["label"], example["text"]))

    df_train = pd.DataFrame(train, columns=["label", "sentence"])
    df_test = pd.DataFrame(test, columns=["label", "sentence"])

    df_train.to_csv("train.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
