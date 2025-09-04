# https://huggingface.co/datasets/stanfordnlp/sst2
from datasets import load_dataset
import pandas as pd


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/sst2")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["test"]
    # del dataset["test"]
    for split in ["train", "test"]:
        df = dataset[split].to_pandas()
        df = df.drop(0)
        df = df[["label", "sentence"]]
        # 2개로 분류
        df["label"] = pd.cut(df["label"], bins=[-0.1, 0.5, 1.1], labels=[0, 1])
        df.to_csv(f"{split}.csv", index=False, header=False)
