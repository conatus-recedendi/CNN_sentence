# This script scans subfolders in ./data, reads train.csv/test.csv,
# and computes dataset statistics matching the requested table.
# If a pre-trained embedding file exists at ./embeddings.vec (word2vec/FastText .vec format),
# it will compute |Vpre| as the number of vocab words present in the embeddings.
# Otherwise, |Vpre| is set to None.
import os, re, pandas as pd
from pathlib import Path
import numpy as np

DATA_ROOT = Path("../data")


def read_split_csv(path: Path):
    # CSV with no header: first field is label, rest is text (may contain commas)
    # Use engine 'python' to allow variable number of fields; join the remainder.
    if not path.exists():
        return pd.DataFrame(columns=["label", "text"])
    df = pd.read_csv(
        path,
        header=None,
        engine="python",
        names=list(range(0, 64)),  # generous columns; we'll combine later
        dtype=str,
    )
    # First column is label
    df["label"] = df.iloc[:, 0]
    # Remaining columns joined by comma back to text (ignoring NaNs)
    text_cols = df.columns[1:]
    df["text"] = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .apply(lambda row: ",".join([c for c in row if c != ""]), axis=1)
    )
    return df[["label", "text"]]


token_re = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(s: str):
    return token_re.findall(str(s).lower())


def stats_for_dataset(ds_dir: Path, embeddings_vocab):
    train = read_split_csv(ds_dir / "train.csv")
    test = read_split_csv(ds_dir / "test.csv")
    valid = read_split_csv(ds_dir / "validation.csv")
    all_df = pd.concat([train, test, valid], ignore_index=True)
    # Classes (unique labels in combined)
    c = all_df["label"].nunique(dropna=True)
    # Average sentence length over combined
    lengths = all_df["text"].map(lambda t: len(tokenize(t)))
    l = float(lengths.mean()) if len(lengths) else 0.0
    # Dataset size (train+test)
    N = int(len(all_df))
    # Vocabulary from combined
    vocab = {}
    for t in all_df["text"]:
        vocab.update(tokenize(t))
    V = len(vocab)
    # |Vpre| (intersect with embeddings vocab) if provided
    #
    Vpre = len(vocab.keys() & embeddings_vocab.keys())
    print(len(vocab))
    print(len(embeddings_vocab))
    print(len(Vpre))
    # print(vocab, embeddings_vocab, vocab & embeddings_vocab)
    # Test column formatted as "CV(n)"
    test_size = int(len(test))
    Test = f"CV({test_size})"
    return {
        "c": int(c),
        "l": round(
            l, 0
        ),  # match the style (integers shown); keep as number for display
        "N": int(N),
        "|V|": int(V),
        "|Vpre|": (int(Vpre) if Vpre is not None else None),
        "Test": Test,
    }


def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b" ":
                    word = b"".join(word)
                    break
                if ch != b"\n":
                    word.append(ch)
            # if word in vocab:
            word = word.decode("utf-8", errors="ignore")
            # word_vecs[word] = np.frombuffer(f.read(binary_len), dtype="float32")
            # word_vecs.add(word.decode("utf-8", errors="ignore"))
            word_vecs[word] = np.frombuffer(f.read(binary_len), dtype="float32")
            # else:
            #     f.read(binary_len)
    return word_vecs


def load_embeddings_vocab(vec_path: Path):
    if not vec_path.exists():
        return None
    vocab = set()
    with vec_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        # Some .vec files start with "<n_words> <dim>", detect and skip if so
        if re.match(r"^\s*\d+\s+\d+\s*$", first):
            pass  # header line consumed
        else:
            # first line was a word vector line; parse it
            parts = first.strip().split()
            if parts:
                vocab.add(parts[0])
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split()
            if parts:
                vocab.add(parts[0])
    return vocab


# Load embeddings once if available
# emb_vocab = load_embeddings_vocab(Path("./embeddings.vec"))
emb_vocab = load_bin_vec(Path("../GoogleNews-vectors-negative300.bin"))


# Discover datasets: immediate subfolders of ./data that contain any of train.csv/test.csv
rows = []
names = []
if DATA_ROOT.exists():
    for p in sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()]):
        has_any = (p / "train.csv").exists() or (p / "test.csv").exists()
        if not has_any:
            continue
        stats = stats_for_dataset(p, emb_vocab)
        rows.append(stats)
        names.append(p.name.upper())  # pretty name like MR, SST-1, etc.
else:
    rows, names = [], []

df_out = pd.DataFrame(
    rows, index=names, columns=["c", "l", "N", "|V|", "|Vpre|", "Test"]
)
# Save and display
out_csv = "p1_table1.csv"
df_out.to_csv(out_csv, encoding="utf-8")
# from caas_jupyter_tools import display_dataframe_to_user

# display_dataframe_to_user("Dataset statistics", df_out)
# out_csv
