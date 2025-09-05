"""
PyTorch version of data processing for CNN sentence classification
Includes data cleaning, word2vec integration, and dataset creation
"""

import numpy as np
import pickle as pkl
import sys
import re
import warnings

warnings.filterwarnings("ignore")


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)

    with open(pos_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                "y": 1,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": np.random.randint(0, cv),
            }
            revs.append(datum)

    with open(neg_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                "y": 0,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": np.random.randint(0, cv),
            }
            revs.append(datum)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype="float32")
    W[0] = np.zeros(k, dtype="float32")
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
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
                    word = b"".join(word).decode("utf-8")
                    break
                if ch != b"\n":
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype="float32")
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    from collections import defaultdict
    import os

    # Set random seed for reproducibility
    np.random.seed(42)

    # Check if data files exist
    w2v_file = "GoogleNews-vectors-negative300.bin"
    pos_file = "rt-polarity.pos"
    neg_file = "rt-polarity.neg"

    print("Building dataset...")
    revs, vocab = build_data_cv([pos_file, neg_file], cv=10, clean_string=True)

    max_l = np.max([len(s["text"].split()) for s in revs])
    print(f"Data loaded!")
    print(f"Number of sentences: {len(revs)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Max sentence length: {max_l}")

    print("Loading word2vec vectors...")
    if os.path.exists(w2v_file):
        print(f"Loading word2vec from {w2v_file}...")
        w2v = load_bin_vec(w2v_file, vocab)
        print(f"Word2vec loaded!")
        print(f"Num words already in word2vec: {len(w2v)}")
    else:
        print(f"Word2vec file {w2v_file} not found. Creating random vectors...")
        w2v = {}

    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    # Create random vectors for comparison
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Word vectors shape: {W.shape}")

    # Save processed data
    print("Saving processed data...")
    with open("mr.p", "wb") as f:
        pkl.dump([revs, W, W2, word_idx_map, vocab], f)

    print("Dataset created successfully!")
    print("Usage: python conv_net_sentence_pytorch.py mr.p -nonstatic -rand")
    print("       python conv_net_sentence_pytorch.py mr.p -static -word2vec")
