# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import csv
import os

import datasets


_CITATION = """\
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard and Perelygin, Alex and Wu, Jean and
      Chuang, Jason and Manning, Christopher D. and Ng, Andrew and Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
"""

_DESCRIPTION = """\
The Stanford Sentiment Treebank, the first corpus with fully labeled parse trees that allows for a
complete analysis of the compositional effects of sentiment in language.
"""

_HOMEPAGE = "https://nlp.stanford.edu/sentiment/"

_LICENSE = ""

_DEFAULT_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
_PTB_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"


class Sst(datasets.GeneratorBasedBuilder):
    """The Stanford Sentiment Treebank"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Sentences and relative parse trees annotated with sentiment labels.",
        ),
        datasets.BuilderConfig(
            name="dictionary",
            version=VERSION,
            description="List of all possible sub-sentences (phrases) with their sentiment label.",
        ),
        datasets.BuilderConfig(
            name="ptb",
            version=VERSION,
            description="Penn Treebank-formatted trees with labelled sub-sentences.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):

        if self.config.name == "default":
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("float"),
                    "tokens": datasets.Value("string"),
                    "tree": datasets.Value("string"),
                }
            )
        elif self.config.name == "dictionary":
            features = datasets.Features(
                {"phrase": datasets.Value("string"), "label": datasets.Value("float")}
            )
        else:
            features = datasets.Features(
                {
                    "ptb_tree": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        default_dir = dl_manager.download_and_extract(_DEFAULT_URL)
        ptb_dir = dl_manager.download_and_extract(_PTB_URL)

        file_paths = {}
        for split_index in range(0, 4):
            file_paths[split_index] = {
                "phrases_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/dictionary.txt"
                ),
                "labels_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/sentiment_labels.txt"
                ),
                "tokens_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/SOStr.txt"
                ),
                "trees_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/STree.txt"
                ),
                "splits_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/datasetSplit.txt"
                ),
                "sentences_path": os.path.join(
                    default_dir, "stanfordSentimentTreebank/datasetSentences.txt"
                ),
                "ptb_filepath": None,
                "split_id": str(split_index),
            }

        ptb_file_paths = {}
        for ptb_split in ["train", "dev", "test"]:
            ptb_file_paths[ptb_split] = {
                "phrases_path": None,
                "labels_path": None,
                "tokens_path": None,
                "trees_path": None,
                "splits_path": None,
                "sentences_path": None,
                "ptb_filepath": os.path.join(ptb_dir, "trees/" + ptb_split + ".txt"),
                "split_id": None,
            }

        if self.config.name == "default":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs=file_paths[1]
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, gen_kwargs=file_paths[3]
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs=file_paths[2]
                ),
            ]
        elif self.config.name == "dictionary":
            return [
                datasets.SplitGenerator(name="dictionary", gen_kwargs=file_paths[0])
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs=ptb_file_paths["train"]
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, gen_kwargs=ptb_file_paths["dev"]
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs=ptb_file_paths["test"]
                ),
            ]

    def _generate_examples(
        self,
        phrases_path,
        labels_path,
        tokens_path,
        trees_path,
        splits_path,
        sentences_path,
        split_id,
        ptb_filepath,
    ):

        if self.config.name == "ptb":
            with open(ptb_filepath, encoding="utf-8") as fp:
                ptb_reader = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
                for id_, row in enumerate(ptb_reader):
                    yield id_, {"ptb_tree": row[0]}
        else:
            labels = {}
            phrases = {}
            with open(labels_path, encoding="utf-8") as g, open(
                phrases_path, encoding="utf-8"
            ) as f:
                label_reader = csv.DictReader(g, delimiter="|", quoting=csv.QUOTE_NONE)
                for row in label_reader:
                    labels[row["phrase ids"]] = float(row["sentiment values"])

                phrase_reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
                if self.config.name == "dictionary":
                    for id_, row in enumerate(phrase_reader):
                        yield id_, {"phrase": row[0], "label": labels[row[1]]}
                else:
                    for row in phrase_reader:
                        phrases[row[0]] = labels[row[1]]

            # Case config=="default"
            # Read parse trees for each complete sentence
            trees = {}
            with open(tokens_path, encoding="utf-8") as tok, open(
                trees_path, encoding="utf-8"
            ) as tr:
                tok_reader = csv.reader(tok, delimiter="\t", quoting=csv.QUOTE_NONE)
                tree_reader = csv.reader(tr, delimiter="\t", quoting=csv.QUOTE_NONE)
                for i, row in enumerate(tok_reader, start=1):
                    trees[i] = {}
                    trees[i]["tokens"] = row[0]
                for i, row in enumerate(tree_reader, start=1):
                    trees[i]["tree"] = row[0]

            with open(splits_path, encoding="utf-8") as spl, open(
                sentences_path, encoding="utf-8"
            ) as snt:
                splits_reader = csv.DictReader(
                    spl, delimiter=",", quoting=csv.QUOTE_NONE
                )
                splits = {
                    row["sentence_index"]: row["splitset_label"]
                    for row in splits_reader
                }

                sentence_reader = csv.DictReader(
                    snt, delimiter="\t", quoting=csv.QUOTE_NONE
                )
                for id_, row in enumerate(sentence_reader):
                    # fix encoding, see https://github.com/huggingface/datasets/pull/1961#discussion_r585969890
                    row["sentence"] = (
                        row["sentence"]
                        .encode("utf-8")
                        .replace(b"\xc3\x83\xc2", b"\xc3")
                        .replace(b"\xc3\x82\xc2", b"\xc2")
                        .decode("utf-8")
                    )
                    row["sentence"] = (
                        row["sentence"].replace("-LRB-", "(").replace("-RRB-", ")")
                    )
                    if splits[row["sentence_index"]] == split_id:
                        tokens = trees[int(row["sentence_index"])]["tokens"]
                        parse_tree = trees[int(row["sentence_index"])]["tree"]
                        yield id_, {
                            "sentence": row["sentence"],
                            "label": phrases[row["sentence"]],
                            "tokens": tokens,
                            "tree": parse_tree,
                        }


sst = Sst()  # For easy access in other scripts such as setup.py, etc.

# download data into train.csv and test.csv
if __name__ == "__main__":
    import tempfile
    import shutil
    import urllib.request
    import zipfile

    print("Downloading SST dataset...")

    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Manual download and processing
            print("Downloading from Stanford...")
            zip_path = os.path.join(temp_dir, "sst.zip")
            urllib.request.urlretrieve(_DEFAULT_URL, zip_path)

            print("Extracting files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Process the files manually
            sst_dir = os.path.join(temp_dir, "stanfordSentimentTreebank")

            # Read sentences and labels
            sentences = {}
            with open(
                os.path.join(sst_dir, "datasetSentences.txt"), "r", encoding="utf-8"
            ) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx = parts[0]
                        sentence = "\t".join(parts[1:])
                        # Clean sentence
                        sentence = sentence.replace("-LRB-", "(").replace("-RRB-", ")")
                        sentences[idx] = sentence

            # Read splits
            splits = {}
            with open(
                os.path.join(sst_dir, "datasetSplit.txt"), "r", encoding="utf-8"
            ) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        idx, split_id = parts[0], parts[1]
                        splits[idx] = split_id

            # Read phrase labels
            phrase_labels = {}
            with open(
                os.path.join(sst_dir, "sentiment_labels.txt"), "r", encoding="utf-8"
            ) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        phrase_id, label = parts[0], float(parts[1])
                        phrase_labels[phrase_id] = label

            # Read phrase dictionary
            phrase_dict = {}
            with open(
                os.path.join(sst_dir, "dictionary.txt"), "r", encoding="utf-8"
            ) as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        phrase, phrase_id = parts[0], parts[1]
                        if phrase_id in phrase_labels:
                            phrase_dict[phrase] = phrase_labels[phrase_id]

            # Create datasets
            train_data, val_data, test_data = [], [], []

            for idx, sentence in sentences.items():
                if idx in splits and sentence in phrase_dict:
                    split_id = splits[idx]
                    raw_label = phrase_dict[sentence]
                    binary_label = int(raw_label > 0.5)

                    data_point = [sentence, binary_label, raw_label]

                    if split_id == "1":  # train
                        train_data.append(data_point)
                    elif split_id == "3":  # validation
                        val_data.append(data_point)
                    elif split_id == "2":  # test
                        test_data.append(data_point)

            # Save to CSV files
            output_dir = os.path.dirname(os.path.abspath(__file__))

            # Save train.csv
            with open(
                os.path.join(output_dir, "train.csv"), "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label", "raw_label"])  # header
                writer.writerows(train_data)

            # Save validation.csv
            with open(
                os.path.join(output_dir, "validation.csv"),
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label", "raw_label"])  # header
                writer.writerows(val_data)

            # Save test.csv
            with open(
                os.path.join(output_dir, "test.csv"), "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label", "raw_label"])  # header
                writer.writerows(test_data)

            print(f"Download completed successfully!")
            print(f"Train samples: {len(train_data)}")
            print(f"Validation samples: {len(val_data)}")
            print(f"Test samples: {len(test_data)}")
            print(f"Files saved in: {output_dir}")
            print("Files created: train.csv, validation.csv, test.csv")

        except Exception as e:
            print(f"Error downloading data: {e}")
            import traceback

            traceback.print_exc()

    print("Data downloaded and extracted.")
    print(sst.info)
