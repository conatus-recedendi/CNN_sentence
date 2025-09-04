# subjectivity dataset v1.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz

import random
import pandas as pd
import os
import tarfile
import urllib.request


if __name__ == "__main__":
    url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"

    urllib.request.urlretrieve(url, "rotten_imdb.tar.gz")
    tar = tarfile.open("rotten_imdb.tar.gz", "r:gz")
    tar.extractall(path="./")
    tar.close()
    os.remove("rotten_imdb.tar.gz")

    data = []
    # subjective
    with open("quote.tok.gt9.5000", "r", encoding="latin-1") as f:
        for line in f:
            sentence = line.strip()
            data.append((1, sentence[1]))

    # objective
    with open("plot.tok.gt9.5000", "r", encoding="latin-1") as f:
        for line in f:
            sentence = line.strip()
            data.append((0, sentence[1]))

    test = []
    train = []

    for i in range(len(data)):
        ran = random.randint(0, 9)
        if ran == 0:
            test.append(data[i])
        else:
            train.append(data[i])

    df_train = pd.DataFrame(train, columns=["label", "sentence"])
    df_test = pd.DataFrame(test, columns=["label", "sentence"])

    df_train.to_csv("train.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)

# http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
