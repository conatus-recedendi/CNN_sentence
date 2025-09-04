# subjectivity dataset v1.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz

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
