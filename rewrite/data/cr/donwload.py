# empty


# read rt-polarity.neg
# read rt-polarity.pos

# data,label 꼴의 csv로 만들어야 함.
# 또한 단일 파일로 저장하되, train/test로 나누어야 함

import pandas as pd
import random
import os
import urllib.request
import zipfile

if __name__ == "__main__":
    print("Downloading and extracting the MR dataset...")
    url = "https://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip"

    urllib.request.urlretrieve(url, "CustomerReviewData.zip")
    with zipfile.ZipFile("CustomerReviewData.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove("CustomerReviewData.zip")

    # "

    # read from ./rt-polarity.neg and ./rt-polarity.pos
    # https://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
    # unzip it
    # then split it into train and test
    # then read the two files
    # 10% for test, 90% for train
