import os
import urllib.request

# import rarfile  # pip install rarfile
import patoolib

if __name__ == "__main__":
    url = "http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar"
    local_path = "CustomerReviewData.rar"

    print("Downloading and extracting the CR dataset...")

    try:
        # 다운로드
        urllib.request.urlretrieve(url, local_path)
        print("Download completed!")

        # RAR 압축 해제
        patoolib.extract_archive(local_path, outdir="./CustomerReviewData")
        # with rarfile.RarFile(local_path, "r") as rf:
        #     rf.extractall("./CustomerReviewData")
        print("Extraction completed!")

    except Exception as e:
        print(f"[ERROR] Failed to download or extract: {e}")

    finally:
        # 임시 파일 제거
        if os.path.exists(local_path):
            os.remove(local_path)
