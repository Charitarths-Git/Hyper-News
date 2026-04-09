import urllib.request
import zipfile
import os
import time
from sentence_transformers import SentenceTransformer

url = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
zip_path = "MINDsmall_train.zip"

print(f"Downloading {url} ...")
urllib.request.urlretrieve(url, zip_path)
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract("news.tsv")

print("Counting lines in news.tsv...")
count = 0
with open("news.tsv", "r", encoding="utf-8") as f:
    for _ in f: count += 1
print(f"Total articles: {count}")

print("Loading model and testing speed...")
model = SentenceTransformer("all-MiniLM-L6-v2")
test_texts = ["This is a test article string repeated."] * 1000

start = time.time()
_ = model.encode(test_texts, batch_size=32)
end = time.time()
print(f"Time for 1000 vectors: {end - start:.2f}s")
print(f"Estimated time for {count}: {(end-start) * (count/1000) / 60:.2f} mins")
