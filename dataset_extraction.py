import tarfile
from datetime import datetime
import pandas as pd
import os
import shutil


# Crea carpetas
os.makedirs("./errors")
os.makedirs("./predictions")
os.makedirs("./raw/train")
os.makedirs("./raw/inference")
os.makedirs("./processed/train")
os.makedirs("./processed/inference")
os.makedirs("./etl/train")
os.makedirs("./etl/inference")


now_str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

amazon_reviews = tarfile.open("./dataset/amazon_review_polarity_csv.tgz")
amazon_reviews.extractall("data")
amazon_reviews.close()

df_train = pd.read_csv("./data/amazon_review_polarity_csv/train.csv",names=["score","title","text"])
# df_train_short = df_train.iloc[:150000,:].copy()
df_train_short = df_train.iloc[:10000,:].copy()
df_train_short.to_csv(f"./raw/train/train_{now_str}.csv",index=False)

df_inference = pd.read_csv("./data/amazon_review_polarity_csv/test.csv",names=["score","title","text"])
# df_inference_save = df_inference[["text"]]
df_inference_save = df_inference[["text"]].iloc[:10000,:].copy()
df_inference_save.to_csv(f"./raw/inference/inference_{now_str}.csv",index=False) 

if os.path.exists("./data"):
    shutil.rmtree("./data")