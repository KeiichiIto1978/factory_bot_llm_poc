import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

# .envファイルからAPIキーを読み込む（もしくは直書きでもOK）
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# データの読み込み（CSV）
df = pd.read_csv("./data/failure_history_cleaned_no_cause.csv")

# 履歴を自然文に変換する関数
def convert_to_text(row):
    return (
        f"障害工程: {row['障害工程']}, "
        f"障害内容: {row['障害内容']}, "
        f"問題工程: {row['問題工程']}, "
        f"障害原因: {row['障害原因']}, "
        f"対処内容: {row['対処内容']}"
    )

# 各履歴を自然文化
texts = df.apply(convert_to_text, axis=1).tolist()

# OpenAI Embedding APIでベクトル化
def get_embeddings(texts, model="text-embedding-3-small"):
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(input=batch, model=model)
        for e in res.data:
            embeddings.append(e.embedding)
    return np.array(embeddings).astype("float32")

# 埋め込み生成
embeddings = get_embeddings(texts)
embedding_dim = len(embeddings[0])

# FAISSインデックス作成
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# 結果を保存
df["text"] = texts
df.to_pickle("rag_records.pkl")
faiss.write_index(index, "rag_faiss.index")

print("✅ ベクトル生成とFAISSインデックスの保存が完了しました。")
