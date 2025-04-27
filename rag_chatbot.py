import faiss
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# .envからAPIキーを読み込み（もしくは直書きでもOK）
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# インデックスと履歴データを読み込み
index = faiss.read_index("rag_faiss.index")
df = pd.read_pickle("rag_records.pkl")

# 入力受付
query_process = input("障害工程を入力してください：")     # 例: 組み立て
query_failure = input("障害内容を入力してください：")     # 例: 停止
query = f"障害工程: {query_process}, 障害内容: {query_failure}"

# クエリをEmbeddingベクトルに変換
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[query]
)
query_vec = np.array(embedding_response.data[0].embedding).astype("float32").reshape(1, -1)

# FAISSで類似検索（トップ3件）
D, I = index.search(query_vec, k=3)
similar_records = df.iloc[I[0]]

# GPTに渡すプロンプトを構築
examples_text = "\n\n".join(similar_records["text"].tolist())

prompt = f"""
あなたは製造現場の障害診断アシスタントです。
以下に示す過去の障害履歴のみを根拠として、障害工程「{query_process}」、障害内容「{query_failure}」に対して、
最も可能性の高い「問題工程」「障害原因」「対処内容」を最大3件ご提案してください。
その上で、なぜそれを推定したのかを自然文で説明してください。

【重要】
- 回答は以下の履歴データに記載された情報に**限定**してください。
- 履歴に登場しない工程名・障害原因・対処方法は**絶対に使わないでください**。
- 推定の理由では、どの履歴を根拠としたかを明記してください（例：「履歴2より判断」など）。

以下が参考とすべき履歴です：

{examples_text}

【出力形式】
推定：
- 問題工程: ○○
- 障害原因: ○○
- 対処内容: ○○

理由：
〜〜（参考にした履歴番号を含めて根拠を記載）
"""

# OpenAI Chat API (v1以降)
response = client.chat.completions.create(
    model="gpt-4o",  # または "gpt-3.5-turbo"
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# 結果表示
print("\n=== ChatGPTの提案 ===")
print(response.choices[0].message.content)
