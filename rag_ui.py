import streamlit as st
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

# APIキー読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"))

# インデックスとデータを一度だけ読み込み
@st.cache_resource
def load_rag_data():
    index = faiss.read_index("rag_faiss.index")
    df = pd.read_pickle("rag_records.pkl")
    return index, df

index, df = load_rag_data()

# UI構成
st.title("🧩 障害診断チャットボット（RAG版）")

query_process = st.text_input("障害工程を入力してください", "")
query_failure = st.text_input("障害内容を入力してください", "")

if st.button("診断する") and query_process and query_failure:
    query = f"障害工程: {query_process}, 障害内容: {query_failure}"
    
    # ベクトル化
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_vec = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

    # FAISS検索（上位3件）
    D, I = index.search(query_vec, k=3)
    similar_records = df.iloc[I[0]]
    examples_text = "\n\n".join(similar_records["text"].tolist())

    # GPT用プロンプト
    prompt = f"""
あなたは製造現場の障害診断アシスタントです。
以下の障害履歴を参考にして、障害工程「{query_process}」および障害内容「{query_failure}」に対して、
最も可能性の高い「問題工程」「障害原因」「対処内容」を最大3件推定してください。
その上で、なぜそれを推定したのかを自然文で説明してください。

【重要】
- 回答は以下の履歴データに記載された内容のみに限定してください。
- 記載のない工程や原因・対処を使ってはいけません。
- どの履歴を根拠としたか（履歴1〜3など）も明記してください。

履歴：
{examples_text}

【出力形式】
推定：
- 問題工程: ○○
- 障害原因: ○○
- 対処内容: ○○

理由：
〜〜
"""

    gpt_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("🧠 GPTの診断結果")
    st.markdown(gpt_response.choices[0].message.content)

