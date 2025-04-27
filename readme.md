# Factory Failure Diagnosis Bot (RAG Version)

工場向け障害診断チャットボットのPoC実装です。  
小規模な障害履歴データ（例：100件）をもとに、ユーザーの入力に応じて過去の類似障害を検索し、AIが最適な「問題工程・障害原因・対処内容」を推測・提案します。  
検索には意味ベース（Embedding＋FAISS）を使い、出力にはOpenAI GPTモデル（gpt-3.5-turbo/gpt-4oなど）を利用しています。

---

## 📂 構成

| ファイル名                                       | 役割                        |
| ------------------------------------------- | ------------------------- |
| `rag_embed_index.py`                        | 障害履歴をベクトル化しFAISSインデックスを作成 |
| `rag_chatbot.py`                            | ターミナルから障害診断を行うチャットボット     |
| `rag_ui.py`                                 | StreamlitベースのWebチャットUI    |
| `data/failure_history_cleaned_no_cause.csv` | 入力となる障害履歴CSVデータ（例）        |
| `.env`                                      | OpenAI APIキーを管理するファイル     |

---

## 🛠️ セットアップ

このプロジェクトは **Poetry** を使って管理されています。

1. Poetryで依存ライブラリをインストールします。
   poetry install
   必要なパッケージは以下に含まれています：

2. .env ファイルを作成して、OpenAI APIキーを設定します。

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```



# 🚀 使い方

① 履歴データからFAISSインデックスを作成する


```
poetry run python rag_embed_index.py
```


これにより、次の2つのファイルが生成されます：

rag_faiss.index （ベクトルインデックス）

rag_records.pkl （自然文化した履歴データ）

② ターミナルチャットボット版を実行する



```
poetry run python rag_chatbot.py
```

プロンプトに従って「障害工程」「障害内容」を入力

OpenAI GPTが類似履歴をもとに、最も疑わしい問題工程・原因・対処を推測し、理由を説明します

③ Streamlit UI版を実行する



```
poetry run streamlit run rag_ui.py
```


ブラウザが開き、フォーム入力で診断ができるシンプルなWebチャットボットが起動します

# ✅ 特徴

✅ 意味ベースの類似検索（Embedding＋FAISS）

✅ GPTが履歴を参考にしながら自然文で推論・説明

✅ 出典（参考にした履歴番号）を明示

✅ 少ないデータでもPoCが可能

✅ ローカル開発にも対応（要インターネット接続）



# ⚠️ 注意事項

現状、外部API（OpenAI API）を使用しているため、インターネット接続が必要です。

工場や閉域環境で運用する場合は、オフラインLLM版への切り替えを検討してください。



# 📜 ライセンス

このプロジェクトはPoC目的のため、ライセンスは付与していません。

