# 🚀 GitHub RAG Chat (Hybrid Search + Reranking)

An advanced Retrieval-Augmented Generation (RAG) application that allows users to **chat with any GitHub repository** using natural language.

This project combines **semantic search, keyword search, and reranking** to deliver accurate, context-aware answers from codebases.

---

## 🔥 Features

* 🔗 Enter any GitHub repository URL
* 💬 Ask questions about code, architecture, or logic
* 🧠 Hybrid Retrieval:

  * Vector Search (semantic understanding)
  * BM25 (keyword matching)
* ⚡ Reranking using embedding similarity
* 📦 Dynamic repository ingestion
* 🖥️ Interactive UI using Streamlit
* 🔐 Environment-based configuration (.env support)

---

## 🏗️ Architecture

```
GitHub Repo → Loader → Chunking → Embeddings → Vector DB
                                                 ↓
User Query → Hybrid Search → Reranking → LLM → Answer
```

---

## 🧰 Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **RAG Framework**: LangChain (LCEL)
* **Embeddings**: OpenAI Embeddings
* **LLM**: OpenAI GPT models
* **Vector Store**: ChromaDB
* **Keyword Search**: BM25 (`rank-bm25`)
* **Version Control**: GitPython

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
├── Dockerfile
├── .env
├── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/github-rag.git
cd github-rag
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

---

### 5️⃣ Run the application

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### Build image

```bash
docker build -t github-rag-app .
```

### Run container

```bash
docker run -p 8501:8501 --env-file .env github-rag-app
```

---

## 🧪 Example Queries

* "What does this repository do?"
* "List all libraries used"
* "Explain the main model architecture"
* "Where is training implemented?"
* "How is authentication handled?"

---

## ⚡ Key Highlights

* Hybrid retrieval improves recall (semantic + keyword)
* Reranking improves precision
* Modular LCEL pipeline design
* Works with any public GitHub repository
* Scalable and production-ready

---

## 🚀 Future Improvements

* 🔍 File-level citations (filename + line numbers)
* 🧠 Conversational memory (multi-turn chat)
* ⚡ Streaming responses
* 🌐 Multi-repository support
* 🔐 GitHub OAuth for private repos

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


---

## 🙌 Acknowledgements

* LangChain
* OpenAI
* ChromaDB
* Streamlit

---

## 👨‍💻 Author

**Pradeep Gupta**
AI & Data Science Engineer

---
