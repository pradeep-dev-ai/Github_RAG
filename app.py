import streamlit as st
import os
import shutil
from git import Repo



from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LangChain imports
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Hybrid search
from rank_bm25 import BM25Okapi
import numpy as np

# ---------------- UI ----------------
st.set_page_config(page_title="GitHub RAG Hybrid")
st.title("💬 Chat with GitHub Repo (Hybrid + Rerank)")

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

repo_url = st.text_input("Enter GitHub Repository URL")

# ---------------- Load Repo ----------------
if st.button("Load Repo"):
    if repo_url:
        with st.spinner("Cloning & processing..."):
            repo_path = "repo"

            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)

            Repo.clone_from(repo_url, repo_path)

            # Load docs
            loader = GitLoader(
                repo_path=repo_path,
                branch="main"
            )
            docs = loader.load()

            # Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # Embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            # Vector DB
            vdb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            vector_retriever = vdb.as_retriever(search_kwargs={"k": 5})

            # ---------------- BM25 ----------------
            corpus = [doc.page_content for doc in chunks]
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)

            def bm25_search(query, k=5):
                scores = bm25.get_scores(query.split())
                top_k_idx = np.argsort(scores)[-k:][::-1]
                return [chunks[i] for i in top_k_idx]

            # ---------------- Hybrid Retriever ----------------
            def hybrid_search(query):
                vector_docs = vector_retriever.invoke(query)
                keyword_docs = bm25_search(query)

                combined = vector_docs + keyword_docs

                # Remove duplicates
                seen = set()
                unique_docs = []
                for doc in combined:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        unique_docs.append(doc)

                return unique_docs

            # ---------------- Reranker ----------------
            def rerank(query, docs, top_k=5):
                query_emb = embeddings.embed_query(query)
                doc_embs = [embeddings.embed_query(d.page_content) for d in docs]

                scores = [
                    np.dot(query_emb, doc_emb) /
                    (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    for doc_emb in doc_embs
                ]

                ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in ranked[:top_k]]

            # ---------------- Format ----------------
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            # ---------------- LCEL Pipeline ----------------
            def retrieve_and_rerank(query):
                docs = hybrid_search(query)
                docs = rerank(query, docs)
                return format_docs(docs)

            retriever_chain = RunnableLambda(retrieve_and_rerank)

            parallel_chain = RunnableParallel({
                "context": retriever_chain,
                "question": RunnablePassthrough()
            })

            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the given GitHub repository context.

If not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            parser = StrOutputParser()

            final_chain = RunnableSequence(prompt, llm, parser)

            rag_pipeline = RunnableSequence(parallel_chain, final_chain)

            st.session_state.rag_pipeline = rag_pipeline

            st.success("Repo indexed successfully!")

# ---------------- Chat ----------------
query = st.text_input("Ask a question about the repo")

if query and st.session_state.rag_pipeline:
    with st.spinner("Thinking..."):
        answer = st.session_state.rag_pipeline.invoke(query)
        st.write("### Answer:")
        st.write(answer)