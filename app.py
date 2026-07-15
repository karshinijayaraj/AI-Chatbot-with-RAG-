import streamlit as st
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

st.set_page_config(page_title="Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot with RAG")

# Load LLM (cached so it only loads once, not on every question)
@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3.2")

# Load Embedding Model (cached)
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

llm = load_llm()
embeddings = load_embeddings()

# Create DB only once (cached so it isn't reopened on every question)
@st.cache_resource
def load_db():

    if not os.path.exists("chroma_db"):

        with st.spinner("Creating vector database... Please wait."):

            loader = PyPDFDirectoryLoader("docs")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

    else:

        db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

    return db

db = load_db()

retriever = db.as_retriever(search_kwargs={"k": 3})

# Mode Selection
mode = st.selectbox(
    "Select Chat Mode",
    [
        "RAG (PDF Only)",
        "General AI",
        "Hybrid"
    ]
)

question = st.text_input("Ask your question")

# Keywords that should be answered directly using system time,
# instead of relying on the LLM (which has no real-time awareness)
time_keywords = ["time", "date", "today", "day is it", "current time", "what time", "what date"]

if question:

    # Direct time/date handling - bypasses the LLM entirely so it
    # always works, regardless of mode or what the local model knows
    if any(keyword in question.lower() for keyword in time_keywords):

        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        st.subheader("Answer")
        st.write(f"The current date and time is: {current_datetime}")

    else:

        with st.spinner("Thinking..."):

            # GENERAL AI
            if mode == "General AI":

                response = llm.invoke(question)

                st.subheader("Answer")
                st.write(response)

            # RAG ONLY
            elif mode == "RAG (PDF Only)":

                # Use similarity scores instead of trusting the LLM to
                # decide relevance on its own (small local models often
                # ignore "only answer from context" instructions)
                docs_with_scores = db.similarity_search_with_score(question, k=3)

                # Lower score = more similar (Chroma uses distance, not similarity)
                RELEVANCE_THRESHOLD = 1.0

                relevant_docs = [
                    doc for doc, score in docs_with_scores
                    if score < RELEVANCE_THRESHOLD
                ]

                if not relevant_docs:

                    st.subheader("Answer")
                    st.write("Information not found in PDF.")

                else:

                    context = "\n\n".join(
                        [doc.page_content for doc in relevant_docs]
                    )

                    prompt = f"""
                    Answer the question using ONLY the context below.
                    Do not use any outside knowledge.
                    Do not explain, just answer directly based on the context.

                    Context:
                    {context}

                    Question:
                    {question}
                    """

                    response = llm.invoke(prompt)

                    st.subheader("Answer")
                    st.write(response)

            # HYBRID
            elif mode == "Hybrid":

                docs = retriever.invoke(question)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                prompt = f"""
                Use the PDF context if relevant.

                If the answer is not present in the PDF,
                answer using your own knowledge.

                Context:
                {context}

                Question:
                {question}
                """

                response = llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response)