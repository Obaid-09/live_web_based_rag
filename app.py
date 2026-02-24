#import streamlit as st
#import os

#st.write("DEBUG: App started")
#st.write("DEBUG: GROQ_API_KEY exists =", bool(os.getenv("GROQ_API_KEY")))


import os
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Live Web RAG",
    page_icon="🔍",
    layout="wide"
)

#st.title("🔍 Live Web-Based RAG System")
st.caption("Ask questions based on live LangChain documentation")

# ---------------------------
# Secrets
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found - UI Loaded without LLM functionality")
    st.stop()

# ---------------------------
# Load & Process Website
# ---------------------------
@st.cache_resource
def load_vectorstore():
    url = "https://python.langchain.com/docs/introduction/"

    loader = WebBaseLoader(web_paths=[url])
    raw_docs = loader.load()

    soup = BeautifulSoup(raw_docs[0].page_content, "html.parser")
    for tag in soup(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    clean_text = soup.get_text(separator="\n")

    docs = [
        Document(
            page_content=clean_text,
            metadata={"source": url}
        )
    ]

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a documentation assistant.
Answer ONLY using the provided context in 50 to 60 words.
If the answer is not in the context, say "Not found in documentation."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key = os.getenv("GROQ_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs = {"prompt": prompt}
)

# ---------------------------
# HISTORY STORAGE
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# UI
# ---------------------------
# ----------------------------
# Chat Input
# ----------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about LangChain docs")

if user_query:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain(user_query)
            answer = result["result"]
            st.markdown(answer)

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


#query = st.chat_input("Ask a question about LangChain docs")

#if user_query:
 #   with st.spinner("Thinking..."):
  #      result = qa_chain(user_query)

   # st.markdown("### Answer")
    #st.write(result["result"])

    #print("\nSources:")
    #print(result["source_documents"][0].metadata["source"])