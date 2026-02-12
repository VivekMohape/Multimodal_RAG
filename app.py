import streamlit as st
import tempfile
import time

from pypdf import PdfReader

from rag.embeddings import get_jina_embeddings
from rag.ocr import extract_text_from_image
from rag.vision import describe_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank
from rag.llm import ask_llm


st.title("Enterprise Multimodal RAG")

if "history" not in st.session_state:
    st.session_state.history = []

groq_key = st.sidebar.text_input("Groq API Key", type="password")
jina_key = st.sidebar.text_input("Jina API Key", type="password")

model = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
)

filter_type = st.sidebar.selectbox(
    "Filter Retrieval",
    ["all", "text", "image"]
)

txt_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if txt_file and groq_key and jina_key:

    if txt_file.name.endswith(".pdf"):
        reader = PdfReader(txt_file)
        raw_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    else:
        raw_text = txt_file.read().decode("utf-8")

    chunks = chunk_text(raw_text)

    metadata = [{"type": "text"} for _ in chunks]

    if img_file:
        image_bytes = img_file.read()

        ocr_text = extract_text_from_image(
            tempfile.NamedTemporaryFile(delete=False).name
        )

        vision_text = describe_image(image_bytes, groq_key)

        combined = " ".join([ocr_text, vision_text]).strip()

        if combined:
            chunks.append(combined)
            metadata.append({"type": "image"})

    embeddings = get_jina_embeddings(chunks, jina_key)
    retriever = FAISSRetriever(embeddings, metadata)

    query = st.text_input("Ask a question")

    if query:
        start = time.time()

        query_emb = get_jina_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type
        ids = retriever.search(query_emb, top_k=5, filter_type=f)

        retrieved_docs = [chunks[i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)

        context = "\n\n".join(reranked[:3])

        answer = ask_llm(context, query, groq_key, model)

        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Latency")
        st.write(f"{latency} seconds")

        st.subheader("Chat History")
        for q, a in st.session_state.history[-5:]:
            st.write("Q:", q)
            st.write("A:", a)

        st.subheader("Retrieved Context")
        st.text(context)
