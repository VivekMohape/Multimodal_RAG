import streamlit as st
import tempfile

from rag.embeddings import get_jina_embeddings
from rag.ocr import extract_text_from_image
from rag.retriever import FAISSRetriever
from rag.llm import ask_groq


st.title("Multimodal RAG with Jina v4 + Groq")

st.sidebar.header("API Keys")

groq_key = st.sidebar.text_input("Groq API Key", type="password")
jina_key = st.sidebar.text_input("Jina API Key", type="password")

if not groq_key or not jina_key:
    st.warning("Enter both API keys to continue.")
    st.stop()

txt_file = st.file_uploader("Upload TXT file", type=["txt"])
img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if txt_file and img_file:
    text_data = txt_file.read().decode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_file.read())
        image_path = tmp.name

    ocr_text = extract_text_from_image(image_path)

    chunks = text_data.split("\n\n")
    if ocr_text:
        chunks.append("Image text: " + ocr_text)

    embeddings = get_jina_embeddings(chunks, jina_key)
    retriever = FAISSRetriever(embeddings)

    query = st.text_input("Ask a question")

    if query:
        query_emb = get_jina_embeddings([query], jina_key)
        top_ids = retriever.search(query_emb, top_k=3)

        context = "\n\n".join([chunks[i] for i in top_ids])

        answer = ask_groq(context, query, groq_key)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Context")
        st.text(context)
