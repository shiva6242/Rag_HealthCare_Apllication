import streamlit as st
from rag_backend import rag_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        padding-top: 1.5rem;
    }
    .title-text {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1f4e79;
    }
    .subtitle-text {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    .answer-box {
        background-color: #f1f6fb;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        color: #000000;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown('<div class="title-text">🏥 Healthcare RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Ask questions based on healthcare documents, policies, and protocols</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------
question = st.text_area(
    "Enter your healthcare question:",
    placeholder="Example: What are the treatment protocols for diabetes?",
    height=100
)

# ---------------- BUTTON ----------------
ask_btn = st.button("🔍 Get Answer", use_container_width=True)

# ---------------- OUTPUT ----------------
if ask_btn:
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching medical knowledge..."):
            answer = rag_pipeline(question)

        st.markdown("### ✅ Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🔒 Powered by ChromaDB, HuggingFace Embeddings & Gemini | RAG-based Healthcare Assistant")
