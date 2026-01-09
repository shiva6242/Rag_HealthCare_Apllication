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
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    /* Title */
    .title-text {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #6ec1ff;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle-text {
        text-align: center;
        font-size: 17px;
        color: #d1e8ff;
        margin-bottom: 25px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #1e2a38;
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #ffcc00;
        color: #ffeaa7;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    /* Text area */
    textarea {
        background-color: #1e2a38 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #4ea8de !important;
        font-size: 15px !important;
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4ea8de, #6ec1ff);
        color: #000000;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #6ec1ff, #4ea8de);
    }

    /* Answer card */
    .answer-box {
        background-color: #1e2a38;
        padding: 22px;
        border-radius: 14px;
        border-left: 6px solid #6ec1ff;
        color: #ffffff;
        font-size: 16px;
        line-height: 1.8;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #a8c7e0;
        font-size: 13px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title-text">🏥 Healthcare RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">AI-powered answers from healthcare documents, policies & protocols</div>',
    unsafe_allow_html=True
)



# ---------------- INPUT ----------------
question = st.text_area(
    "Enter your healthcare question:",
    placeholder="Example: What are the treatment protocols for diabetes?",
    height=120
)

# ---------------- BUTTON ----------------
ask_btn = st.button("🔍 Get Answer", use_container_width=True)

# ---------------- OUTPUT ----------------
if ask_btn:
    if question.strip() == "":
        st.warning("Please enter a healthcare-related question.")
    else:
        with st.spinner("🔎 Searching healthcare knowledge..."):
            answer = rag_pipeline(question)

        st.markdown("### ✅ Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer-box">
⚠️ <b>Disclaimer:</b> This application is designed strictly for <b>healthcare-related queries</b> 
based on the provided documents and datasets.  
It does <b>not</b> provide medical diagnosis or advice.  
For non-healthcare or unrelated questions, the system may respond with 
"I don't know. Information not available in the documents."
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>🔒 Powered by ChromaDB • HuggingFace • Gemini | RAG-based Healthcare Assistant</div>",
    unsafe_allow_html=True
)
