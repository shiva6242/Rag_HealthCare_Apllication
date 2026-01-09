# 🏥 Healthcare RAG Assistant

A **Retrieval-Augmented Generation (RAG)** based healthcare assistant that answers medical and policy-related questions using **real healthcare documents**, ensuring responses are **fact-based and document-grounded**.

---

## 🚀 Live Demo
🔗 https://raghealthcareapllication.streamlit.app/

---

## 🧠 What is this project about?

Healthcare data is usually spread across:
- Policies
- Guidelines
- Medical documents
- CSV datasets

This project solves that problem by:
- Retrieving **relevant information** from healthcare documents
- Generating answers using **Gemini LLM**
- Ensuring answers are based **only on retrieved data**, not hallucinations

---

## 🏗 RAG Architecture (How it works)

1. **Document Loading**
   - CSV files (diseases, symptoms, precautions)
   - Text files (policies, guidelines, protocols)
   - PDF files (clinical practices, emergency procedures)

2. **Text Splitting**
   - Documents are split into small overlapping chunks
   - Helps preserve context during retrieval

3. **Embedding Generation**
   - Each chunk is converted into a vector using HuggingFace embeddings
   - Semantic meaning is captured numerically

4. **Vector Storage**
   - All embeddings are stored in **ChromaDB**
   - Enables fast semantic search

5. **Retrieval**
   - User question is converted to an embedding
   - Most relevant chunks are retrieved from ChromaDB

6. **Answer Generation**
   - Retrieved context is sent to **Gemini**
   - Gemini answers **only using the provided context**
   - If information is missing → model clearly says *“I don’t know”*

---

## 📂 Data Sources Used

- **CSV Files**
  - Disease symptoms
  - Disease precautions

- **Text Files**
  - Hospital policies
  - Patient care guidelines
  - Treatment protocols

- **PDF Files**
  - Clinical best practices
  - Emergency medical procedures

---

## 🧪 Technologies Used

| Component | Technology |
|--------|-----------|
LLM | Gemini API |
Embeddings | HuggingFace (sentence-transformers) |
Vector Database | ChromaDB |
Framework | LangChain |
Frontend | Streamlit |
Language | Python |

---

## 📌 Project Structure

```text
healthcare-rag-assistant/
├── app.py
├── rag_backend.py
├── requirements.txt
├── README.md
├── chroma_db/
├── Healthcare_RAG_Datasets/
│   ├── *.csv
│   ├── *.txt
│   ├── *.pdf
