
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------- GEMINI ----------
client = genai.Client(api_key="AIzaSyAwjwLjLsZDFvg1Oze20vyIU__jHNTnv-0")

# ---------- EMBEDDINGS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- VECTOR DB ----------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})

def ask_gemini(context, question):
    prompt = f"""
You are a professional healthcare AI assistant.

Answer the question ONLY using the information from the context below.
Do NOT add medical facts outside the context.

Formatting rules (VERY IMPORTANT):
- Give ONLY 2 to 3 key points.
- Each point must be written as a short paragraph (1–2 sentences).
- Each point MUST be on a separate line with a blank line between points.
- number the points.
- Do NOT use labels like "Symptom_1" or "Precaution_1".
- For symptoms or precautions, explain each in its own paragraph.

If the answer is not found in the context, reply exactly:
"I don't know. Information not available in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text

    return text

def rag_pipeline(question):
    docs = retriever.invoke(question)

    if not docs:
        return "I don't know. Information not available in the documents."

    context = "\n\n".join(d.page_content for d in docs)
    return ask_gemini(context, question)
