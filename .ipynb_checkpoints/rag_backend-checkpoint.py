
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------- GEMINI ----------
client = genai.Client(api_key="AIzaSyBOI9LdpF75tZbf95qH98FTF7eJo4udQUg")

# ---------- EMBEDDINGS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- VECTOR DB ----------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def ask_gemini(context, question):
    prompt = f"""
You are a healthcare assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"I don't know. Information not available in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.models.generate_content(
        model="models/gemma-3-1b-it",
        contents=prompt
    )
    return response.text

def rag_pipeline(question):
    docs = retriever.invoke(question)

    if not docs:
        return "I don't know. Information not available in the documents."

    context = "\n\n".join(d.page_content for d in docs)
    return ask_gemini(context, question)
