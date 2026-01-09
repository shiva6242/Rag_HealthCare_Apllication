from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_documents(): 
    documents = [] 
    csv_files = [ "Disease_precaution.csv", "DiseaseAndSymptoms.csv" ] 
    for file in csv_files: 
        documents.extend(CSVLoader(file).load()) 
    text_files = [ "hospital_policies.txt", "patient_care_guidelines.txt", "treatment_protocols.txt" ] 
    for file in text_files: 
        documents.extend(TextLoader(file).load()) 
    pdf_files = [ "Clinical_Best_Practices.pdf", "Emergency_Medical_Procedures.pdf" ] 
    for file in pdf_files: 
        documents.extend(PyPDFLoader(file).load()) 
    return documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(load_documents())

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Vector DB created successfully")
