from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import time
load_dotenv()

loader = PyPDFLoader("ML(doc).pdf")
docs = loader.load()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = text_splitter.split_documents(docs)
embedding_model = HuggingFaceEmbeddings()

vectorstores = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "Chroma_db"
    
)
query = input("Ask Query: ")
retriver =vectorstores.as_retriever(
    search_type = "similarity",
    search_kwargs= {"k": 5}
)
result = retriver.invoke(query)
if not result:
    print("No relevant context found.")
    exit()

    
model = ChatGroq(model="openai/gpt-oss-120b")
context = "\n\n".join([r.page_content for r in result])
prompt = f"""
You are an AI assistant.

Answer the question only using the context below.

Context:
{context}

Question:
{query}
"""

start = time.time()
response = model.invoke([HumanMessage(prompt)])

print("\nAnswer: \n")
print(response.text)
print("\nSources:")
for doc in result:
    print(doc.metadata)
end = time.time()
print(f"\nLatency: {end - start:.2f} seconds")

try:
    print("\nToken Usage:")
    print(response.usage_metadata)
except:
    print("Token usage not available.")