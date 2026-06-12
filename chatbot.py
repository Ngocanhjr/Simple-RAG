import os

from dotenv import load_dotenv

load_dotenv() #

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

#if embedding is local
#from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

if not os.getenv("NVIDIA_API_KEY"):
    raise RuntimeError("Thiếu NVIDIA_API_KEY trong file .env")

from langchain_core.prompts import ChatPromptTemplate
loader = DirectoryLoader(
    path="./dataset",
    glob="**/*.pdf", # **=.dataset
    loader_cls=PyPDFLoader,
    silent_errors=False,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load() #Store the document to memory

# print(docs)
# print(len(docs))

#Chunk order
MARKDOWN_SEPERATOR = [
    "\n#{1,6}", # Markdown headers
    "```\n", # Code blocks
    "\n\\*\\*\\**\n", # Horizontal rules (---, ***, ___)
    "\n---+\n", # Horizontal rules (---, ***, ___)
    "\n___+\n", # Horizontal rules (---, ***, ___)
    "\n\n", # Paragraphs
    "\n", # Newlines
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True, #store the index of the original document in the metadata of each chunk
    strip_whitespace=True, #remove whitespace at the beginning and end of each chunk
    separators=MARKDOWN_SEPERATOR,
)

splits = text_splitter.split_documents(docs)

# from pprint import pprint
# pprint(splits)

#Run local
# embedding_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={
#         "device": "cpu",
#     },
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "batch_size": 2,
#     },
# )

#Run API key
embedding_model = NVIDIAEmbeddings(
    model="baai/bge-m3",
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 2,
    },
)

FAISS_PATH = "./faiss_index"

index_file = os.path.join(FAISS_PATH, "index.faiss")
metadata_file = os.path.join(FAISS_PATH, "index.pkl")

if os.path.exists(index_file) and os.path.exists(metadata_file):
    print("[FAISS] Đang tải index đã lưu...")

    vectorstore = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE,
    )
else:
    print("[FAISS] Chưa có index, đang embedding tài liệu...")

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    vectorstore.save_local(FAISS_PATH)
    print("[FAISS] Đã lưu index vào thư mục faiss_index")
    
"""
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embedding_model,
    # index_path="./faiss_index",
    # index_factory_str="Flat",
    distance_strategy=DistanceStrategy.COSINE, # Use cosine similarity for better performance with normalized embeddings
)
"""
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2},
)

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n" 
    "RULES:\n" 
    "1) Use ONLY the provided context to answer.\n" 
    "2) If the answer is not clearly contained in the context, say: " "\"I don't know based on the provided documents.\"\n" 
    "3) Do NOT use outside knowledge, guessing, or web information.\n" 
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n" 
    "Context:\n{context}\n\n" 
    "Question: {question}"
)

llm = ChatNVIDIA(
    model="qwen/qwen3-next-80b-a3b-instruct",
    temperature=0.1,
    top_p=0.7, #Limit scope of token most likely ones
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_template(template)

#Represent RAG pipe
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm 
    | StrOutputParser()
)

question = input("Question: ")

answer = rag_chain.invoke(question)

print("Answer:", answer)

with open("answer.txt", "w", encoding="utf-8") as file:
    file.write(answer)

print("Đã lưu câu trả lời vào answer.txt")