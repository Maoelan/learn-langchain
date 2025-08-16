import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "Mengapa Resident Evil 4 (2005) dianggap revolusioner dibandingkan dengan game horor lainnya yang menang GOTY?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")