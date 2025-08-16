import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_rec_char")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

query = "Game mana yang memperkenalkan mekanik kamera over-the-shoulder yang kini standar?"

print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_rec_char", query, embeddings, "similarity", {"k": 1})

print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store("chroma_db_rec_char", query, embeddings, "mmr", {"k": 1, "fetch_k": 20, "lambda_mult": 0.5})

print("\n--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_rec_char", query, embeddings, "similarity_score_threshold", {"k": 1, "score_threshold": 0.1})

print("Querying demonstrations with different search types completed.")

