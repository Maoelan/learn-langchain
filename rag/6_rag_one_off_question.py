import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Apa saja dampak dari penampilan Limp Bizkit di Woodstock 1999 terhadap penonton dan suasana festival?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = (
    "Ini adalah beberapa dokument yang setidaknya membantu kamu menjawab pertanyaan: "
    + query
    + "\n\nDokument Relevan:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nTolong berikan jawaban hanya berdasarkan dari dokument. Jika tidak ada jawaban yang ditemukan, respon hanya dengan 'Saya tidak tahu pasti'."
)

model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="Kamu adalah AI asisten yang sangat membantu."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
#print("Full result:")
#print(result)
print("Content only: ")
print(result.content)

