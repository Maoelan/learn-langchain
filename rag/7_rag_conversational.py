import os
from dotenv import load_dotenv
import warnings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

llm = ChatOpenAI(model="gpt-4o-mini")

contextualize_q_system_prompt = (
    "Kalau ada history chat dan pertanyaan terbaru dari user "
    "dimana ternyataan tersebut nyambung ke chat sebelumnya, "
    "ubah pertanyaan itu menjadi versi yang dapat kamu pahami sendiri tanpa melihat history "
    "Jangan dijawab pertanyaannya, cukup ubah aja kalau perlu. Kalau sudah jelas, biarin aja."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "Kamu adalah AI asisten yang bertugas untuk tanya-jawab. Silakan gunakan "
    "potongan konteks yang diberikan berikut ini untuk menjawab pertanyaannya. "
    "Kalau tidak tahu jawabannya, bilang aja kalau kamu tidak tahu. "
    "Jawaban maksimal yang harus diberikan terdiri dari tiga kalimat dan dibuat singkat. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Mulai chat dengan AI! Ketik 'exit' untuk menyelesaikan percakapan.")
    chat_history = []
    while True:
        query = input("Kamu: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        print(f"AI: {result['answer']}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()