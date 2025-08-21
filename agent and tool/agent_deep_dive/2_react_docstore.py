import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

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

react_docstroe_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context"
    )
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstroe_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_error=True,
)

chat_history = []
while True:
    query = input("Kamu: ")
    if query.lower() == "exit":
        break
    respone = agent_executor.invoke(
        {"input": query, "chat_history": chat_history}
    )
    print(f"AI: {respone['output']}")

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=respone["output"]))
