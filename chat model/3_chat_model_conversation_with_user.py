from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history = []

system_message = SystemMessage(content="Kamu adalah AI asisten yang sangat membantu")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    respone = result.content
    chat_history.append(AIMessage(content=respone))

    print(f"AI: {respone}")

print("---- Riwayat Chat ----")
print(chat_history)