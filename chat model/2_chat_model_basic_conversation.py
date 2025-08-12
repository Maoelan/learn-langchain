from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="Selesaikan permasalahan matematika berikut"),
    HumanMessage(content="Berapa 81 dibagi 9? jawab singkat saja"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

messages = [
    SystemMessage(content="Selesaikan permasalahan matematika berikut"),
    HumanMessage(content="Berapa 81 dibagi 9? jawab singkat saja"),
    AIMessage(content="81 dibagi 9 sama dengan 9."),
    HumanMessage(content="Berapa 10 dikali 5? jawab singkat aja"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

