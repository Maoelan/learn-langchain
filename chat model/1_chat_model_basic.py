from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke("Berapa 81 dibagi 9? jawab singkat saja")
print("Full result: ")
print(result)
print("Content only: ")
print(result.content)