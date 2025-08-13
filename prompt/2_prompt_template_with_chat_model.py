from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# PART 1
print("---- Promt dari Template ----")

template = "Berikan aku lelucon singkat tentang {topik}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topik":"kucing"})
print(prompt)
result = model.invoke(prompt)
print(result.content)

# PART 2
print("\n---- Prompt dengan Beberapa Placeholder ----\n")

template_multiple = """Kamu adalah AI asisten yang sangat membantu.
Human: Ceritakan aku cerita {kata_sifat} singkat tentang {hewan}.
Assistant:"""
prompt_multipe = ChatPromptTemplate.from_template(template_multiple)

prompt = prompt_multipe.invoke({"kata_sifat": "lucu", "hewan": "kucing"})
print(prompt)
result = model.invoke(prompt)
print(result.content)

# PART 3
print("\n---- Prompt dengan System dan Human Messages (Tuple) ----\n")

messages = [
    ("system", "Kamu adalah seorang komedian yang menceritakan candaan sarkastik tentang {topik}."),
    ("human", "Berikan aku {jumlah_candaan} candaan."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topik": "pemerintah republik indonesia", "jumlah_candaan": 3})
print(prompt)
result = model.invoke(prompt)
print(result.content)

# PART 4
print("\n---- Prompt dengan System dan Human Messages (Tuple) 2 ----\n")

messages = [
    ("system", "Kamu adalah seorang komedian yang menceritakan candaan sarkastik tentang {topik}."),
    HumanMessage(content="Berikan aku 3 candaan")
]
prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topik": "pemerintah republik indonesia"})
print(prompt)
result = model.invoke(prompt)
print(result.content)



