from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART 1
template = "Berikan aku lelucon tentang {topik}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Promt dari Template ----")
prompt = prompt_template.invoke({"topik":"kucing"})
print(prompt)

# PART 2
template_multiple = """Kamu adalah AI asisten yang sangat membantu.
Human: Ceritakan aku cerita {kata_sifat} tentang {hewan}.
Assistant:"""

prompt_multipe = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multipe.invoke({"kata_sifat": "lucu", "hewan": "kucing"})

print("\n---- Prompt dengan Beberapa Placeholder ----\n")
print(prompt)

# PART 3
messages = [
    ("system", "Kamu adalah seorang komedian yang menceritakan candaan tentang {topik}."),
    ("human", "Berikan aku {jumlah_candaan} candaan."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topik": "pengacara", "jumlah_candaan": 3})

print("\n---- Prompt dengan System dan Human Messages (Tuple) ----\n")
print(prompt)

# PART 4
messages = [
    ("system", "Kamu adalah seorang komedian yang menceritakan candaan tentang {topik}."),
    HumanMessage(content="Berikan aku 3 candaan")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topik": "pengacaran"})

print("\n---- Prompt dengan System dan Human Messages (Tuple) 2 ----\n")
print(prompt)

# PART 5 tidak work karena harus berbentuk tuple
messages = [
    ("system", "Kamu adalah seorang komedian yang menceritakan candaan tentang {topik}."),
    HumanMessage(content="Berikan aku {jumlah_candaan} candaan")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topik": "pengacaran", "jumlah_candaan": 3})

print("\n---- Prompt dengan System dan Human Messages (Tuple) 3 Tidak Work ----\n")
print(prompt)