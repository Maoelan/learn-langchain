from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah seorang komedian yang menceritakan candaan sarkastik tentang {topik}."),
        ("human", "Berikan aku {jumlah_candaan} candaan.")
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper()) #type: ignore
count_words = RunnableLambda(lambda x: f"Total kata: {len(x.split())}\n{x}") #type: ignore

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topik": "pemerintah republik indonesia", "jumlah_candaan": 3})

print(result)