from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah seorang komedian yang menceritakan candaan sarkastik tentang {topik}."),
        ("human", "Berikan aku {jumlah_candaan} candaan.")
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages())) #type: ignore
parse_output = RunnableLambda(lambda x: x.content) #type: ignore

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topik": "pemerintahan republik indonesia", "jumlah_candaan": 3})

print(response)
