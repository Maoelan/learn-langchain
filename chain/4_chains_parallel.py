from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah ahli dalam review produk."),
        ("human", "Daftar fitur utama dari produk {nama_produk}."),
    ] 
)

def analisis_keuntungan(fitur):
    untung_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Kamu adalah ahli dalam review produk."),
            ("human", "Dari fitur berikut: {fitur}, berikan daftar keuntungannya"),
        ]
    )
    return untung_template.format_prompt(fitur=fitur)

def analisis_kerugian(fitur):
    rugi_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Kamu adalah ahli dalam review produk."),
            ("human", "Dari fitur berikut: {fitur}, berikan daftar kerugiannya"),
        ]
    )
    return rugi_template.format_prompt(fitur=fitur)

def kombinasi_untung_rugi(untung, rugi):
    return f"Keuntungan:\n{untung}\n\nKerugian:\n{rugi}"

untung_branch_chain = (
    RunnableLambda(lambda x: analisis_keuntungan(x)) | model | StrOutputParser()
)

rugi_branch_chain = (
    RunnableLambda(lambda x: analisis_kerugian(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"untung": untung_branch_chain, "rugi": rugi_branch_chain})
    | RunnableLambda(lambda x: kombinasi_untung_rugi(x["branches"]["untung"], x["branches"]["rugi"])) #type: ignore
)

result = chain.invoke({"nama_produk": "Reamle C25S"})

print(result)