from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

positif_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI asisten yang membantu kami untuk membalas feedback dalam bentuk kalimat kepelanggan"),
        (
            "human", 
            "Balas dengan ucapan terima kasih untuk feedback positif ini: {feedback}."
        ),
    ]
)

negatif_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI asisten yang membantu kami untuk membalas feedback dalam bentuk kalimat kepelanggan"),
        (
            "human", 
            "Balas dengan tanggapan untuk menanggapi feedback negatif ini: {feedback}."
        ),
    ]
)

netral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI asisten yang membantu kami untuk membalas feedback dalam bentuk kalimat kepelanggan"),
        (
            "human", 
            "Balas dengan permintaan untuk informasi lebih lanjut mengenai feedback netral ini: {feedback}."
        ),
    ]
)

eskalasi_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI asisten yang membantu kami untuk membalas feedback dalam bentuk kalimat kepelanggan"),
        (
            "human", 
            "Balas dengan pesan untuk meneruskan feedback ini ke agen manusia: {feedback}."
        ),
    ]
)

klasifikasi_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI asisten yang sangat membantu."),
        (
            "human",
            "Klasifikasikan sentimen dari feedback ini sebagai positif, negatif, netral, atau eskalasi: {feedback}."
        ),
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positif" in x, #type: ignore
        positif_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negatif" in x, #type: ignore
        negatif_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "netral" in x, #type: ignore
        netral_feedback_template | model | StrOutputParser()
    ),
    eskalasi_feedback_template | model | StrOutputParser()
)

klasifikasi_chain = klasifikasi_template | model | StrOutputParser()

chain = klasifikasi_chain | branches

# Jalankan chain dengan contoh review
# Good review - "Produknya luar biasa. Saya sangat suka menggunakan produk ini dan produk ini sangat berguna"
# Bad review - "Produknya jelek banget. Baru dipakai sekali sudah rusak dan kualitasnya jelek banget"
# Neutral review - "Produknya oke aja sih. Ini berfungsi sesuai dengan yang saya harapkan tanpa ada pengecualian"
# Default - "Saya tidak tau pasti tentang produk ini. Bisakah kamu menjelaskan lebih detail tentang fitur dan keuntungan dari produk ini"

'''review = [
    "",
    "",
    "",
    "",
]'''

review = "Saya tidak tau pasti tentang produk ini. Bisakah kamu menjelaskan lebih detail tentang fitur dan keuntungan dari produk ini"
result = chain.invoke({"feedback": review})

print(result)

'''

results = []
for r in review:
    result = chain.invoke({"feedback": r})
    results.append(result)

for res in results:
    print(res)'''
