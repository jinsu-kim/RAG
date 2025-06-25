from dotenv import load_dotenv
from langchain_teddynote import logging # https://smith.langchain.com
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate   # prompt formatter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
logging.langsmith("RAG_test")

# Load Document
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")

# Split Document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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

# Embedding
embeddings = OpenAIEmbeddings()

# Vectorstore
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Retriever
retriever = vectorstore.as_retriever()

# Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Define Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run Chain
question = "삼성전자가 자체 개발한 AI 의 이름은?"
response = chain.invoke(question)
print(response)