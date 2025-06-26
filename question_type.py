from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Determine whether the following question is:
- "fact" if it asks for a simple factual answer, or
- "reasoning" if it requires summarization, inference, or interpretation.

Respond with only one word: either "fact" or "reasoning".

Question:
{question}

Answer:"""
)

def question_type_llm(**kwargs):

    if 'llm_model' in kwargs:
        llm = kwargs['llm_model']
        question = kwargs['question']

        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke(question)

    return response
