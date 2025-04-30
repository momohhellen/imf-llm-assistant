# summarizer.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

simplify_template = """
You are an assistant helping non-experts understand financial policy. 
Simplify this explanation in clear, layperson-friendly language:

"{text}"

Make it concise and easy to understand.
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=simplify_template,
)

def summarize_text(text):
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text)
