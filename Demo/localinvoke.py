from langchain_community.llms import Ollama
llm = Ollama(model="qwen:7b")

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
respond = chain.invoke({"input": "how can langsmith help with testing?"})
print(type(respond))
print(respond)