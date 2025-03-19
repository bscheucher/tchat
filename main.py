import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# Define the prompt
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[HumanMessagePromptTemplate.from_template("{content}")]
)

# Instantiate ChatOpenAI
chat = ChatOpenAI(api_key=api_key)

# Create the LLMChain
chain = LLMChain(
    llm=chat,
    prompt=prompt
)

while True:
    content = input(">> ")

    if content.lower() == "exit":
        print("Good bye!")
        break

    result = chain.run({"content": content})
    print(result)
