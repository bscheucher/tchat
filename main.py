import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

chat = ChatOpenAI(api_key=api_key)
memory = ConversationBufferMemory(chat_memory=FileChatMessageHistory("messages.json"), memory_key="messages", return_messages=True)

# Define the prompt
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# Create the LLMChain
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input(">> ")

    if content.lower() == "exit":
        print("Good bye!")
        break

    result = chain.run({"content": content})
    print(result)
