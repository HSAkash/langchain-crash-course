from langchain.prompts import ChatPromptTemplate
from chatModel import get_model


# Create a ChatOpenAI model
model = get_model()

# PART 1: Create a ChatPromptTemplate using a template string
print("-----Prompt from Template-----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

response = model.invoke(prompt)
print(response)

# PART 2: Prompt Syatem and Human Messages
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print(prompt)

response = model.invoke(prompt)
print(response)