from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


# Create a chat model
model_path =  "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a text generation pipeline 
text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# Create a chat pipeline
llm = HuggingFacePipeline(pipeline=text_generator)



# Define the user input
user_input = "What is the meaning of life?"

# Use the chat model to generate a response
response = llm.invoke(user_input)

# Print the response
print(f"""
User Input: {user_input}
Response: {response}""")