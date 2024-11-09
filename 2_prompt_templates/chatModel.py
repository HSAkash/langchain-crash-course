from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import yaml

def get_model():
    with open("config.yaml", "r") as file:
        model_config = yaml.safe_load(file)
        # Create a chat model
        model_path = model_config['chatBot_model']['model_name']
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create a text generation pipeline 
        text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

        # Create a chat pipeline
        llm = HuggingFacePipeline(pipeline=text_generator)
        return llm

if __name__ == '__main__':
    llm = get_model()
    # Define the user input
    user_input = "What is the meaning of life?"
    # Use the chat model to generate a response
    response = llm.invoke(user_input)
    # Print the response
    print(f"""
    User Input: {user_input}
    Response: {response}""")