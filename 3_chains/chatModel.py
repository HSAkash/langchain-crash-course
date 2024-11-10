from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import yaml
import torch

def get_model():
    with open("config.yaml", "r") as file:
        model_config = yaml.safe_load(file)
        
        # Load model in FP16 if GPU is available to save memory
        model_path = model_config['chatBot_model']['model_name']
        device = 0 if torch.cuda.is_available() else -1
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16 if device == 0 else torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set up text generation pipeline with reduced tokens
        text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, device=device)
        
        # HuggingFacePipeline for chat
        llm = HuggingFacePipeline(pipeline=text_generator)
        return llm

if __name__ == '__main__':
    llm = get_model()
    user_input = "What is the meaning of life?"
    response = llm.invoke(user_input)
    print(f"""
    User Input: {user_input}
    Response: {response}""")
