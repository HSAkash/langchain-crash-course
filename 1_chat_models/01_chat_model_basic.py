from chatModel import get_model


model = get_model()
# Define the user input
user_input = "What is the meaning of life?"

# Use the chat model to generate a response
response = model.invoke(user_input)

# Print the response
print(f"""
User Input: {user_input}
Response: {response}""")