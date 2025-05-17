import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

def ask_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can use "gpt-3.5-turbo" if you want
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Test the chatbot
user_input = input("You: ")
while user_input.lower() not in ['exit', 'quit']:
    answer = ask_gpt(user_input)
    print("Bot:", answer)
    user_input = input("You: ")

