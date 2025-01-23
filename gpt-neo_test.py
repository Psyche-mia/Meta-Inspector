import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载GPT-Neo 2.7B模型和tokenizer
neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
neo_model.eval()

# Function to get tokenized text using GPT-Neo's tokenizer
def get_tokenized_text_neo(text):
    tokenized_output = neo_tokenizer(text, return_tensors="pt")
    return tokenized_output

# Function to generate text using GPT-Neo with a simple text input
def generate_text_neo(text):
    inputs = neo_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = neo_model.generate(**inputs, max_length=100)
    generated_text = neo_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Function to generate text using GPT-Neo with tokenized text input
def generate_text_neo_with_tokenized_input(tokenized_input):
    with torch.no_grad():
        output = neo_model.generate(input_ids=tokenized_input['input_ids'].to(device), max_length=100)
    generated_text = neo_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example text to test
simple_text = "This is a test sentence to check the response from GPT-Neo."
tokenized_text = get_tokenized_text_neo(simple_text)

# Generate text with GPT-Neo using both simple text and tokenized input
generated_text_from_simple = generate_text_neo(simple_text)
generated_text_from_tokenized = generate_text_neo_with_tokenized_input(tokenized_text)

# Print the results
print("Generated Text from Simple Input:", generated_text_from_simple)
print("Generated Text from Tokenized Input:", generated_text_from_tokenized)
