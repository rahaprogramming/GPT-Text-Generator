import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
model_path = 'gpt2_medium_pretrained'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set generation parameters
max_length = 200
temperature = 0.7

# Input prompt
input_text = input("Enter some text: ")

# Encode prompt using tokenizer
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
output = model.generate(input_ids=input_ids, 
                         max_length=max_length, 
                         temperature=temperature,
                         do_sample=True)

# Decode output text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print generated text
print(output_text)
