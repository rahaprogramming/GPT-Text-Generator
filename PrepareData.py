import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling

# Step 1: Load Data
import requests
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
data = response.text.split("\n")

# Step 2: Tokenize Data
tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
encoded_data = [tokenizer.encode(text) for text in data]

# Step 3: Define the Model
model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

# Step 4: Train the Model
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.examples = []
        for text in data:
            input_ids = tokenizer.encode(text)
            for i in range(0, len(input_ids)-block_size+1, block_size):
                self.examples.append(input_ids[i:i+block_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])
        
block_size = 128
train_dataset = TextDataset(encoded_data, tokenizer, block_size)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = transformers.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    for i, input_ids in enumerate(train_loader):
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

# Step 5: Generate Text
generated_text = model.generate(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(generated_text[0], skip_special_tokens=True))

# Step 5: Generate Text
generated_text = model.generate(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(generated_text[0], skip_special_tokens=True))

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
