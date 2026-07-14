import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#get large GPT2 tokenizer and GPT2 model
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_tokenHere is the exact code extracted from the Jupyter Notebook cell in the image:


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#get large GPT2 tokenizer and GPT2 model
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)

# Loading pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium" # Model size can be switched accordingly (e.g., "gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()