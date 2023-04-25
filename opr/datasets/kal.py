import torch
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(["Hello, my dog is cute", "However ur mother is slute"], return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs)
    print(logits)