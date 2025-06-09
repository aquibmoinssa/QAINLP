import pdfplumber
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

@torch.inference_mode()
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if (t := page.extract_text()):
                text += t + "\n"
    return text

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

@torch.inference_mode()
def load_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tokenizer, model

@torch.inference_mode()
def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return F.normalize(embeddings, p=2, dim=1)