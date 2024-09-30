# OBS: Pra rodar o arquivo .py é necessário rodar no terminal "python teste/testeIA.py" pois suas dependencias
# se encontram no ambiente virtual isolado (a pasta .venv do projeto)

# Importação e instalação do modelo pre-treinado
from transformers import RobertaTokenizer, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=10)


# Baixa os datasets das linguagens de programação (Python, Java, Javascript, Go, Ruby e PHP)
from datasets import load_dataset
dataset = load_dataset("code_search_net", trust_remote_code=True)


# Realiza a classificação de texto
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
model_name = 'philomath-1209/programming-language-identification'
loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_name)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text = """
print('Hello')
"""
inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
  logits = loaded_model(**inputs).logits
predicted_class_id = logits.argmax().item()
var = loaded_model.config.id2label[predicted_class_id]