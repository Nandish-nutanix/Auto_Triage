import pandas as pd
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_embedding(summary, description):
    combined_text = f"{summary} {description}"
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    return embedding

file_path = 'JIRA_new_issues.xlsx'
df_jira = pd.read_excel(file_path)

df_jira['Embedding'] = df_jira.apply(lambda row: generate_embedding(row['Summary'], row['Description']), axis=1)

output_dir = 'excel_dump'
output_file = f'{output_dir}/create_jira_issues_with_embeddings_t5.xlsx'

df_jira['Embedding'] = df_jira['Embedding'].apply(lambda x: np.array2string(x, separator=',', precision=6, suppress_small=True))

os.makedirs(output_dir, exist_ok=True)

df_jira.to_excel(output_file, index=False)

print("The new Excel file with similarity scores has been created:", output_file)