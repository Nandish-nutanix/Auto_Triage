# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# jira_file_path = 'excel_dump/create_jira_issues_with_embeddings_t5.xlsx'
# jita_file_path = 'excel_dump/groupings_reduced_with_similarity_and_embeddings.xlsx'

# df_jira = pd.read_excel(jira_file_path)
# df_jita = pd.read_excel(jita_file_path, sheet_name='Groupings with Embeddings')

# df_jira['Embedding'] = df_jira['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else np.zeros(768))
# df_jita['Group Embedding'] = df_jita['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else np.zeros(768))

# similarity_data = []
# similarity_threshold = 0.94

# for jira_index, jira_row in df_jira.iterrows():
#     jira_embedding = jira_row['Embedding']
    
#     for group_index, group_row in df_jita.iterrows():
#         jita_group_embedding = group_row['Group Embedding']
#         similarity_score = cosine_similarity([jira_embedding], [jita_group_embedding])[0][0]
        
#         if similarity_score >= similarity_threshold:
#             similarity_data.append({
#                 'Jira Ticket ID': jira_row.get('Ticket ID', 'N/A'),
#                 'Jira Summary': jira_row.get('Summary', 'N/A'),
#                 'Jita Group ID': group_row.get('Group ID', 'N/A'),
#                 'Jita Test Case': group_row.get('Test Name', 'N/A'),
#                 'Jita Group Text': group_row.get('Exception Summary & Traceback', 'N/A'),
#                 'Similarity Score': similarity_score
#             })

# df_similarity = pd.DataFrame(similarity_data)
# df_similarity_cleaned = df_similarity.drop_duplicates()
# df_similarity_cleaned = df_similarity_cleaned.sort_values(by='Similarity Score', ascending=False)

# output_dir = 'excel_dump'
# os.makedirs(output_dir, exist_ok=True)
# output_file_cleaned = f'{output_dir}/jira_jita_linked_similarity_with_test_cases_cleaned.xlsx'
# df_similarity_cleaned.to_excel(output_file_cleaned, index=False)

# print("JIRA and JITA Group Similarity Results with Test Case Names (duplicates removed):")
# print(df_similarity_cleaned)



import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

jira_file_path = 'excel_dump/create_jira_issues_with_embeddings_t5.xlsx'
jita_file_path = 'excel_dump/groupings_reduced_with_similarity_and_embeddings.xlsx'
output_file_cleaned = 'excel_dump/jira_jita_linked_similarity_with_test_cases_cleaned.xlsx'

df_jira = pd.read_excel(jira_file_path)
df_jita = pd.read_excel(jita_file_path, sheet_name='Grouped Test Cases')

df_jira['Embedding'] = df_jira['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else np.zeros(768))
df_jita['Group Embedding'] = df_jita['Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else np.zeros(768))

similarity_data = []
similarity_threshold = 0.94

for jira_index, jira_row in df_jira.iterrows():
    jira_embedding = jira_row['Embedding']
    
    for group_index, group_row in df_jita.iterrows():
        jita_group_embedding = group_row['Group Embedding']
        similarity_score = cosine_similarity([jira_embedding], [jita_group_embedding])[0][0]
        
        if similarity_score >= similarity_threshold:
            similarity_data.append({
                'Jira Ticket ID': jira_row.get('Ticket ID', 'N/A'),
                'Jira Summary': jira_row.get('Summary', 'N/A'),
                'Jita Group ID': group_row.get('Group ID', 'N/A'),
                'Jita Test Case': group_row.get('Test Name', 'N/A'),
                'Jita Group Text': group_row.get('Exception Summary & Traceback', 'N/A'),
                'Similarity Score': similarity_score
            })

df_similarity = pd.DataFrame(similarity_data)
df_similarity_cleaned = df_similarity.drop_duplicates()
df_similarity_cleaned = df_similarity_cleaned.sort_values(by='Similarity Score', ascending=False)

output_dir = 'excel_dump'
os.makedirs(output_dir, exist_ok=True)

df_similarity_cleaned.to_excel(output_file_cleaned, index=False)

sheet1 = pd.read_excel(output_file_cleaned, sheet_name='Sheet1')

grouped_data = sheet1.groupby('Jita Test Case')['Jira Ticket ID'].apply(lambda x: '\n'.join(x.unique())).reset_index()

with pd.ExcelWriter(output_file_cleaned, mode='a', engine='openpyxl') as writer:
    grouped_data.to_excel(writer, sheet_name='Formatted Data', index=False)

print("JIRA and JITA Group Similarity Results (duplicates removed) with Test Case Names have been saved.")
print("Grouped Data by 'Jita Test Case' has been saved to 'Formatted Data' sheet.")
