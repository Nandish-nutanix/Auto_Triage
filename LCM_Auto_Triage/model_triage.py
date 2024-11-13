"""
Module for generating embeddings, grouping test cases,
and calculating similarity between JIRA issues and test cases.

Copyright (c) 2024 Nutanix Inc. All rights reserved.

Author:nandish.chokshi@nutanix.com
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from framework.lib.nulog import STEP, INFO

class EmbeddingProcessor:
  """
  A processor class for generating embeddings, grouping test cases,
  and calculating similarity between JIRA issues
  and test cases using embeddings.
  """
  def __init__(self, model_name="google/flan-t5-base",
               distance_threshold=0.4,
               similarity_threshold=0.94):
    """
    Initializes the EmbeddingProcessor with the given model, distance
    threshold, and similarity threshold.

    Args:
      model_name (str): The name of the pre-trained model to be used
      for generating embeddings.
      distance_threshold (float): The threshold used for clustering.
      similarity_threshold (float): The threshold for cosine similarity
      between JIRA issues and test cases.
    """
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    self.distance_threshold = distance_threshold
    self.similarity_threshold = similarity_threshold

  def generate_embedding(self, text):
    """
    Generates an embedding for the provided text using the model's encoder.

    Args:
      text (str): The text for which to generate an embedding.

    Returns:
      np.ndarray: The generated embedding as a numpy array.
    """
    inputs = self.tokenizer(
      text, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
      encoder_outputs = self.model.encoder(**inputs)
    embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()\
      .flatten()
    return embedding

  def process_jira_issues(self, jira_file_path, output_dir):
    """
    Processes JIRA issues by reading the input file, generating embeddings,
    and saving the results to an Excel file.

    Args:
      jira_file_path (str): Path to the JIRA issues file (Excel).
      output_dir (str): Directory where the output Excel file with embeddings
      will be saved.

    Returns:
      pd.DataFrame: DataFrame containing the JIRA issues with embeddings.
    """
    df_jira = pd.read_excel(jira_file_path)
    df_jira['Embedding'] = df_jira.apply(
      lambda row: self.generate_embedding(f"{row['Summary']}\
          {row['Description']}"),
      axis=1
    )
    df_jira['Embedding'] = df_jira['Embedding'].apply(
      lambda x: np.array2string(x, separator=',', precision=6,\
          suppress_small=True)
    )

    output_file = f'{output_dir}/create_jira_issues_with_embeddings_t5.xlsx'
    os.makedirs(output_dir, exist_ok=True)
    df_jira.to_excel(output_file, index=False)
    STEP(f"JIRA Issues with embeddings saved to {output_file}")
    return df_jira

  def group_test_cases(self, exception_tracebacks):
    """
    Groups test cases based on their embeddings\
        and computes similarity among them.

    Args:
      exception_tracebacks (list): List of exception traceback strings from
      the test cases.

    Returns:
      tuple: A tuple containing the grouped test cases and similarity matrix.
    """
    embeddings = [self.generate_embedding(text)\
        for text in exception_tracebacks]
    similarity_matrix = cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(
      n_clusters=None, distance_threshold=self.distance_threshold,
      linkage='average'
    )
    clusters = clustering.fit_predict(embeddings)
    grouped_cases = {i: [] for i in np.unique(clusters)}
    for idx, cluster in enumerate(clusters):
      grouped_cases[cluster].append((exception_tracebacks[idx], clusters[idx]))
    return grouped_cases, similarity_matrix

  def process_test_cases(self, test_file_path, output_dir):
    """
    Processes test cases by reading the input file, generating embeddings,
    grouping the test cases, and saving the results to an Excel file.

    Args:
      test_file_path (str): Path to the test cases file (Excel).
      output_dir (str): Directory where the output Excel file with grouped test
      cases and embeddings will be saved.

    Raises:
      ValueError: If the input Excel file does not contain the required columns.

    Returns:
      pd.DataFrame: DataFrame containing the grouped test cases with embeddings.
    """
    df_test = pd.read_excel(test_file_path)
    if not all(
        col in df_test.columns
        for col in ['Test Name', 'Test ID', 'Exception Summary', 'Traceback']
    ):
      raise ValueError(
        "The input Excel file must contain 'Test Name', 'Test ID', "
        "'Exception Summary', and 'Traceback' columns."
      )

    test_names = df_test['Test Name'].tolist()
    exception_summaries = df_test['Exception Summary'].tolist()
    tracebacks = df_test['Traceback'].tolist()
    exception_tracebacks = [
      f"{summary} {traceback}" for summary, traceback in
      zip(exception_summaries, tracebacks)
    ]

    grouped_test_cases, similarity_matrix = self.\
        group_test_cases(exception_tracebacks)

    group_data = []
    group_embeddings = []
    for group_id, cases in grouped_test_cases.items():
      group_indexes = [exception_tracebacks.index(summary)\
          for summary, _ in cases]
      if len(group_indexes) == 1:
        average_similarity_score = 1.00
      else:
        group_similarity_scores = similarity_matrix[
          np.ix_(group_indexes, group_indexes)
        ]
        average_similarity_score = np.mean(
          group_similarity_scores[
            np.triu_indices_from(group_similarity_scores, k=1)
          ]
        )

      combined_text = " ".join([summary for summary, _ in cases])
      group_embedding = self.generate_embedding(combined_text)

      for summary, _ in cases:
        test_name = test_names[exception_tracebacks.index(summary)]
        group_data.append([test_name, summary, group_id,\
            average_similarity_score])
        group_embeddings.append(group_embedding)

    group_df = pd.DataFrame(
      group_data,
      columns=['Test Name', 'Exception Summary & Traceback', 'Group ID',
               'Average Similarity Score']
    )
    group_df['Embedding'] = [
      np.array2string(embedding, separator=',') for\
          embedding in group_embeddings
    ]

    output_file = f'{output_dir}/groupings_reduced_with_similarity_and_embeddings.xlsx'
    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
      df_test.to_excel(writer, sheet_name='Original Data', index=False)
      group_df.to_excel(writer, sheet_name='Grouped Test Cases', index=False)

    STEP(f"Test Case groupings and embeddings saved to {output_file}")
    INFO(
      f"Total unique/distinct test cases in the 'Grouped Test Cases' sheet:"f"{group_df['Test Name'].nunique()}")
    return group_df

  def calculate_jira_jita_similarity(self, df_jira, df_jita, output_dir):
    """
    Calculates the similarity between JIRA issues and test cases based on
    their embeddings, and saves the results to an Excel file.

    Args:
      df_jira (pd.DataFrame): DataFrame containing JIRA issues with embeddings.
      df_jita (pd.DataFrame): DataFrame containing grouped test cases with
      embeddings.
      output_dir (str): Directory where the output Excel file with similarity
      results will be saved.
    """
    df_jira['Embedding'] = df_jira['Embedding'].apply(
      lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str)\
          else np.zeros(768)
    )
    df_jita['Group Embedding'] = df_jita['Embedding'].apply(
      lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str)
      else np.zeros(768)
    )

    similarity_data = []

    for _, jira_row in df_jira.iterrows():
      jira_embedding = jira_row['Embedding']
      for _, group_row in df_jita.iterrows():
        jita_group_embedding = group_row['Group Embedding']
        similarity_score = cosine_similarity(
          [jira_embedding], [jita_group_embedding]
        )[0][0]

        if similarity_score >= self.similarity_threshold:
          similarity_data.append({
            'Jira Ticket ID': jira_row.get('Ticket ID', 'N/A'),
            'Jira Summary': jira_row.get('Summary', 'N/A'),
            'Jita Group ID': group_row.get('Group ID', 'N/A'),
            'Jita Test Case': group_row.get('Test Name', 'N/A'),
            'Jita Group Text': group_row.get(
              'Exception Summary & Traceback', 'N/A'
            ),
            'Similarity Score': similarity_score})

    df_similarity = pd.DataFrame(similarity_data)
    df_similarity_cleaned = df_similarity.drop_duplicates()
    df_similarity_cleaned = df_similarity_cleaned.sort_values(
      by='Similarity Score', ascending=False)

    output_file_cleaned = (
      f'{output_dir}/jira_jita_linked_similarity_with_test_cases_cleaned.xlsx'
    )
    df_similarity_cleaned.to_excel(output_file_cleaned, index=False)

    sheet1 = pd.read_excel(output_file_cleaned, sheet_name='Sheet1')
    grouped_data = sheet1.groupby('Jita Test Case')['Jira Ticket ID'].apply(
      lambda x: '\n'.join(x.unique())
    ).reset_index()

    with pd.ExcelWriter(output_file_cleaned, mode='a', \
        engine='openpyxl') as writer:
      grouped_data.to_excel(writer, sheet_name='Formatted Data', index=False)

    STEP("JIRA and JITA Group Similarity Results with Test Case Names have been saved.")
    STEP("Grouped Data by 'JITA Test Case' and JIRA has been saved to 'Formatted Data' sheet.")

    formatted_test_cases = set(grouped_data['Jita Test Case'].tolist())
    grouped_test_cases = set(df_jita['Test Name'].tolist())

    missing_test_cases = [
      test_case for test_case in grouped_test_cases
      if test_case not in formatted_test_cases
    ]

    if missing_test_cases:
      INFO(f"Number of unmatched test cases found: {len(missing_test_cases)}")
      formatted_missing_test_cases = "\n".join(
        [f"{i+1}. {test_case}" for i, test_case in\
            enumerate(missing_test_cases)]
      )
      INFO(f"List of missing test cases:\n{formatted_missing_test_cases}")
    else:
      INFO("All unique test cases matched with JIRA tickets.")

processor = EmbeddingProcessor()
jira_file_path = 'JIRA_new_issues.xlsx'
test_file_path = 'failed_test_results.xlsx'
output_dir = 'excel_dump'
df_jira = processor.process_jira_issues(jira_file_path, output_dir)
df_jita = processor.process_test_cases(test_file_path, output_dir)
processor.calculate_jira_jita_similarity(df_jira, df_jita, output_dir)
