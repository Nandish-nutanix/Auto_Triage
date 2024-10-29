import requests
import pandas as pd
import os
from datetime import datetime

JIRA_BASE_URL = "https://jira.nutanix.com"
JIRA_API_TOKEN = "MjkyNjcwNTI3MTkyOi+BIralrh16SXTYpuE7t/Wk2+OH"
JIRA_LABELS = [
    "lcm-master-branch-qual", "lcm-fw-qa", "lcm-fw-smoke", "lcm-fw-master-br-issues", 
    "lcm-fw-cluster-config", "lcm-fw-regression", "lcm-fw-released-feat-regression", 
    "lcm-fw-integration", "lcm-fw-ui", "lcm-fw-system-test", "lcm-fw-new-feat", 
    "lcm-v4-27-merge", "lcm-v4-api", "lcm-v4-api-b1", "lcm-v4-api-b1-automation", 
    "lcm-v4-api-ga", "lcm-v4-api-ga-automation", "lcm-v4-api-issue", "lcm-v4-api-pentest"
]
current_directory = os.getcwd()
output_excel_file = os.path.join(current_directory, "jira_issues.xlsx")
today = datetime.today()
date_filter_start = today.replace(month=8, day=15).strftime('%Y-%m-%d')  # August 15th
date_filter_end = today.replace(month=10, day=31).strftime('%Y-%m-%d')  # October 31st
def get_jira_issues_by_labels(labels):
    print(f"Fetching JIRA issues with labels: {labels}")
    jql_query = '(' + ' OR '.join([f'labels = "{label}"' for label in labels]) + f') AND created >= "{date_filter_start}" AND created <= "{date_filter_end}"'
    print(f"JQL Query: {jql_query}")
    url = f"{JIRA_BASE_URL}/rest/api/2/search"
    params = {
        'jql': jql_query,  
        'fields': 'key,summary,description,created,versions',  
        'maxResults': 2000
    }
    headers = {
        "Authorization": f"Bearer {JIRA_API_TOKEN}",
        "Accept": "application/json"
    }
    print("Sending request to JIRA API...")
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        issues = response.json()['issues']
        print(f"Successfully fetched {len(issues)} issues from JIRA.")
        return issues
    else:
        print(f"Failed to fetch issues. Status Code: {response.status_code}")
        print(response.text)
        return []
def save_issues_to_excel(issues, output_file):
    if not issues:
        print("No issues to save.")
        return
    print(f"Saving {len(issues)} issues to Excel file: {output_file}")
    data = []
    for issue in issues:
        ticket_id = issue['key']
        if not ticket_id.startswith('ENG'):
            continue
        summary = issue['fields']['summary']
        description = issue['fields']['description']
        created_date = issue['fields']['created']
        affects_versions = ', '.join([version['name'] for version in issue['fields'].get('versions', [])
                                      if version['name'] in ['master', 'LCM-3.1']])
        if not affects_versions:
            continue
        data.append([ticket_id, summary, description, created_date, affects_versions])
    if not data:
        print("No issues with 'ENG' tickets and relevant 'Affects Versions' (master or LCM-3.1) were found.")
        return
    df = pd.DataFrame(data, columns=['Ticket ID', 'Summary', 'Description', 'Date Created', 'Affects Versions'])
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")
jira_issues = get_jira_issues_by_labels(JIRA_LABELS)
if jira_issues:
    save_issues_to_excel(jira_issues, output_excel_file)
else:
    print("No issues were fetched, so no file was created.")
