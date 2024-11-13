# Auto Triage Tool

## Overview

This tool automates the process of:
- Fetching and analyzing JITA failure data
- Clustering similar error messages
- Comparing against existing JIRA tickets
- Creating new JIRA tickets for unique issues
- Using ML-based similarity matching for accurate issue identification

## Installation

1. Clone the repository in your nutest repo
2. Run the installation script:
```bash
bash install_requirements.sh
```

## Configuration

1. Open `jita_urls.json` and add:
   - JITA failure URLs
   - Your JITA account credentials
   ```json
   {
     "username": "your_username",
     "password": "your_password",
     "urls": [
       "your_jita_failure_urls"
     ]
   }
   ```

  - Example of JITA urls: 
  ```json
  {
     "urls": [
        "https://jita-web-server-1.eng.nutanix.com/results?task_ids=6732f3448e79ced0e93a67dc&active_tab=2&merge_tests=false",
        "https://jita-web-server-1.eng.nutanix.com/results?task_ids=6732bcfa8e79cec01c35e326&active_tab=2&merge_tests=false",
        "https://jita-web-server-1.eng.nutanix.com/results?task_ids=672f198a2bc0c41b29373495&active_tab=2&merge_tests=false"
     ]
  }
```
  - Note: Update `generic_failures.json` with the exception summary of any specific failures you want to remove from your execution run
  - Set it as False if not required
    
## Usage

### 1. JITA Data Collection
```bash
python jita_fetching.py jita_urls.json
```
This script:
- Fetches failure data from provided JITA URLs
- Groups similar error messages
- Generates an Excel file with clustered errors

### 2. JIRA Ticket Analysis
```bash
python jira_nutest.py
```
This script:
- Captures JIRA tickets based on specified parameters like date range, priority, labels, versions
- Parameters can be customized in the script based on requirements

### 3. Similarity Analysis
```bash
python model_triage.py
```
Features:
- Uses `google/flan-t5-base` model for generating embeddings
- Compares JIRA and JITA data using cosine similarity
- Configurable similarity threshold for ticket matching
- Automated creation of new JIRA tickets for unique issues

## Technical Details

- **ML Model**: google/flan-t5-base
- **Similarity Metric**: Cosine Similarity
- **API Integration**: [JIRA Python API](https://jira.readthedocs.io/api.html#jira.client.JIRA.status)

