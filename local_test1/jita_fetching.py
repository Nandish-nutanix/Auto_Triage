import sys
import json
import re
import argparse
import pandas as pd
from urllib.parse import urlparse, parse_qs
from framework.lib.nulog import INFO, STEP, WARN
from framework.exceptions.entity_error import NuTestError
from experimental.dr.lib.jita import gen, DraasJita

def get_failed_test_results(jita_url, username, password):
    STEP(f"Processing URL: {jita_url}")

    failed_results = []

    try:
        parsed_url = urlparse(jita_url)
        query_params = parse_qs(parsed_url.query)
        task_ids = query_params.get('task_ids', [])
        
        if not task_ids:
            raise ValueError("No valid task ID found in the provided JITA URL.")
        
        task_id = task_ids[0]
        INFO(f"Extracted Task ID: {task_id}")
    except Exception as e:
        INFO(f"Error parsing JITA URL: {str(e)}")
        return failed_results

    try:
        obj = DraasJita(username=username, password=password)
        task_url = f"https://jita-web-server-1.eng.nutanix.com/api/v2/agave_tasks/{task_id}"
        
        try:
            resp = obj.send_request(gen.RestMethods.GET, task_url)

            if resp.status_code != 200 or not resp.__dict__.get("_content"):
                raise Exception("Received an invalid or empty response from the JITA server.")
            
            INFO(f"Acquired results from task ID: {task_id}")
            
            task_data = json.loads(resp.__dict__["_content"])
            test_results = task_data.get("data", {}).get("AgaveTestResults", [])

            test_id_to_name = {}
            for test_set in task_data.get("data", {}).get("test_sets", []):
                for test in test_set.get("tests", []):
                    test_result_id = test.get("test_result_id", {}).get("$oid")
                    if test_result_id:
                        test_id_to_name[test_result_id] = test.get("name")

            for result in test_results:
                try:
                    test_id = result.get("$oid")
                    if not test_id:
                        continue
                    
                    test_url = f"https://jita-web-server-1.eng.nutanix.com/api/v2/agave_test_results/{test_id}"
                    test_resp = obj.send_request(gen.RestMethods.GET, test_url)
                    test_data = json.loads(test_resp.__dict__["_content"]).get("data", {})
                    
                    if test_data.get("status") == "Failed":
                        INFO(f"Found failed test with ID: {test_id}")
                        
                        test_name = test_id_to_name.get(test_id) or test_data.get("name")
                        
                        failed_results.append({
                            'Test Name': test_name,
                            'Test ID': test_id,
                            'Test Log URL': test_data.get("test_log_url"),
                            'Exception Summary': test_data.get("exception_summary"),
                            'Traceback': test_data.get("exception")
                        })
                        
                        INFO(f"Failed Test Name: {test_name}")
                        
                except Exception as e:
                    NuTestError(f"Error processing test result {test_id}: {str(e)}")
                    
        except Exception as e:
            NuTestError(str(e))
            INFO(f"Couldn't acquire test results from the provided JITA URL: {jita_url}")
            
    except Exception as e:
        INFO(f"Failed to retrieve test results from URL: {jita_url}")

    return failed_results

def process_urls_from_json(file_path):
    all_failed_results = []

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            urls = data.get("urls", [])
            username = data.get("username")
            password = data.get("password")

            if not urls:
                raise ValueError("No URLs found in the provided JSON file.")
            if not username or not password:
                raise ValueError("Username and password are required in the JSON file.")
    except Exception as e:
        INFO(f"Error reading URL JSON file: {str(e)}")
        sys.exit()

    for url in urls:
        results = get_failed_test_results(url, username, password)
        if results:
            all_failed_results.extend(results)

    if all_failed_results:
        df = pd.DataFrame(all_failed_results)
        
        output_file = 'failed_test_results.xlsx'
        df.to_excel(output_file, index=False)
        INFO(f"Failed test results have been saved to {output_file}.")
    else:
        INFO("No failed test results found to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JITA URLs from a JSON file to get failed test results.")
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing JITA URLs.')

    args = parser.parse_args()

    process_urls_from_json(args.file_path)



