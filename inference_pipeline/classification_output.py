import json

def save_classification_results(results, output_path):
    """
    Save classification results to a JSON file.

    Args:
        results (dict): Classification results.
        output_path (str): Path to save the results JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def print_classification_results(results):
    """
    Print classification results to the console.

    Args:
        results (dict): Classification results.
    """
    for key, value in results.items():
        print(f"{key}: {value}")