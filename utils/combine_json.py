import os
import json

def read_json_files_from_folder(folder_path):
    """
    Read all JSON files from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        
    Returns:
        list: List of JSON data from each file
    """
    json_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename)) as f:
                data = json.load(f)
                json_data.append(data)
    return json_data

def combine_json_data(json_data_list):
    """
    Combine multiple JSON data dictionaries into one.
    
    Args:
        json_data_list (list): List of JSON data dictionaries
        
    Returns:
        dict: Combined JSON data
    """
    combined_data = {}
    for data in json_data_list:
        combined_data.update(data)
        # If you want to combine the data into a list, you can do:
        # combined_data.extend(data)
    return combined_data

def write_combined_json(combined_data, output_file):
    """
    Write combined JSON data to a file.
    
    Args:
        combined_data (dict): Combined JSON data
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

if __name__ == "__main__":
    # Paths for HaGRID annotation processing
    folder_path = "/media/okour/e12fd9d4-f6ed-4f14-9248-4e7960c486052/UNSW/Term2/COMP9444__Deep_Learning/Project/test/dataset/ann_subsample"  # Replace this with the actual path to your folder
    output_file = "/media/okour/e12fd9d4-f6ed-4f14-9248-4e7960c486052/UNSW/Term2/COMP9444__Deep_Learning/Project/test/dataset/ann_subsample.json"  # Replace this with the desired output file name

    json_data_list = read_json_files_from_folder(folder_path)
    combined_data = combine_json_data(json_data_list)
    write_combined_json(combined_data, output_file)
