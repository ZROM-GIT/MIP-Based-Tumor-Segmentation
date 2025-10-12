import json
import re


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def find_unique_patterns(data):
    # Pattern to match 'PETCT_' followed by exactly 10 alphanumeric characters (letters or digits)
    pattern = re.compile(r'PETCT_[A-Za-z0-9]{10}')
    unique_patterns = set()

    def traverse(d):
        if isinstance(d, dict):
            for key, value in d.items():
                traverse(value)  # Recursively process the value
        elif isinstance(d, list):
            for item in d:
                traverse(item)  # Recursively process each item in the list
        elif isinstance(d, str):
            matches = pattern.findall(d)
            if matches:
                unique_patterns.update(matches)

    traverse(data)
    return unique_patterns


def main():
    json_file_path = '/mnt/sda1/PET/json_datasets/archs_comparison/MIPs16_75th_25vth_IncSplit_0_180.json'  # Replace with the path to your JSON file

    data = load_json(json_file_path)
    unique_patterns = find_unique_patterns(data)

    print("Unique 'PETCTxxxxxxxxxx' patterns found:")
    for pattern in unique_patterns:
        print(pattern)

    print('Number of unique patterns found:', len(unique_patterns))

if __name__ == "__main__":
    main()
