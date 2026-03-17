import zipfile
import os

def extract_all_zips(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_path = os.path.splitext(zip_path)[0]

                # Skip if already extracted
                if os.path.exists(extract_path):
                    continue

                print(f"Extracting: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(extract_path)

                # Recursively check inside the extracted folder
                extract_all_zips(extract_path)

# Usage
extract_all_zips("Data/")