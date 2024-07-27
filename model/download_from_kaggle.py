import os
import subprocess
import argparse
import shutil
import zipfile

def download_kaggle_dataset(dataset, destination):
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', destination, '--unzip'], check=True)
    print(f"Downloaded and unzipped {dataset} to {destination}")

def move_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))
    print(f"Moved files from {src_dir} to {dest_dir}")
def print_directory_contents(path):
    print (f"path:{path}")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path} to {extract_to}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset from Kaggle')
    parser.add_argument('--dataset', type=str, required=True, help='Kaggle dataset identifier')
    parser.add_argument('--destination', type=str, required=True, help='Destination directory to unzip the dataset')

    args = parser.parse_args()
    os.makedirs(args.destination, exist_ok=True)
    temp_dir = os.path.join(args.destination, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    download_kaggle_dataset(args.dataset, args.destination)
    move_files(temp_dir, args.destination)
    print_directory_contents(args.destination)
    shutil.rmtree(temp_dir)