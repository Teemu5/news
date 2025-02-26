#!/usr/bin/env python3
"""
upload_to_hf.py

A script to create a Hugging Face model repository (if needed) and upload a file.

Usage:
  python upload_to_hf.py --local_file_path PATH_TO_FILE --path_in_repo FILE_NAME_IN_REPO --repo_id "your-username/your-repo" [--repo_type model]
"""

import argparse
from huggingface_hub import create_repo, upload_file

def create_hf_repo(repo_id: str, repo_type: str):
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Failed to create or access repository: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Upload a file to a Hugging Face model/dataset repository, creating the repo if needed."
    )
    parser.add_argument(
        "--local_file_path", 
        type=str, 
        required=True,
        help="Path to the local file you want to upload."
    )
    parser.add_argument(
        "--path_in_repo", 
        type=str, 
        required=True,
        help="The path (or file name) in the repository where the file will be stored."
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Your repository ID on Hugging Face (e.g., 'Teemu5/news')."
    )
    parser.add_argument(
        "--repo_type", 
        type=str, 
        default="model",
        help="Type of the repository, 'model' (default) or 'dataset'."
    )
    
    args = parser.parse_args()
    
    # Create the repository if it does not exist
    create_hf_repo(args.repo_id, args.repo_type)
    
    # Upload the file
    try:
        upload_file(
            path_or_fileobj=args.local_file_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type
        )
        print("File uploaded successfully!")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()
