#!/usr/bin/env python3
"""
gdrive.py

A simple script to upload or download a file from Google Drive using PyDrive2.

Usage:
  To upload a file:
      python gdrive.py upload --file /path/to/local_file --title "Desired Title"
  To download a file:
      python gdrive.py download --id FILE_ID --output /path/to/save_file

Requirements:
  - PyDrive2: Install using `pip install PyDrive2`
  - A browser for OAuth (the first run will prompt you to authenticate)
  - Your credentials will be saved to "mycreds.txt" in the current directory.
"""

import argparse
import sys
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def authenticate():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)


def upload_file(drive, file_path, title):
    try:
        file = drive.CreateFile({'title': title})
        file.SetContentFile(file_path)
        file.Upload()
        print(f"File uploaded successfully! File ID: {file['id']}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")


def download_file(drive, file_id, output_path):
    try:
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(output_path)
        print(f"File downloaded successfully and saved as: {output_path}")
    except Exception as e:
        print(f"An error occurred during download: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="A simple script to upload or download files from Google Drive using PyDrive2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a file to Google Drive")
    upload_parser.add_argument("--file", type=str, required=True, help="Path to the local file to upload")
    upload_parser.add_argument("--title", type=str, required=True, help="Title for the file in Google Drive")

    download_parser = subparsers.add_parser("download", help="Download a file from Google Drive")
    download_parser.add_argument("--id", type=str, required=True, help="Google Drive file ID to download")
    download_parser.add_argument("--output", type=str, required=True, help="Path to save the downloaded file")

    args = parser.parse_args()

    drive = authenticate()

    if args.command == "upload":
        upload_file(drive, args.file, args.title)
    elif args.command == "download":
        download_file(drive, args.id, args.output)
    else:
        print("Unknown command")
        sys.exit(1)


if __name__ == "__main__":
    main()
