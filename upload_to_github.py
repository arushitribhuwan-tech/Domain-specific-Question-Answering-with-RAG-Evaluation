"""
GitHub Direct Uploader Script
Uploads all project files directly to GitHub using GitHub REST API.
Does NOT require Git to be installed!

Usage:
    python upload_to_github.py <GITHUB_PERSONAL_ACCESS_TOKEN>
"""

import os
import sys
import base64
import json
import urllib.request
import urllib.error

REPO_OWNER = "arushitribhuwan-tech"
REPO_NAME = "Domain-specific-Question-Answering-with-RAG-Evaluation"
BRANCH = "main"

# Files and extensions to skip
IGNORE_EXTS = {".pyc", ".docx"}
IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", ".venv", "chroma_db"}
IGNORE_FILES = {"Complete_Project_Code.txt", "Complete_Project_Code.docx", "Domain_Specific_QA_RAG_Project_Documentation.docx", "create_word_doc.py", "upload_to_github.py"}

def get_files_to_upload(root_dir):
    files_map = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IGNORE_EXTS or f in IGNORE_FILES:
                continue
            
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
            files_map[rel_path] = full_path
    return files_map

def upload_file(rel_path, abs_path, token):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{rel_path}"
    
    with open(abs_path, "rb") as f:
        content_bytes = f.read()
    
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RAG-Project-Uploader",
        "Content-Type": "application/json"
    }
    
    # Check if file already exists to obtain its SHA (for updating)
    sha = None
    try:
        check_req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(check_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            sha = data.get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  [Warning] Checking {rel_path}: HTTP {e.code}")
    
    payload = {
        "message": f"Add {rel_path}",
        "content": content_b64,
        "branch": BRANCH
    }
    if sha:
        payload["sha"] = sha
        payload["message"] = f"Update {rel_path}"
        
    req_data = json.dumps(payload).encode("utf-8")
    put_req = urllib.request.Request(url, data=req_data, headers=headers, method="PUT")
    
    try:
        with urllib.request.urlopen(put_req) as resp:
            if resp.status in (200, 201):
                print(f"  [SUCCESS] Uploaded: {rel_path}")
                return True
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        print(f"  [FAILED] {rel_path}: HTTP {e.code} - {error_msg}")
        return False
    return False

def main():
    if len(sys.argv) < 2:
        token = input("Enter your GitHub Personal Access Token (PAT): ").strip()
    else:
        token = sys.argv[1].strip()
        
    if not token:
        print("Error: GitHub token is required to upload files to your repository.")
        sys.exit(1)
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    files = get_files_to_upload(project_root)
    
    print(f"\nTarget Repository: https://github.com/{REPO_OWNER}/{REPO_NAME}")
    print(f"Found {len(files)} files to upload from '{project_root}'.\n")
    
    success_count = 0
    for rel_path, abs_path in files.items():
        if upload_file(rel_path, abs_path, token):
            success_count += 1
            
    print(f"\nUpload complete! {success_count}/{len(files)} files uploaded successfully.")
    print(f"View your repository at: https://github.com/{REPO_OWNER}/{REPO_NAME}\n")

if __name__ == "__main__":
    main()
