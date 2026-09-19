"""
Upload ONLY the 4 Input Specification Documents to GitHub.
Target Repository: https://github.com/arushitribhuwan-tech/Domain-specific-Question-Answering-with-RAG-Evaluation

Usage:
    python upload_input_docs_only.py <GITHUB_TOKEN>
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

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_documents")

FILES_TO_UPLOAD = {
    "1_Functional_Requirements_Document.md": "1_Functional_Requirements_Document.md",
    "2_Product_Requirements_Document.md": "2_Product_Requirements_Document.md",
    "3_Technical_Requirements_Document.md": "3_Technical_Requirements_Document.md",
    "4_Project_Directory_Structure.md": "4_Project_Directory_Structure.md"
}

def upload_file(repo_filename, local_filename, token):
    local_path = os.path.join(DOCS_DIR, local_filename)
    if not os.path.exists(local_path):
        print(f"  [ERROR] File not found: {local_path}")
        return False
        
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_filename}"
    
    with open(local_path, "rb") as f:
        content_bytes = f.read()
    
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Input-Docs-Uploader",
        "Content-Type": "application/json"
    }
    
    # Check if file exists to get sha
    sha = None
    try:
        check_req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(check_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            sha = data.get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  [Warning] Checking {repo_filename}: HTTP {e.code}")
            
    payload = {
        "message": f"Add input document: {repo_filename}",
        "content": content_b64,
        "branch": BRANCH
    }
    if sha:
        payload["sha"] = sha
        payload["message"] = f"Update input document: {repo_filename}"
        
    req_data = json.dumps(payload).encode("utf-8")
    put_req = urllib.request.Request(url, data=req_data, headers=headers, method="PUT")
    
    try:
        with urllib.request.urlopen(put_req) as resp:
            if resp.status in (200, 201):
                print(f"  [SUCCESS] Uploaded: {repo_filename}")
                return True
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        print(f"  [FAILED] {repo_filename}: HTTP {e.code} - {error_msg}")
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
        
    print(f"\nUploading ONLY your 4 input documents to: https://github.com/{REPO_OWNER}/{REPO_NAME}")
    print("-" * 70)
    
    success_count = 0
    for repo_file, local_file in FILES_TO_UPLOAD.items():
        if upload_file(repo_file, local_file, token):
            success_count += 1
            
    print("-" * 70)
    print(f"Completed! {success_count}/{len(FILES_TO_UPLOAD)} input documents uploaded successfully.")
    print(f"View your repository at: https://github.com/{REPO_OWNER}/{REPO_NAME}\n")

if __name__ == "__main__":
    main()
