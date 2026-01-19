"""
Hugging Face Hub integration for model upload/download.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download, login, whoami
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError


# Default repository settings
DEFAULT_REPO_ID = "luismarcelovaf/motherboard-fault-detector"
TOKEN_FILE = Path(__file__).parent.parent.parent / ".hf_token"
MODEL_FILES = {
    "patchcore": "patchcore.pt",
    "classifier": "classifier.pt",
}

# Auto-login on module load
_logged_in = False


def _get_token() -> Optional[str]:
    """Get token from environment or token file."""
    # Check environment variable first
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # Check token file
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text().strip()
            if token:
                return token
        except Exception:
            pass

    return None


def _prompt_for_token() -> Optional[str]:
    """Prompt user for Hugging Face token and save it."""
    print("\n" + "=" * 50)
    print("HUGGING FACE TOKEN SETUP")
    print("=" * 50)
    print("\nTo upload/download models, you need a Hugging Face token.")
    print("\n1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a token with 'write' access")
    print("3. Paste it below")
    print("\nThis only needs to be done once.")
    print("(Press Enter to skip if you only want to use local models)")

    token = input("\nHugging Face Token: ").strip()

    if token:
        try:
            # Verify token works
            login(token=token, add_to_git_credential=False)
            # Save token
            TOKEN_FILE.write_text(token)
            print(f"\n[OK] Token saved to {TOKEN_FILE.name}")
            return token
        except Exception as e:
            print(f"\n[ERROR] Invalid token: {e}")
            return None
    else:
        print("\n[SKIP] No token provided. You can set it later.")
        return None


def _auto_login():
    """Automatically login with stored token."""
    global _logged_in
    if _logged_in:
        return True

    token = _get_token()
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            _logged_in = True
            return True
        except Exception:
            # Token might be invalid, delete it
            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink()
            return False

    return False


def get_repo_id(prompt_if_missing: bool = True) -> str:
    """Get repository ID from environment or use default."""
    env_repo = os.environ.get("HF_REPO_ID")
    if env_repo:
        return env_repo
    return DEFAULT_REPO_ID


def is_logged_in() -> bool:
    """Check if user is logged in to Hugging Face."""
    _auto_login()
    try:
        whoami()
        return True
    except Exception:
        return False


def ensure_login() -> bool:
    """Ensure user is logged in, prompt for token if needed."""
    if _auto_login():
        return True

    # No token found, prompt for one
    token = _prompt_for_token()
    if token:
        return _auto_login()

    return False


def upload_model(
    local_path: Path,
    model_type: str,
    repo_id: Optional[str] = None,
) -> bool:
    """
    Upload a model to Hugging Face Hub.

    Args:
        local_path: Path to the local model file
        model_type: Type of model ('patchcore' or 'classifier')
        repo_id: Repository ID (default: from env or DEFAULT_REPO_ID)

    Returns:
        True if upload successful, False otherwise
    """
    if not local_path.exists():
        print(f"[ERROR] Model file not found: {local_path}")
        return False

    if not ensure_login():
        print("[WARN] Skipping upload - not logged in to Hugging Face")
        return False

    repo_id = repo_id or get_repo_id()
    filename = MODEL_FILES.get(model_type, f"{model_type}.pt")

    print(f"\nUploading {model_type} model to Hugging Face...")
    print(f"  Repository: {repo_id}")
    print(f"  File: {filename}")

    try:
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            print(f"[OK] Repository ready: {repo_id}")
        except Exception as e:
            error_msg = str(e)
            if "reserved" in error_msg.lower():
                print(f"[ERROR] Repository name is reserved. Please choose a different name.")
                return False
            elif "401" in error_msg or "403" in error_msg:
                print(f"[ERROR] Authentication failed. Please run: huggingface-cli login")
                return False
            # Repo might already exist, continue
            print(f"[INFO] Using existing repository")

        # Upload file
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"[OK] Model uploaded successfully!")
        print(f"     URL: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"[ERROR] Repository not found: {repo_id}")
            print(f"        Please create it at: https://huggingface.co/new")
        elif "401" in error_msg or "403" in error_msg:
            print(f"[ERROR] Not authorized. Please run: huggingface-cli login")
        else:
            print(f"[ERROR] Upload failed: {e}")
        return False


def download_model(
    local_path: Path,
    model_type: str,
    repo_id: Optional[str] = None,
) -> bool:
    """
    Download a model from Hugging Face Hub.

    Args:
        local_path: Path to save the model locally
        model_type: Type of model ('patchcore' or 'classifier')
        repo_id: Repository ID (default: from env or DEFAULT_REPO_ID)

    Returns:
        True if download successful, False otherwise
    """
    repo_id = repo_id or get_repo_id()
    filename = MODEL_FILES.get(model_type, f"{model_type}.pt")

    print(f"\nDownloading {model_type} model from Hugging Face...")
    print(f"  Repository: {repo_id}")
    print(f"  File: {filename}")

    try:
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )

        # Copy to local path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(downloaded_path, local_path)

        print(f"[OK] Model downloaded successfully!")
        print(f"     Saved to: {local_path}")
        return True

    except RepositoryNotFoundError:
        print(f"[ERROR] Repository not found: {repo_id}")
        print("        You may need to train the models first and upload them.")
        return False
    except EntryNotFoundError:
        print(f"[ERROR] Model file not found in repository: {filename}")
        print("        You may need to train this model first and upload it.")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def check_remote_models(repo_id: Optional[str] = None) -> dict:
    """
    Check which models are available on Hugging Face.

    Returns:
        Dict with model availability status
    """
    repo_id = repo_id or get_repo_id()
    status = {}

    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        for model_type, filename in MODEL_FILES.items():
            status[model_type] = filename in files

    except RepositoryNotFoundError:
        for model_type in MODEL_FILES:
            status[model_type] = False
    except Exception:
        for model_type in MODEL_FILES:
            status[model_type] = False

    return status


def ensure_models_exist(
    patchcore_path: Path,
    classifier_path: Path,
    repo_id: Optional[str] = None,
) -> bool:
    """
    Ensure both models exist locally, downloading from HF if needed.

    Returns:
        True if both models are available, False otherwise
    """
    all_exist = True

    # Check PatchCore
    if not patchcore_path.exists():
        print(f"\n[!] PatchCore model not found locally: {patchcore_path}")
        if not download_model(patchcore_path, "patchcore", repo_id):
            all_exist = False

    # Check Classifier
    if not classifier_path.exists():
        print(f"\n[!] Classifier model not found locally: {classifier_path}")
        if not download_model(classifier_path, "classifier", repo_id):
            all_exist = False

    return all_exist
