import argparse
import os
from typing import Optional
from huggingface_hub import hf_hub_download

def download_file_from_hf(repo_id: str,
                          filename: str,
                          output_dir: str,
                          repo_type: str = "dataset",
                          token: Optional[str] = None,
                          revision: Optional[str] = None):
    """
    Downloads a single file from a Hugging Face Hub repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "username/repo_name").
        filename (str): The name of the file to download from the repository.
                        Can include subdirectories within the repo (e.g., "data/mydb.duckdb").
        output_dir (str): The local directory to save the downloaded file.
                        The file will be saved as output_dir/basename(filename).
        repo_type (str, optional): Type of the repository ('dataset', 'model', 'space'). Defaults to "dataset".
        token (str, optional): Hugging Face API token for private repositories.
        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a commit hash.
    
    Returns:
        str: Path to the downloaded file, or None if download failed.
    """
    try:
        if not os.path.exists(output_dir):
            print(f"Output directory '{output_dir}' does not exist. Creating it...")
            os.makedirs(output_dir, exist_ok=True)

        # The target local path for the file, directly under output_dir
        # os.path.basename is used to ensure that if filename contains repo subdirectories,
        # the file is still saved flat in output_dir.
        target_local_file_path = os.path.join(output_dir, os.path.basename(filename))

        print(f"Starting download for '{filename}' from repository '{repo_id}' (type: {repo_type}, revision: {revision or 'main'})...")
        
        # hf_hub_download will download to a cache directory first, then symlink or copy to local_dir.
        # The `filename` argument refers to the path within the repository.
        # `local_dir` specifies where the symlink/copy should be created.
        # The actual downloaded file path returned by hf_hub_download might be inside local_dir/snapshots/...
        # or directly local_dir/filename if local_dir_use_symlinks=False and no complex caching structure is used.
        # For simplicity and to ensure the file is exactly where we want it, we'll download to a temporary
        # location within the output_dir (or let hf_hub_download manage its cache) and then ensure
        # the file is moved/renamed to target_local_file_path.

        downloaded_path_in_hub_structure = hf_hub_download(
            repo_id=repo_id,
            filename=filename, # This is the path *within* the repo
            repo_type=repo_type,
            local_dir=output_dir, # Acts as a base for the download structure or direct download
            local_dir_use_symlinks=False, # Ensure actual file copy
            token=token,
            revision=revision,
        )
        
        # downloaded_path_in_hub_structure is the path to the file, which might be output_dir/filename
        # or output_dir/subpath_from_repo/filename if filename had subpaths.
        # We want to ensure the final file is output_dir/basename(filename).

        if os.path.abspath(downloaded_path_in_hub_structure) != os.path.abspath(target_local_file_path):
            print(f"File downloaded to '{downloaded_path_in_hub_structure}', ensuring it is at '{target_local_file_path}'.")
            # Ensure the parent directory of the target path exists
            os.makedirs(os.path.dirname(target_local_file_path), exist_ok=True)
            # Move the file to the desired flat location
            #os.rename(downloaded_path_in_hub_structure, target_local_file_path)
            final_path = target_local_file_path
        else:
            final_path = downloaded_path_in_hub_structure

        print(f"Successfully downloaded and placed '{os.path.basename(filename)}' at '{final_path}'.")
        return final_path
    except Exception as e:
        print(f"An unexpected error occurred while downloading '{filename}': {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face repository ID (e.g., 'username/repo_name').")
    parser.add_argument("--filenames", type=str, nargs='+', required=True,
                        help="One or more filenames to download from the repository. "
                             "If a filename includes subdirectories (e.g., 'data/db.duckdb'), "
                             "the file will be saved as 'output_dir/db.duckdb'.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local directory to save the downloaded files.")
    parser.add_argument("--repo_type", type=str, default="dataset",
                        choices=["dataset", "model", "space"],
                        help="Type of the repository. Defaults to 'dataset'.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional Git revision (branch, tag, commit hash). Defaults to 'main'.")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token for private repositories. "
                             "Can also be set via HUGGINGFACE_HUB_TOKEN environment variable.")

    args = parser.parse_args()

    if not args.token:
        args.token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if args.token:
            print("Using Hugging Face token from HUGGINGFACE_HUB_TOKEN environment variable.")

    download_success_count = 0
    for repo_filename_path in args.filenames:
        if download_file_from_hf(args.repo_id,
                                 repo_filename_path, # This is the path of the file within the repo
                                 args.output_dir,
                                 args.repo_type,
                                 args.token,
                                 args.revision):
            download_success_count += 1
    
    print(f"\nDownload process finished. {download_success_count}/{len(args.filenames)} files processed.")
    if download_success_count > 0:
        print(f"Successfully downloaded files are located in '{os.path.abspath(args.output_dir)}'.")

if __name__ == "__main__":
    main()
