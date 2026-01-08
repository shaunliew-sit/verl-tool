#!/usr/bin/env python3
"""
Upload training checkpoints to AWS S3.

Usage:
    python scripts/upload_to_s3.py --checkpoint-dir ./checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2 --bucket hoi-dataset --prefix checkpoints-new/

    #upload the hoi_cof_sft dataset
    python scripts/upload_to_s3.py --checkpoint-dir ./data/hoi_cof_sft --bucket hoi-dataset --prefix data/
    #upload the sft checkpoint
    python scripts/upload_to_s3.py --checkpoint-dir ./saves/qwen3-vl-8b/lora/hoi_cof_sft --bucket hoi-dataset --prefix saves/qwen3-vl-8b/lora/hoi_cof_sft/
Environment Variables (or use AWS CLI profile):
    AWS_ACCESS_KEY_ID: Your AWS access key
    AWS_SECRET_ACCESS_KEY: Your AWS secret key
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("boto3 is not installed. Install it with: pip install boto3")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_s3_client(
    profile_name: Optional[str] = None,
    region: Optional[str] = None,
) -> boto3.client:
    """Create S3 client with credentials."""
    session_kwargs = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    if region:
        session_kwargs["region_name"] = region
    
    session = boto3.Session(**session_kwargs)
    return session.client("s3")


def get_all_files(directory: Path) -> list[Path]:
    """Recursively get all files in a directory."""
    files = []
    for path in directory.rglob("*"):
        if path.is_file():
            files.append(path)
    return files


def should_upload(
    s3_client: boto3.client,
    local_path: Path,
    bucket: str,
    s3_key: str,
) -> bool:
    """Check if file needs to be uploaded (doesn't exist or size differs)."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        s3_size = response["ContentLength"]
        local_size = local_path.stat().st_size
        # Skip if sizes match
        return s3_size != local_size
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # File doesn't exist in S3, needs upload
            return True
        raise


def upload_file(
    s3_client: boto3.client,
    local_path: Path,
    bucket: str,
    s3_key: str,
    skip_existing: bool = False,
) -> tuple[bool, str, Optional[str], bool]:
    """Upload a single file to S3. Returns (success, path, error, skipped)."""
    try:
        if skip_existing and not should_upload(s3_client, local_path, bucket, s3_key):
            return True, str(local_path), None, True  # Skipped
        s3_client.upload_file(str(local_path), bucket, s3_key)
        return True, str(local_path), None, False  # Uploaded
    except ClientError as e:
        return False, str(local_path), str(e), False


def upload_directory(
    checkpoint_dir: str,
    bucket: str,
    prefix: str = "",
    profile_name: Optional[str] = None,
    region: Optional[str] = None,
    max_workers: int = 8,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict:
    """
    Upload all files from a checkpoint directory to S3.
    
    Args:
        checkpoint_dir: Local directory containing checkpoints
        bucket: S3 bucket name
        prefix: S3 key prefix (e.g., 'checkpoints/run_name/')
        profile_name: AWS profile name (optional)
        region: AWS region (optional)
        max_workers: Number of parallel upload threads
        dry_run: If True, only show what would be uploaded
        skip_existing: If True, skip files that already exist in S3 with same size
    
    Returns:
        Dictionary with upload statistics
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if not checkpoint_path.is_dir():
        raise ValueError(f"Path is not a directory: {checkpoint_dir}")
    
    # Get all files to upload
    files = get_all_files(checkpoint_path)
    
    if not files:
        print(f"No files found in {checkpoint_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # Normalize prefix
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    
    # Create file -> S3 key mapping
    uploads = []
    for file_path in files:
        relative_path = file_path.relative_to(checkpoint_path)
        s3_key = f"{prefix}{relative_path}"
        uploads.append((file_path, s3_key))
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f, _ in uploads)
    total_size_gb = total_size / (1024**3)
    
    print(f"\n{'=' * 60}")
    print(f"S3 Upload Summary")
    print(f"{'=' * 60}")
    print(f"Source:      {checkpoint_dir}")
    print(f"Destination: s3://{bucket}/{prefix}")
    print(f"Files:       {len(uploads)}")
    print(f"Total size:  {total_size_gb:.2f} GB")
    if skip_existing:
        print(f"Mode:        Skip existing (only upload new/changed files)")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        print("Dry run - files that would be uploaded:")
        for file_path, s3_key in uploads[:20]:
            print(f"  {file_path} -> s3://{bucket}/{s3_key}")
        if len(uploads) > 20:
            print(f"  ... and {len(uploads) - 20} more files")
        return {"total": len(uploads), "success": 0, "failed": 0, "dry_run": True}
    
    # Create S3 client
    try:
        s3_client = get_s3_client(profile_name, region)
        # Test credentials
        s3_client.head_bucket(Bucket=bucket)
    except NoCredentialsError:
        print("\nError: AWS credentials not found!")
        print("Please configure credentials using one of these methods:")
        print("  1. Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("  2. Use AWS CLI: aws configure")
        print("  3. Use --profile flag with a configured AWS profile")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            print(f"\nError: Bucket '{bucket}' does not exist!")
        elif error_code == "403":
            print(f"\nError: Access denied to bucket '{bucket}'!")
        else:
            print(f"\nError accessing bucket: {e}")
        sys.exit(1)
    
    # Upload files with progress bar
    results = {"total": len(uploads), "success": 0, "failed": 0, "skipped": 0, "errors": []}
    
    if tqdm:
        desc = "Syncing" if skip_existing else "Uploading"
        progress = tqdm(total=len(uploads), desc=desc, unit="file")
    else:
        progress = None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file, s3_client, local_path, bucket, s3_key, skip_existing): (local_path, s3_key)
            for local_path, s3_key in uploads
        }
        
        for future in as_completed(futures):
            success, path, error, skipped = future.result()
            if success:
                if skipped:
                    results["skipped"] += 1
                else:
                    results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append((path, error))
            
            if progress:
                progress.update(1)
            else:
                # Simple progress for when tqdm is not available
                processed = results["success"] + results["failed"] + results["skipped"]
                pct = processed / results["total"] * 100
                print(f"\rProgress: {pct:.1f}%", end="", flush=True)
    
    if progress:
        progress.close()
    else:
        print()  # New line after progress
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Upload Complete")
    print(f"{'=' * 60}")
    print(f"Uploaded:   {results['success']}/{results['total']}")
    if skip_existing:
        print(f"Skipped:    {results['skipped']}/{results['total']} (already exist)")
    print(f"Failed:     {results['failed']}/{results['total']}")
    
    if results["errors"]:
        print(f"\nFailed uploads:")
        for path, error in results["errors"][:10]:
            print(f"  {path}: {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    print(f"\nFiles uploaded to: s3://{bucket}/{prefix}")
    print(f"{'=' * 60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Upload training checkpoints to AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload checkpoints with default settings
  python upload_to_s3.py --checkpoint-dir ./checkpoints/my_run --bucket my-bucket

  # Upload with custom prefix
  python upload_to_s3.py --checkpoint-dir ./checkpoints/my_run --bucket my-bucket --prefix experiments/v2/

  # Skip existing files (sync mode - only upload new/changed files)
  python upload_to_s3.py --checkpoint-dir ./checkpoints/my_run --bucket my-bucket --skip-existing

  # Use specific AWS profile
  python upload_to_s3.py --checkpoint-dir ./checkpoints/my_run --bucket my-bucket --profile my-profile

  # Dry run to see what would be uploaded
  python upload_to_s3.py --checkpoint-dir ./checkpoints/my_run --bucket my-bucket --dry-run
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory to upload",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 key prefix (e.g., 'checkpoints/run_name/')",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (from ~/.aws/credentials)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: from environment or us-east-1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel upload threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in S3 with the same size (sync mode)",
    )
    
    args = parser.parse_args()
    
    try:
        results = upload_directory(
            checkpoint_dir=args.checkpoint_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            profile_name=args.profile,
            region=args.region,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        
        # Exit with error code if any uploads failed
        if results["failed"] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


