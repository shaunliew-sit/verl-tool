#!/usr/bin/env python3
"""
Download files from S3 bucket to local directory.

Usage:
    python scripts/download_s3.py --bucket hoi-dataset --prefix data/ --local-dir /workspace/verl-tool/data/
    
    # Download entire bucket
    python scripts/download_s3.py --bucket my-bucket --local-dir data/
    
    # Download specific folder
    python scripts/download_s3.py --bucket my-bucket --prefix datasets/hico/ --local-dir data/hico/
    
    # With specific AWS profile
    python scripts/download_s3.py --bucket my-bucket --prefix data/ --local-dir data/ --profile myprofile
    
    # With explicit credentials
    python scripts/download_s3.py --bucket my-bucket --local-dir data/ \
        --access-key AKIAXXXXXXXX --secret-key XXXXXXXX --region us-east-1
"""

import os
import argparse
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_s3_client(profile=None, access_key=None, secret_key=None, region=None):
    """Create S3 client with optional credentials."""
    if access_key and secret_key:
        return boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region or 'us-east-1',
            config=Config(max_pool_connections=50)
        )
    elif profile:
        session = boto3.Session(profile_name=profile)
        return session.client('s3', config=Config(max_pool_connections=50))
    else:
        return boto3.client('s3', config=Config(max_pool_connections=50))


def list_objects(s3_client, bucket, prefix=''):
    """List all objects in bucket with given prefix."""
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            # Skip directory markers
            if not obj['Key'].endswith('/'):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size']
                })
    
    return objects


def download_file(s3_client, bucket, key, local_path):
    """Download a single file."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    return key


def download_folder(
    bucket: str,
    prefix: str = '',
    local_dir: str = 'data/',
    profile: str = None,
    access_key: str = None,
    secret_key: str = None,
    region: str = None,
    max_workers: int = 10,
    dry_run: bool = False
):
    """Download all files from S3 bucket/prefix to local directory."""
    
    # Create S3 client
    s3_client = get_s3_client(profile, access_key, secret_key, region)
    
    # List all objects
    print(f"Listing objects in s3://{bucket}/{prefix}...")
    objects = list_objects(s3_client, bucket, prefix)
    
    if not objects:
        print("No objects found!")
        return
    
    # Calculate total size
    total_size = sum(obj['size'] for obj in objects)
    total_size_mb = total_size / (1024 * 1024)
    print(f"Found {len(objects)} files ({total_size_mb:.2f} MB)")
    
    if dry_run:
        print("\n[DRY RUN] Would download:")
        for obj in objects[:10]:
            print(f"  - {obj['key']} ({obj['size']} bytes)")
        if len(objects) > 10:
            print(f"  ... and {len(objects) - 10} more files")
        return
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Download files with progress bar
    print(f"\nDownloading to {local_dir}...")
    
    def download_task(obj):
        key = obj['key']
        # Remove prefix from key to get relative path
        relative_path = key[len(prefix):] if prefix else key
        local_path = os.path.join(local_dir, relative_path)
        download_file(s3_client, bucket, key, local_path)
        return obj['size']
    
    downloaded_size = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_task, obj): obj for obj in objects}
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for future in as_completed(futures):
                try:
                    size = future.result()
                    downloaded_size += size
                    pbar.update(size)
                except Exception as e:
                    obj = futures[future]
                    print(f"\nError downloading {obj['key']}: {e}")
    
    print(f"\nDownload complete! {len(objects)} files saved to {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download files from S3 bucket to local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire bucket
  python scripts/download_s3.py --bucket my-bucket --local-dir data/
  
  # Download specific folder
  python scripts/download_s3.py --bucket my-bucket --prefix datasets/hico/ --local-dir data/hico/
  
  # With AWS profile
  python scripts/download_s3.py --bucket my-bucket --local-dir data/ --profile myprofile
  
  # Dry run (see what would be downloaded)
  python scripts/download_s3.py --bucket my-bucket --local-dir data/ --dry-run
"""
    )
    
    parser.add_argument("--bucket", "-b", required=True,
                        help="S3 bucket name")
    parser.add_argument("--prefix", "-p", default='',
                        help="S3 prefix/folder path (default: root)")
    parser.add_argument("--local-dir", "-d", default='data/',
                        help="Local directory to download to (default: data/)")
    
    # Authentication options
    parser.add_argument("--profile", 
                        help="AWS profile name (from ~/.aws/credentials)")
    parser.add_argument("--access-key",
                        help="AWS access key ID")
    parser.add_argument("--secret-key",
                        help="AWS secret access key")
    parser.add_argument("--region", default='us-east-1',
                        help="AWS region (default: us-east-1)")
    
    # Other options
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Number of parallel download workers (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    
    args = parser.parse_args()
    
    download_folder(
        bucket=args.bucket,
        prefix=args.prefix,
        local_dir=args.local_dir,
        profile=args.profile,
        access_key=args.access_key,
        secret_key=args.secret_key,
        region=args.region,
        max_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
