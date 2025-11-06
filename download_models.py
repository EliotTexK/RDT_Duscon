#!/usr/bin/env python3

# Download and make symlinks as per the README

import os
from pathlib import Path
from transformers import AutoProcessor, AutoModel

# Download models
print("Downloading siglip model...")
AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir='model_downloads')
AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", cache_dir='model_downloads')

print("Downloading t5 model...")
AutoModel.from_pretrained("google/t5-v1_1-xxl", cache_dir='model_downloads')
AutoProcessor.from_pretrained("google/t5-v1_1-xxl", cache_dir='model_downloads')

# Create google directory
os.makedirs('google', exist_ok=True)

# Find the snapshot directories
cache_dir = Path('model_downloads')

siglip_snapshots = cache_dir / 'models--google--siglip-so400m-patch14-384' / 'snapshots'
siglip_hash = next(siglip_snapshots.iterdir()).name

t5_snapshots = cache_dir / 'models--google--t5-v1_1-xxl' / 'snapshots'
t5_hash = next(t5_snapshots.iterdir()).name

# Create symlinks with absolute paths
src_siglip = (cache_dir / 'models--google--siglip-so400m-patch14-384' / 'snapshots' / siglip_hash).resolve()
dst_siglip = Path('google/siglip-so400m-patch14-384').resolve()

src_t5 = (cache_dir / 'models--google--t5-v1_1-xxl' / 'snapshots' / t5_hash).resolve()
dst_t5 = Path('google/t5-v1_1-xxl').resolve()

# Remove existing symlinks if they exist
dst_siglip.unlink(missing_ok=True)
dst_t5.unlink(missing_ok=True)

# Create the symlinks
os.symlink(src_siglip, dst_siglip)
os.symlink(src_t5, dst_t5)

print(f"Created symlink: {dst_siglip} -> {src_siglip}")
print(f"Created symlink: {dst_t5} -> {src_t5}")