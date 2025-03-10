import os
import numpy as np
import requests
from tqdm.auto import tqdm
import subprocess
import hashlib

def download(url, dest, force=False, chunk_size=2**20, verbose=False):
    dest_dir, dest_name = os.path.split(dest)
    if force or not(os.path.exists(dest)):
        if os.path.exists(dest):
            os.remove(dest)
        os.makedirs(dest_dir, exist_ok=True)
        if verbose:
            print(f'Downloading to path \'{dest}\' from url \'{url}\'...')
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            if verbose:
                data_iter = tqdm(
                    response.iter_content(chunk_size=chunk_size),
                    total=int(np.ceil(int(response.headers['Content-length'])/chunk_size)),
                    unit='MB'
                )
            else:
                data_iter = response.iter_content(chunk_size=chunk_size)
            for data in data_iter:
                f.write(data)

def unzip(filename, base_dir, split=False):
    if split:
        combined_name = 'combined.zip'
        subprocess.call(['zip', '-FF', os.path.join(base_dir, filename), '--out', os.path.join(base_dir, combined_name)])
        subprocess.call(['unzip', '-FF', os.path.join(base_dir, combined_name), '-d', base_dir])
        os.remove(os.path.join(base_dir, combined_name))
    else:
        subprocess.call(['unzip', os.path.join(base_dir, filename), '-d', base_dir])

# Based on https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def verify_sha256(path, sha256):
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    digest = h.hexdigest()
    return digest == sha256