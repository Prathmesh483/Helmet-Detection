import os
import urllib.request
import sys

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Official ultralytics yolov8n asset (small). If this 404s later, replace with another URL or commit weights.
YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v0.0/yolov8n.pt"
YOLOV8N_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

def download(url, dest, min_bytes=200_000):
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size >= min_bytes:
            print(f"{dest} already exists (size {size} bytes).")
            return True
        else:
            print(f"{dest} exists but is too small ({size} bytes). Re-downloading.")
            os.remove(dest)
    print(f"Downloading {url} -> {dest} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print("Download failed:", e)
        return False
    size = os.path.getsize(dest)
    if size < min_bytes:
        print("Downloaded file too small (likely corrupted). Removing and aborting.")
        os.remove(dest)
        return False
    print("Downloaded:", dest, "size:", size)
    return True

if __name__ == "__main__":
    ok = download(YOLOV8N_URL, YOLOV8N_PATH)
    if not ok:
        print("Failed to download yolov8n.pt. You can also let ultralytics auto-download on first run by not providing the file.")
        sys.exit(1)
    print("All model downloads complete.")
