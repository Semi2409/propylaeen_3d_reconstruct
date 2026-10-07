import glob
import re
from pathlib import Path
import cv2

# Path to the current script file (src/dataset.py)
SCRIPT_DIR = Path(__file__).resolve().parent

# Move one level up to get the project root directory
PROJECT_ROOT = SCRIPT_DIR.parent

# Construct the exact path to the data directory
DATA_DIR = PROJECT_ROOT / "data"

def extract_img_number(file_path: Path) -> int:
    """
    Extracts numeric index from filename
    Falls back to 0 if no digit is found 
    """
    match = re.search(r'\d+', file_path.name)
    return int(match.group()) if match else 0
# Usage with Python's sorted()
def get_image_paths(data_dir: Path = DATA_DIR) -> list[Path]:
    """
    gathers all files that end on given extension and sorts them numerically
    """
    extensions = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for item in data_dir.iterdir():
        # check for file existence and valid image extension
        if item.is_file() and item.suffix.lower() in extensions:
            paths.append(item)
    # Sort with numeric extractor
    return sorted(paths, key = extract_img_number)

class ImageDataset:
    def __init__(self, data_dir: Path = DATA_DIR):
        #Store the list of Path objects in memory
        self.image_paths = get_image_paths(data_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx:int):
        path = str(self.image_paths[idx]) # Path corresponding to idx
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return img