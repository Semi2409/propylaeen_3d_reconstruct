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
        # Check if dataset has any images loaded
        if len(self.image_paths) == 0:
            raise IndexError("Cannot retrieve item: Dataset is empty.")

        # Check for valid index boundaries (supports negative indexing like -1)
        if idx < -len(self.image_paths) or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} out of range for dataset with length {len(self.image_paths)}."
            )

        path = str(self.image_paths[idx])  # Safe to access now
        img = cv2.imread(path)  #[cite: 1]

        if img is None:  #[cite: 1]
            raise FileNotFoundError(
                f"Could not read image at {path}"
            )  #[cite: 1]

        return img  #[cite: 1]