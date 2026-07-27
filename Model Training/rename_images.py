from pathlib import Path
 
# ---------------------------------------------------------------------------
# Configuration - edit these to match your folder
# ---------------------------------------------------------------------------
FOLDER_PATH = Path(
    r"C:\Users\jecsh\OneDrive\Desktop\RSW Y3S1\Apple_Ripeness_Assessment"
    r"\Apple Ripeness Levels Image Dataset\Overripe"
)
PREFIX = "apple_overripe"          # files become apple_overripe_001.jpg, _002.jpg, ...
START_NUMBER = 1                   # first number to use
NUMBER_WIDTH = 3                   # zero-padding width, e.g. 3 -> 001, 4 -> 0001
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
 
 
def rename_images(folder: Path, prefix: str, start: int, width: int):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
 
    # Sort files for a stable, predictable order (by current filename)
    image_files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    )
 
    if not image_files:
        print(f"No image files found in {folder}")
        return
 
    print(f"Found {len(image_files)} images. Renaming...\n")
 
    # Step 1: rename everything to a temporary name first.
    # This avoids collisions when a new target name already exists
    # among the current files (e.g. renaming citra1.jpg -> apple_overripe_001.jpg
    # while apple_overripe_001.jpg already exists as a different file).
    temp_paths = []
    for i, path in enumerate(image_files):
        temp_path = path.with_name(f"__tmp_{i}__{path.suffix.lower()}")
        path.rename(temp_path)
        temp_paths.append(temp_path)
 
    # Step 2: rename from temp names to final sequential names
    for i, temp_path in enumerate(temp_paths):
        number = start + i
        new_name = f"{prefix}_{number:0{width}d}{temp_path.suffix.lower()}"
        new_path = temp_path.with_name(new_name)
        temp_path.rename(new_path)
        print(f"  -> {new_name}")
 
    print(f"\nDone. Renamed {len(temp_paths)} images in {folder}")
 
 
if __name__ == "__main__":
    rename_images(FOLDER_PATH, PREFIX, START_NUMBER, NUMBER_WIDTH)