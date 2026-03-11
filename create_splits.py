import os
import random
import shutil




# ----------------------------------
# CONFIGURATION
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")
SPLITS_DIR = os.path.join(PROJECT_DIR, "splits")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ----------------------------------
# CREATE SPLIT FOLDERS
# ----------------------------------
def create_split_folders(labels):
    for split in ["train", "val", "test"]:
        for label in labels:
            path = os.path.join(SPLITS_DIR, split, label)
            os.makedirs(path, exist_ok=True)

# ----------------------------------
# SPLIT DATA
# ----------------------------------
def split_dataset():

    labels = [
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ]

    create_split_folders(labels)

    for label in labels:

        label_path = os.path.join(PROCESSED_DIR, label)
        clips = os.listdir(label_path)

        random.shuffle(clips)

        total = len(clips)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_clips = clips[:train_end]
        val_clips = clips[train_end:val_end]
        test_clips = clips[val_end:]

        print(f"\n📂 {label}")
        print(f"Total: {total}")
        print(f"Train: {len(train_clips)}")
        print(f"Val: {len(val_clips)}")
        print(f"Test: {len(test_clips)}")

        for clip in train_clips:
            shutil.move(
                os.path.join(label_path, clip),
                os.path.join(SPLITS_DIR, "train", label, clip)
            )

        for clip in val_clips:
            shutil.move(
                os.path.join(label_path, clip),
                os.path.join(SPLITS_DIR, "val", label, clip)
            )

        for clip in test_clips:
            shutil.move(
                os.path.join(label_path, clip),
                os.path.join(SPLITS_DIR, "test", label, clip)
            )

    print("\n✅ Dataset Split Complete!")


if __name__ == "__main__":
    split_dataset()