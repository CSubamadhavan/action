import os
import random
import shutil



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up from scripts/
PROJECT_DIR = os.path.dirname(BASE_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")

random.seed(42)

# -----------------------------------
# COUNT CLIPS
# -----------------------------------
class_counts = {}

for label in os.listdir(PROCESSED_DIR):
    label_path = os.path.join(PROCESSED_DIR, label)
    if os.path.isdir(label_path):
        clips = os.listdir(label_path)
        class_counts[label] = len(clips)

print("\n📊 Before Balancing:")
for label, count in class_counts.items():
    print(f"{label}: {count} clips")

# -----------------------------------
# FIND MINIMUM CLASS SIZE
# -----------------------------------
min_count = min(class_counts.values())

print(f"\n🎯 Target clips per class: {min_count}")

# -----------------------------------
# BALANCE BY DOWNSAMPLING
# -----------------------------------
for label in class_counts:
    label_path = os.path.join(PROCESSED_DIR, label)
    clips = os.listdir(label_path)

    if len(clips) > min_count:
        clips_to_remove = random.sample(clips, len(clips) - min_count)

        for clip in clips_to_remove:
            clip_path = os.path.join(label_path, clip)
            shutil.rmtree(clip_path)

print("\n✅ Balancing Complete!\n")

# -----------------------------------
# FINAL COUNT
# -----------------------------------
print("📊 After Balancing:")
for label in os.listdir(PROCESSED_DIR):
    label_path = os.path.join(PROCESSED_DIR, label)
    if os.path.isdir(label_path):
        print(f"{label}: {len(os.listdir(label_path))} clips")