import os
import csv

# ----------------------------------
# CONFIGURATION
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "splits")

# Label mapping (IMPORTANT)
LABEL_MAP = {
    "NonViolence": 0,
    "Violence": 1
}

# ----------------------------------
# FUNCTION TO CREATE CSV
# ----------------------------------
def generate_csv(split_name):

    split_path = os.path.join(SPLITS_DIR, split_name)
    csv_path = os.path.join(BASE_DIR, f"{split_name}.csv")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        for label in os.listdir(split_path):

            label_path = os.path.join(split_path, label)

            if not os.path.isdir(label_path):
                continue

            label_id = LABEL_MAP[label]

            for clip in os.listdir(label_path):

                clip_path = os.path.join(
                    "splits",
                    split_name,
                    label,
                    clip
                )

                writer.writerow([clip_path, label_id])

    print(f"✅ Created {split_name}.csv")


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        generate_csv(split)

    print("\n🎉 All CSV files generated successfully!")