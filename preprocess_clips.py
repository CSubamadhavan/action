import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level UP from scripts/
PROJECT_DIR = os.path.dirname(BASE_DIR)

INPUT_DIR = os.path.join(PROJECT_DIR, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed")

CLIP_DURATION = 4        # seconds per clip
TARGET_FPS = 30          # expected fps
FRAME_SIZE = (224, 224)  # resize for SlowFast

# -----------------------------
# FUNCTION TO EXTRACT CLIPS
# -----------------------------
def extract_clips_from_video(video_path, label):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print(f"⚠ Skipping {video_path} (fps=0)")
        return

    frames_per_clip = CLIP_DURATION * fps
    clip_index = 0
    start_frame = 0

    print(f"🎥 Processing: {video_name} | FPS: {fps} | Frames: {total_frames}")

    while start_frame + frames_per_clip <= total_frames:

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_folder = os.path.join(
            OUTPUT_DIR,
            label,
            f"{video_name}_clip{clip_index}"
        )
        os.makedirs(clip_folder, exist_ok=True)

        for i in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            frame_path = os.path.join(clip_folder, f"{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)

        clip_index += 1
        start_frame += frames_per_clip  # non-overlapping clips

    cap.release()


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():

    for label in os.listdir(INPUT_DIR):

        label_input_path = os.path.join(INPUT_DIR, label)

        if not os.path.isdir(label_input_path):
            continue

        print(f"\n📂 Processing class: {label}")

        for video_file in os.listdir(label_input_path):

            if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):

                video_path = os.path.join(label_input_path, video_file)
                extract_clips_from_video(video_path, label)

    print("\n✅ Preprocessing Complete!")


if __name__ == "__main__":
    main()