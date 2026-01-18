#!/usr/bin/env python3
"""
Scan dataset/<name>/*.jpg, compute face encodings, and save to encodings/encodings.pickle
"""
import os
import pickle
from pathlib import Path
import face_recognition

DATASET_DIR = "dataset"
OUTPUT_DIR = "encodings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "encodings.pickle")

def build_encodings(dataset_dir=DATASET_DIR, output_file=OUTPUT_FILE):
    encodings = []
    names = []

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset directory '{dataset_dir}' not found. Create dataset/<person>/*.jpg")

    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        for img_path in sorted(person_dir.glob("*")):
            try:
                image = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(image, model="hog")
                if len(face_locations) == 0:
                    print(f"[WARN] No face found in {img_path}, skipping.")
                    continue
                # Use first face found in the image
                face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                encodings.append(face_encoding)
                names.append(person_name)
                print(f"[OK] Added encoding for {person_name} from {img_path.name}")
            except Exception as e:
                print(f"[ERR] {img_path}: {e}")

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"Encodings saved to {output_file}")

if __name__ == "__main__":
    build_encodings()
