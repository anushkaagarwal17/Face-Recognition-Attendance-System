#!/usr/bin/env python3
"""
Run webcam, recognize faces using encodings/encodings.pickle, and save attendance to attendance.csv
Press 'q' to quit.
"""
import cv2
import pickle
import time
import csv
from datetime import datetime
import os
import face_recognition
import numpy as np

ENCODINGS_FILE = "encodings/encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.5  # lower = stricter

def load_known():
    if not os.path.exists(ENCODINGS_FILE):
        raise SystemExit(f"Encodings file not found: {ENCODINGS_FILE}. Run encode_faces.py first.")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def mark_attendance(name, out_file=ATTENDANCE_FILE):
    # Only mark once per run for simplicity
    seen = set()
    if os.path.exists(out_file):
        with open(out_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                seen.add(row["name"])
    if name in seen:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(out_file)
    with open(out_file, "a", newline="") as csvfile:
        fieldnames = ["name", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"name": name, "timestamp": now})
    print(f"[ATTEND] {name} at {now}")

def main():
    known_encodings, known_names = load_known()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (0).")

    print("Starting camera. Press 'q' to quit.")
    process_every_n_frames = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Resize to speed up
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if frame_count % process_every_n_frames == 0:
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            face_names = []

            for encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                if len(distances) > 0:
                    best_idx = np.argmin(distances)
                    if distances[best_idx] <= TOLERANCE:
                        name = known_names[best_idx]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                face_names.append(name)
                if name != "Unknown":
                    mark_attendance(name)

            # Map back to original frame size and draw
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # since we scaled to 0.5
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    
if __name__ == "__main__":
    main()
