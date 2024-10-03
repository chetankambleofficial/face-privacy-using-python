import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import queue

# Initialize the video capture
capture = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for selected effect and the frame queue
selected_effect = 'blur'
frame_queue = queue.Queue(maxsize=10)

# Function to apply pixelation effect
def apply_pixelation(face_region, pixel_size=10):
    h, w, _ = face_region.shape
    small = cv2.resize(face_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

# Function to process frames
def process_frame():
    global selected_effect
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            if selected_effect == 'blur':
                blurred_face = cv2.GaussianBlur(face_region, (21, 21), 0)
                frame[y:y + h, x:x + w] = blurred_face
            elif selected_effect == 'pixelate':
                pixelated_face = apply_pixelation(face_region, pixel_size=10)
                frame[y:y + h, x:x + w] = pixelated_face

        cv2.imshow('Face Detection & Effects', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Video capture thread function
def video_capture_thread():
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        if not frame_queue.full():
            frame_queue.put(frame)

# GUI function
def open_gui():
    global selected_effect
    root = tk.Tk()
    root.title("Face Blurring & Effects App")

    effects_frame = ttk.LabelFrame(root, text="Effects", padding="10")
    effects_frame.grid(row=0, column=0, padx=10, pady=10)

    blur_button = ttk.Button(effects_frame, text="Blur", command=lambda: apply_effect('blur'))
    blur_button.grid(row=0, column=0, padx=5, pady=5)

    pixelate_button = ttk.Button(effects_frame, text="Pixelate", command=lambda: apply_effect('pixelate'))
    pixelate_button.grid(row=1, column=0, padx=5, pady=5)

    root.mainloop()

def apply_effect(effect):
    global selected_effect
    selected_effect = effect

# Start threads
capture_thread = threading.Thread(target=video_capture_thread)
process_thread = threading.Thread(target=process_frame)
gui_thread = threading.Thread(target=open_gui)

capture_thread.start()
process_thread.start()
gui_thread.start()

capture_thread.join()
process_thread.join()
gui_thread.join()

capture.release()
cv2.destroyAllWindows()

