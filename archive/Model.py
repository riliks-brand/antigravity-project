import cv2
import numpy as np
import pygetwindow as gw
import mss
import time
import queue
import threading
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

def get_edge_window():
    windows = gw.getWindowsWithTitle('Edge')
    if not windows:
        raise Exception('Edge browser window not found!')
    return windows[0]

def process_frame(frame, model):
    results = model(frame)
    return results[0].plot()

def frame_capture(monitor, frame_queue, stop_event,capture_interval=1/60):
    with mss.mss() as sct:
        while not stop_event.is_set():
            img = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
            if not frame_queue.full():
                frame_queue.put(frame)

def frame_processing(model, frame_queue, annotated_queue, stop_event):
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            annotated_frame = process_frame(frame, model)
            if not annotated_queue.full():
                annotated_queue.put(annotated_frame)
        except queue.Empty:
            pass

def display_frames(annotated_queue, stop_event):
    while not stop_event.is_set() or not annotated_queue.empty():
        try:
            annotated_frame = annotated_queue.get(timeout=1)
            cv2.imshow('Edge Browser Stream with YOLOv8 Detections', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            pass

def capture_window(window):
    model = YOLO('D:\Candlestick_Detection\custom_yolov8.pt')
    model.conf = 0.10
    model.iou = 0.5
    model.agnostic_nms = False
    model.max_det = 1000
    print(model.device)
    monitor = {
        "top": window.top,
        "left": window.left,
        "width": window.width,
        "height": window.height,
        "mon": 2
    }

    frame_queue = queue.Queue(maxsize=1)
    annotated_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    capture_thread = threading.Thread(target=frame_capture, args=(monitor, frame_queue, stop_event))
    display_thread = threading.Thread(target=display_frames, args=(annotated_queue, stop_event))

    with ThreadPoolExecutor(max_workers=6) as executor:
        capture_thread.start()
        display_thread.start()

        for _ in range(1):
            executor.submit(frame_processing, model, frame_queue, annotated_queue, stop_event)

        capture_thread.join()
        display_thread.join()

    cv2.destroyAllWindows()

def main():
    try:
        edge_window = get_edge_window()
        capture_window(edge_window)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
