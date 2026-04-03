import cv2
import numpy as np
import pygetwindow as gw
import mss
import time
from concurrent.futures import ThreadPoolExecutor
from openvino.runtime import Core

def get_edge_window():
    windows = gw.getWindowsWithTitle('Edge')
    if not windows:
        raise Exception('Edge browser window not found!')
    return windows[0]

def process_frame(frame, compiled_model, input_layer, output_layer):
    # Resize the frame to the model's expected input size
    resized_frame = cv2.resize(frame, (640, 640))
    input_data = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)
    # Run inference
    results = compiled_model([input_data])[output_layer]
    # Debugging: Print the shape of the results
    print(f"Results shape: {results.shape}")
    annotated_frame = draw_detections(resized_frame, results)
    return annotated_frame

def draw_detections(frame, results):
    for detection in results[0][0]:
        conf = detection[2]
        if conf > 0.10:  # Confidence threshold
            xmin, ymin, xmax, ymax = map(int, detection[3:7])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'{int(detection[1])}: {conf:.2f}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def capture_window(window):
    core = Core()
    model = core.read_model('custom_yolov8.onnx')
    compiled_model = core.compile_model(model, 'GPU')
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    with mss.mss() as sct:
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
            "mon": 1
        }

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            while True:
                start_time = time.time()

                img = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)

                if future:
                    annotated_frame = future.result()
                    cv2.imshow('Edge Browser Stream with YOLOv8 Detections', annotated_frame)

                future = executor.submit(process_frame, frame, compiled_model, input_layer, output_layer)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print(f"FPS: {1 / (time.time() - start_time):.2f}")

        cv2.destroyAllWindows()

def main():
    try:
        edge_window = get_edge_window()
        capture_window(edge_window)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
