import cv2
import numpy as np
import pygetwindow as gw
import mss
import time

def get_edge_window():
    windows = gw.getWindowsWithTitle('Edge')
    if not windows:
        raise Exception('Edge browser window not found!')
    return windows[0]

def capture_window(window):
    with mss.mss() as sct:
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
            "mon": 1
        }
        while True:
            img = sct.grab(monitor)
            img_np = np.array(img)
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Edge Browser Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add a small delay to prevent high CPU usage
            time.sleep(0.01)

        cv2.destroyAllWindows()

def main():
    try:
        edge_window = get_edge_window()
        capture_window(edge_window)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
