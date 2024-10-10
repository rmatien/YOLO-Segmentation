import time
import cv2
import torch
from ultralytics import YOLO

# torch.cuda.set_device(0)
model = YOLO('D:/Documents/PythonProjects/CobaYOLO/runs/segment/platnomor-seg3/weights/best.pt')

video = 'F:/cctv-doublew-in/datadir0/hiv00017.mp4'

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: Video is not ready")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to retrieve frame.")
        break

    start = time.perf_counter()
    results = model(frame)
    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time

    anotate_frame = results[0].plot()

    # Show the frame with detections
    cv2.putText(anotate_frame, f"FPS: {int(fps)}", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv11 Segmentation Detection", anotate_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
