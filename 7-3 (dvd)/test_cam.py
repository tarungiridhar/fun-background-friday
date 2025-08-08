import cv2

print("scanning: ")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
        cap.release()
    else:
        print(f"Camera {i} is not available.")