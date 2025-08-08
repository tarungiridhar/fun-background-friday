import cv2
import numpy as np
import mediapipe as mp

# Load helmet image (with transparent visor)
helmet_filter = cv2.imread("helmet 2.png", cv2.IMREAD_UNCHANGED)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL

def overlay_transparent(background, overlay, center, scale=2.5):
    if overlay is None or overlay.shape[2] != 4:
        return background

    oh, ow = overlay.shape[:2]
    new_h = int(oh * scale)
    new_w = int(ow * scale)
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = int(center[0] - new_w / 2)
    y = int(center[1] - new_h / 2)

    # Clamp coordinates to image bounds:
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + new_w, background.shape[1])
    y2 = min(y + new_h, background.shape[0])

    # Calculate region of overlay to use (crop if x or y < 0)
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    alpha_s = overlay_crop[:, :, 3:] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        roi[:, :, c] = (alpha_s[:, :, 0] * overlay_crop[:, :, c] + alpha_l[:, :, 0] * roi[:, :, c])

    background[y1:y2, x1:x2] = roi

    return background

def get_face_center_and_size(face_landmarks, image_shape):
    h, w, _ = image_shape
    points = []
    for connection in FACE_OVAL:
        for idx in connection:
            point = face_landmarks.landmark[idx]
            points.append((int(point.x * w), int(point.y * h)))
    points = np.array(points)

    # Use mean as center and bounding rect to estimate size
    center = points.mean(axis=0).astype(int)
    _, _, width, height = cv2.boundingRect(points)
    return center, width, height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Create green background
    green_background = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            center, face_w, face_h = get_face_center_and_size(face_landmarks, frame.shape)

            # Place original face onto green background (no mesh fill)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, tuple(center), int(max(face_w, face_h) * 0.6), 255, -1)
            mask_3ch = cv2.merge([mask, mask, mask])

            face_area = cv2.bitwise_and(frame, mask_3ch)
            background_area = cv2.bitwise_and(green_background, cv2.bitwise_not(mask_3ch))
            frame = cv2.add(face_area, background_area)

            # Overlay helmet on top
            helmet_center = center.copy()
            helmet_center[1] -= int(face_h * 0.15)  # Move helmet up by 15% of face height
            frame = overlay_transparent(frame, helmet_filter, helmet_center, scale=face_w / helmet_filter.shape[1] * 2.5)


    cv2.imshow("Helmet Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
