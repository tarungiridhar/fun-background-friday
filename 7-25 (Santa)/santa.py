import cv2
import numpy as np
import mediapipe as mp

# Load filters
hat_filter = cv2.imread("santa hat.png", cv2.IMREAD_UNCHANGED)
beard_filter = cv2.imread("santa beard.png", cv2.IMREAD_UNCHANGED)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL

def overlay_transparent(background, overlay, center, scale=1.0):
    if overlay is None or overlay.shape[2] != 4:
        return background

    oh, ow = overlay.shape[:2]
    new_w = int(ow * scale)
    new_h = int(oh * scale)
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = int(center[0] - new_w / 2)
    y = int(center[1] - new_h / 2)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + new_w), min(background.shape[0], y + new_h)

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    mask = overlay_crop[:, :, 3:] / 255.0
    inv_mask = 1.0 - mask

    roi = background[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = (mask[:, :, 0] * overlay_crop[:, :, c] + inv_mask[:, :, 0] * roi[:, :, c])
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

    # Use plain green screen background
    background = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            center, face_w, face_h = get_face_center_and_size(face_landmarks, frame.shape)

            # Mask face from webcam to composite on background
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, tuple(center), int(max(face_w, face_h) * 0.45), 255, -1)
            mask_3ch = cv2.merge([mask, mask, mask])
            face_cutout = cv2.bitwise_and(frame, mask_3ch)
            background = cv2.bitwise_and(background, cv2.bitwise_not(mask_3ch))
            composed = cv2.add(face_cutout, background)

            # Hat: move up above center
            hat_center = center.copy()
            hat_center[1] -= int(face_h * 0.45)
            hat_center[0] += int(face_w * 0.3)
            composed = overlay_transparent(composed, hat_filter, hat_center, scale=face_w / hat_filter.shape[1] * 2.5)

            # Beard: move down below center
            beard_center = center.copy()
            beard_center[1] += int(face_h * 0.1)
            composed = overlay_transparent(composed, beard_filter, beard_center, scale=face_w / beard_filter.shape[1] * 1.2)

            frame = composed

    else:
        frame = background  # fallback to just background if no face

    cv2.imshow("Santa Face Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
