import cv2
import numpy as np
import mediapipe as mp

# Load filter
spaceman_filter = cv2.imread("spaceman transparent.png", cv2.IMREAD_UNCHANGED)


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
            # Extract the masked face cutout
            face_only = cv2.bitwise_and(frame, mask_3ch)
            x, y, w, h = cv2.boundingRect(mask)
            face_roi = face_only[y:y+h, x:x+w]

            # Crop and resize the corresponding mask
            shrink_factor = 0.3
            mask_roi = mask[y:y+h, x:x+w]
            mask_roi_resized = cv2.resize(mask_roi, (0, 0), fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)
            face_roi_resized = cv2.resize(face_roi, (mask_roi_resized.shape[1], mask_roi_resized.shape[0]), interpolation=cv2.INTER_AREA)

            # Create 3-channel mask
            mask_roi_resized_3ch = cv2.merge([mask_roi_resized]*3)
            mask_roi_resized_3ch = mask_roi_resized_3ch.astype(float) / 255.0
            inv_mask = 1.0 - mask_roi_resized_3ch

            # Compute paste position
            paste_x = x + (w - face_roi_resized.shape[1]) // 2
            paste_y = y + (h - face_roi_resized.shape[0]) // 2

            # Blend face into background using mask
            roi = background[paste_y:paste_y+face_roi_resized.shape[0], paste_x:paste_x+face_roi_resized.shape[1]]
            blended = (face_roi_resized.astype(float) * mask_roi_resized_3ch + roi.astype(float) * inv_mask).astype(np.uint8)
            background[paste_y:paste_y+face_roi_resized.shape[0], paste_x:paste_x+face_roi_resized.shape[1]] = blended

            composed = background



            # Determine scale based on face width
            scale_factor = face_w / spaceman_filter.shape[1] * 1.6  # Adjust as needed for better alignment

            # Adjust position so helmet aligns with face
            suit_center = center.copy()
            suit_center[1] += int(face_h * 0.73)  # move filter up so helmet aligns with face

            # Overlay the full spaceman suit
            composed = overlay_transparent(composed, spaceman_filter, suit_center, scale=scale_factor)


            frame = composed

    else:
        frame = background  # fallback to just background if no face

    cv2.imshow("Spaceman Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
