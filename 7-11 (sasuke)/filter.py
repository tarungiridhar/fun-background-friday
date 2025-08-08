import cv2
import numpy as np
import mediapipe as mp
import itertools

# Load filter images
left_eye_filter = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
right_eye_filter = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
headband_filter = cv2.imread("sasuke.png", cv2.IMREAD_UNCHANGED)


# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL

FOREHEAD_POINTS = [10, 338, 297, 332, 284]  # from MediaPipe's top forehead region


def getSize(image, face_landmarks, INDEXES):
    h, w, _ = image.shape
    points = [(
        int(face_landmarks.landmark[i[0]].x * w),
        int(face_landmarks.landmark[i[0]].y * h)
    ) for i in INDEXES]
    _, _, width, height = cv2.boundingRect(np.array(points))
    return width, height, np.array(points)

def get_open_status(image, face_landmarks, index_set, threshold):
    _, h, _ = getSize(image, face_landmarks, index_set)
    _, face_h, _ = getSize(image, face_landmarks, FACE_OVAL)
    return 'OPEN' if (h / face_h) * 100 > threshold else 'CLOSE'

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
    overlay_x1 = max(0, -x)  # how much to skip in overlay if x < 0
    overlay_y1 = max(0, -y)  # how much to skip in overlay if y < 0
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # If the overlay is completely out of bounds, return original image
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_status = get_open_status(frame, face_landmarks, LEFT_EYE, threshold=4.5)
            right_status = get_open_status(frame, face_landmarks, RIGHT_EYE, threshold=4.5)

            def center_of_landmarks(indexes):
                _, _, pts = getSize(frame, face_landmarks, indexes)
                return pts.mean(axis=0).astype(int)
            
            def center_of_indexes(landmarks, indexes, image_shape):
                h, w, _ = image_shape
                points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indexes]
                return np.mean(points, axis=0).astype(int)


            if left_status == 'OPEN':
                center = center_of_landmarks(LEFT_EYE)
                frame = overlay_transparent(frame, left_eye_filter, center, scale=0.025)


            if right_status == 'OPEN':
                center = center_of_landmarks(RIGHT_EYE)
                frame = overlay_transparent(frame, right_eye_filter, center, scale=0.025)
            
            forehead_center = center_of_indexes(face_landmarks, FOREHEAD_POINTS, frame.shape)
            forehead_center[0] -= 40
            forehead_center[1] -= 10 
            frame = overlay_transparent(frame, headband_filter, forehead_center, scale=0.3)



    cv2.imshow("Sasuke Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
