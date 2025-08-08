import cv2
import numpy as np
import mediapipe as mp

canvas_width, canvas_height = 1280, 720
cam_width, cam_height = 320, 240
x, y = 100, 100
vx, vy = 5, 5

CHROMA_KEY_COLOR = (0, 255, 0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

def resize_with_crop(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    return resized[start_y:start_y + target_height, start_x:start_x + target_width]

def apply_circle_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)

    masked = cv2.bitwise_and(image, image, mask=mask)

    green_background = np.full_like(image, CHROMA_KEY_COLOR)
    circle_output = np.where(mask[..., None] == 255, masked, green_background)
    return circle_output


mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cv2.namedWindow("BouncingCam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("BouncingCam", canvas_width, canvas_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = resize_with_crop(frame, cam_width, cam_height)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmentor.process(rgb_frame)

    mask = results.segmentation_mask > 0.5
    background = np.full_like(frame, CHROMA_KEY_COLOR)
    segmented = np.where(mask[..., None], frame, background)
    output_frame = apply_circle_mask(segmented)


    if x + cam_width >= canvas_width or x <= 0:
        vx = -vx
    if y <= 0 or y + cam_height >= canvas_height:
        vy = -vy
    x += vx
    y += vy

    x = max(0, min(x, canvas_width - cam_width))
    y = max(0, min(y, canvas_height - cam_height))

    canvas = np.full((canvas_height, canvas_width, 3), CHROMA_KEY_COLOR, dtype=np.uint8)
    canvas[y:y+cam_height, x:x+cam_width] = output_frame

    cv2.imshow("BouncingCam", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
