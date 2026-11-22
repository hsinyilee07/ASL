import time
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

lastest_result = None

def on_result(result, output_image, timestamp_ms):
    global lastest_result
    lastest_result = result

def draw_landmarks_on_frame(frame, detection_result):
    if not detection_result or not detection_result.hand_landmarks:
        return frame

    height, width, _ = frame.shape

    for hand_landmarks in detection_result.hand_landmarks:
        # 1. Convert normalized (0.0 - 1.0) coordinates to pixel coordinates
        pixel_points = []
        for lm in hand_landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            pixel_points.append((px, py))

        # 2. Draw the skeleton lines connecting the joints
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (0, 255, 0), 2)

        # 3. Draw a circle on each of the 21 landmarks
        for px, py in pixel_points:
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    return frame

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

# 1. Open the default webcam (0 is built-in camera)
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    # 2. Loop continuously while the camera is active
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect   
        
        # 3. Convert OpenCV's BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 4. Generate a strictly increasing timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # 5. Send the frame to MediaPipe asynchronously
        landmarker.detect_async(mp_image, timestamp_ms)

        # 6. Draw the hand landmarks on the frame
        frame = draw_landmarks_on_frame(frame, lastest_result)

        # 6. Show the camera feed in a window (press 'q' to quit)
        cv2.imshow("MediaPipe Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 7. Clean up camera resources
cap.release()
cv2.destroyAllWindows()
