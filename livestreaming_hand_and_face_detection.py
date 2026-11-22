import time
import cv2
import mediapipe as mp

#1. Load the necessary modules from MediaPipe
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Hand
HandLandmarker = mp.tasks.vision.HandLandmarker    #Model skeleton
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions    #Model configuration options

#Face
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

#2. Create Model Congiguration Options
latest_hand_result = None
latest_face_result = None

def on_hand_result(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result

def on_face_result(result, output_image, timestamp_ms):
    global latest_face_result
    latest_face_result = result

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./landmarker_weights/hand_landmarker.task"),  #where the weights of the model are stored
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_hand_result,
)
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./landmarker_weights/face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=on_face_result,
)

#3. Visualization functions to draw the landmarks on the frame
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FACE_CONNECTIONS = mp.tasks.vision.FaceLandmarksConnections   

def draw_hand_landmarks_on_frame(frame, detection_result):
    if not detection_result or not detection_result.hand_landmarks:
        return frame

    height, width, _ = frame.shape

    for hand_landmarks in detection_result.hand_landmarks:
        # 1. Convert normalized (0.0 - 1.0) coordinates to pixel coordinates
        pixel_points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

        # 2. Draw the skeleton lines connecting the joints
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (0, 255, 0), 2)

        # 3. Draw a circle on each of the 21 landmarks
        for px, py in pixel_points:
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    return frame

def draw_face_landmarks_on_frame(frame, detection_result):
    if not detection_result or not detection_result.face_landmarks:
        return frame

    height, width, _ = frame.shape

    for face_landmarks in detection_result.face_landmarks:
        # 1. Convert normalized (0.0 - 1.0) coordinates to pixel coordinates
        pixel_points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks]
    
        # 2. Draw the skeleton lines connecting the joints
        for connection in FACE_CONNECTIONS.FACE_LANDMARKS_TESSELATION:  #or use FACE_LANDMARKS_CONTOURS
            cv2.line(frame, pixel_points[connection.start], pixel_points[connection.end], (0, 255, 0), 2)

        # 3. Draw a circle on each of the face landmarks
        for px, py in pixel_points:
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

    return frame

def draw_landmarks_on_frame(frame, hand_result, face_result):
    frame = draw_hand_landmarks_on_frame(frame, hand_result)
    frame = draw_face_landmarks_on_frame(frame, face_result)
    return frame   


#4. Main Loop

cap = cv2.VideoCapture(0)  #Open the default webcam (0 is built-in camera)

with HandLandmarker.create_from_options(hand_options) as hand_landmarker:  # create the hand landmarker model
    with FaceLandmarker.create_from_options(face_options) as face_landmarker:  #create the face landmarker model
        
        #Loop continuously while the camera is active
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect   
            
            #Convert OpenCV's BGR image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   #open cv stores frame as numpy array, but mediapipe needs it in mp.Image format

            # Generate timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            #Send the frame to MediaPipe asynchronously
            hand_landmarker.detect_async(mp_image, timestamp_ms)
            face_landmarker.detect_async(mp_image, timestamp_ms)

            # Draw the hand & Face landmarks on the frame
            frame = draw_landmarks_on_frame(frame, latest_hand_result, latest_face_result)

            #Show the camera feed in a window (press 'q' to quit)
            cv2.imshow("MediaPipe Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

#5. Clean up camera resources
cap.release()  #disconnects the camera 
cv2.destroyAllWindows()  #destroys all the windows created by OpenCV
