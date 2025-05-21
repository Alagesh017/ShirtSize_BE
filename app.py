import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_size = 192
height_cm = 170  # Real-world height of the person (adjust if needed)

# MoveNet keypoint index mapping
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def preprocess(image):
    img = cv2.resize(image, (input_size, input_size))
    input_type = input_details[0]['dtype']

    if input_type == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    img = np.expand_dims(img, axis=0)
    return img

def detect_pose(frame):
    input_image = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return keypoints

def get_coords(keypoint, frame_width, frame_height):
    y, x, confidence = keypoint
    return int(x * frame_width), int(y * frame_height)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = detect_pose(frame)
    h, w, _ = frame.shape
    landmark_points = {}

    # Draw and label all keypoints
    for name, idx in KEYPOINT_DICT.items():
        x, y = get_coords(keypoints[idx], w, h)
        landmark_points[name] = (x, y)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("MoveNet Pose", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # Press 's' to calculate
        try:
            shoulder_px = calculate_distance(landmark_points['left_shoulder'], landmark_points['right_shoulder'])
            hip_px = calculate_distance(landmark_points['left_hip'], landmark_points['right_hip'])
            arm_px = (
                calculate_distance(landmark_points['left_shoulder'], landmark_points['left_elbow']) +
                calculate_distance(landmark_points['left_elbow'], landmark_points['left_wrist'])
            )
            leg_px = (
                calculate_distance(landmark_points['left_hip'], landmark_points['left_knee']) +
                calculate_distance(landmark_points['left_knee'], landmark_points['left_ankle'])
            )
            height_px = calculate_distance(landmark_points['left_shoulder'], landmark_points['left_ankle'])

            pixels_per_cm = height_px / height_cm

            print("\n--- Measurements ---")
            print(f"Shoulder Width: {shoulder_px:.2f} px ({shoulder_px/pixels_per_cm:.2f} cm)")
            print(f"Hip Width: {hip_px:.2f} px ({hip_px/pixels_per_cm:.2f} cm)")
            print(f"Left Arm Length: {arm_px:.2f} px ({arm_px/pixels_per_cm:.2f} cm)")
            print(f"Left Leg Length: {leg_px:.2f} px ({leg_px/pixels_per_cm:.2f} cm)")
            print(f"Body Height (Shoulder to Ankle): {height_px:.2f} px ({height_px/pixels_per_cm:.2f} cm)")
            print(f"Pixels per cm: {pixels_per_cm:.4f}")
        except Exception as e:
            print("Measurement failed. Some landmarks may be missing or misdetected.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
