import math
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import mediapipe as mp

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(static_image_mode=True)

def preprocess_image(image):
    image_resized = cv2.resize(image, (192, 192))
    input_type = input_details[0]['dtype']
    image_input = image_resized.astype(np.float32) / 255.0 if input_type != np.uint8 else image_resized.astype(np.uint8)
    return np.expand_dims(image_input, axis=0)

def run_inference(image_np):
    input_data = preprocess_image(image_np)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def calculate_distance(p1, p2, image_width, image_height):
    x1, y1 = int(p1[1] * image_width), int(p1[0] * image_height)
    x2, y2 = int(p2[1] * image_width), int(p2[0] * image_height)
    return np.linalg.norm([x2 - x1, y2 - y1])


def calculate_distance1(p1, p2, image_width, image_height, height_cm=170):
    """
    Calculate the real-world distance (in cm) between two keypoints (normalized) 
    using image dimensions and reference height in cm.
    """
    
    x1, y1 = p1[1] * image_width, p1[0] * image_height
    x2, y2 = p2[1] * image_width, p2[0] * image_height

    
    pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])

    
    pixels_per_cm = image_height / height_cm
    real_distance_cm = pixel_distance / pixels_per_cm

    return real_distance_cm


@app.route('/measurement1', methods=['POST']) 
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    height_cm = float(request.form.get('height_cm', 170))
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    image_height, image_width, _ = image_np.shape

    keypoints = run_inference(image_np)

    def kp(index):
        return keypoints[index][:2]

    nose = kp(0)
    left_ankle = kp(15)
    right_ankle = kp(16)
    left_shoulder = kp(5)
    right_shoulder = kp(6)
    left_hip = kp(11)
    left_elbow = kp(7)
    left_wrist = kp(9)

 
    avg_ankle = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
    pixel_height = calculate_distance(nose, avg_ankle, image_width, image_height)
    pixels_per_cm = pixel_height / height_cm


    shoulder_width_px = calculate_distance(left_shoulder, right_shoulder, image_width, image_height)
    shirt_length_px = calculate_distance(left_shoulder, left_hip, image_width, image_height)
    half_sleeve_px = calculate_distance(left_shoulder, left_elbow, image_width, image_height)
    full_sleeve_px = half_sleeve_px + calculate_distance(left_elbow, left_wrist, image_width, image_height)

    result = {
        "shoulder_width": {
            "pixels": round(shoulder_width_px, 2),
            "cm": round(shoulder_width_px / pixels_per_cm, 2)
        },
        "shirt_length": {
            "pixels": round(shirt_length_px, 2),
            "cm": round(shirt_length_px / pixels_per_cm, 2)
        },
        "half_sleeve_length": {
            "pixels": round(half_sleeve_px, 2),
            "cm": round(half_sleeve_px / pixels_per_cm, 2)
        },
        "full_sleeve_length": {
            "pixels": round(full_sleeve_px, 2),
            "cm": round(full_sleeve_px / pixels_per_cm, 2)
        },
    }

    print(result)
    return jsonify(result)


@app.route('/measurement2', methods=['POST'])
def measure_wrist_distances():
    if 'front_image' not in request.files or 'side_image' not in request.files:
        return jsonify({"error": "Both front and side images are required"}), 400

    try:
        height_cm = float(request.form.get('height_cm', 170))
    except ValueError:
        return jsonify({"error": "Invalid height"}), 400

    def extract_keypoints(image_file):
        image = Image.open(image_file.stream).convert('RGB')
        np_image = np.array(image)
        h, w, _ = np_image.shape
        keypoints = run_inference(np_image)  
        return keypoints, h, w

    def calculate_distance(p1, p2, image_width, image_height, height_cm=170):
        x1, y1 = p1[1] * image_width, p1[0] * image_height
        x2, y2 = p2[1] * image_width, p2[0] * image_height
        pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])
        pixels_per_cm = image_height / height_cm
        return pixel_distance / pixels_per_cm

    try:
        front_kp, front_h, front_w = extract_keypoints(request.files['front_image'])
        side_kp, side_h, side_w = extract_keypoints(request.files['side_image'])

        front_left_wrist = front_kp[9][:2]   
        front_right_wrist = front_kp[10][:2]
        side_left_wrist = side_kp[9][:2]
        side_right_wrist = side_kp[10][:2]

        front_distance_cm = calculate_distance(front_left_wrist, front_right_wrist, front_w, front_h, height_cm)
        side_distance_cm = calculate_distance(side_left_wrist, side_right_wrist, side_w, side_h, height_cm)
        
        print({
            "front_wrist_distance_cm": round(front_distance_cm, 2),
            "side_wrist_distance_cm": round(side_distance_cm, 2)
        })
        return jsonify({
            "front_wrist_distance_cm": round(front_distance_cm, 2),
            "side_wrist_distance_cm": round(side_distance_cm, 2)
        })

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400
    
    

@app.route('/measurement3', methods=['POST'])
def measure_hip_circumference():
    if 'front_image' not in request.files or 'side_image' not in request.files:
        return jsonify({"error": "Both front and side images are required"}), 400

    try:
        height_cm = float(request.form.get('height_cm', 170))
    except ValueError:
        return jsonify({"error": "Invalid height"}), 400

    def extract_keypoints(image_file):
        image = Image.open(image_file.stream).convert('RGB')
        np_image = np.array(image)
        h, w, _ = np_image.shape
        keypoints = run_inference(np_image) 
        return keypoints, h, w

    def calculate_euclidean_distance(p1, p2, image_width, image_height, height_cm):
        x1, y1 = p1[1] * image_width, p1[0] * image_height
        x2, y2 = p2[1] * image_width, p2[0] * image_height
        pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])
        pixels_per_cm = image_height / height_cm
        return pixel_distance / pixels_per_cm

    def calculate_horizontal_distance(p1, p2, image_width, image_height, height_cm):
        x1 = p1[1] * image_width
        x2 = p2[1] * image_width
        pixel_distance = abs(x2 - x1)
        physical_width_cm = height_cm * (image_width / image_height)
        pixels_per_cm = image_width / physical_width_cm
        return pixel_distance / pixels_per_cm

    try:
        front_kp, front_h, front_w = extract_keypoints(request.files['front_image'])
        side_kp, side_h, side_w = extract_keypoints(request.files['side_image'])

        if len(front_kp) < 13 or len(side_kp) < 13:
            return jsonify({"error": "Not enough keypoints detected"}), 400

        front_left_hip = front_kp[11][:2]
        front_right_hip = front_kp[12][:2]
        front_hip_width = calculate_euclidean_distance(front_left_hip, front_right_hip, front_w, front_h, height_cm)

        side_hip = side_kp[12][:2]
        side_back = side_kp[6][:2]
        side_hip_depth = calculate_horizontal_distance(side_hip, side_back, side_w, side_h, height_cm)

        a = front_hip_width / 2
        b = side_hip_depth / 2
        ellipse_circumference = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

        correction_factor = 0.85
        corrected_cm = ellipse_circumference * correction_factor
        corrected_in = corrected_cm / 2.54

        return jsonify({
            "front_hip_width_cm": round(front_hip_width, 2),
            "side_hip_depth_cm": round(side_hip_depth, 2),
            "estimated_hip_circumference_cm": round(corrected_cm, 2),
            "estimated_hip_circumference_in": round(corrected_in, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


    




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
