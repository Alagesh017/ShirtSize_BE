import math
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import mediapipe as mp
from chart import measurement_chart

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(image_file):
    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)
    image_height, image_width, _ = image_np.shape
    results = pose_tracker.process(image_np)
    
    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected")
    
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.y, landmark.x))
    
    return keypoints, image_height, image_width

def calculate_distance(p1, p2, image_width, image_height):
    x1, y1 = p1[1] * image_width, p1[0] * image_height
    x2, y2 = p2[1] * image_width, p2[0] * image_height
    return np.linalg.norm([x2 - x1, y2 - y1])

def calculate_real_distance(p1, p2, image_width, image_height, height_cm):
    pixel_distance = calculate_distance(p1, p2, image_width, image_height)
    pixels_per_cm = image_height / height_cm
    return pixel_distance / pixels_per_cm


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle(a, b, c):
    """Returns angle (in degrees) at point b formed by points a-b-c"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_rad = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle_rad)

@app.route('/measurement4', methods=['POST'])
def upload_image3():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        height_cm = float(request.form.get('height_cm', 175))
        image_file = request.files['image']
        keypoints, image_height, image_width = extract_keypoints(image_file)

        def kp(index): return keypoints[index]

        
        nose = kp(0)
        left_shoulder, right_shoulder = kp(11), kp(12)
        left_elbow, right_elbow = kp(13), kp(14)
        left_wrist, right_wrist = kp(15), kp(16)
        left_hip, right_hip = kp(23), kp(24)
        left_ankle, right_ankle = kp(27), kp(28)

        
        left_elbow_angle = angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle(right_shoulder, right_elbow, right_wrist)

        
        def is_on_hip(wrist, hip, elbow_angle):
            dist = euclidean(wrist, hip)
            horizontal_diff = abs(wrist[0] - hip[0])
            vertical_diff = abs(wrist[1] - hip[1])

            dist_ok = dist < 80
            angle_ok = 55 <= elbow_angle <= 90
            horiz_ok = horizontal_diff < 60
            vert_ok = vertical_diff < 100

            confidence_score = sum([dist_ok, angle_ok, horiz_ok, vert_ok]) / 4.0
            return confidence_score >= 0.8

        left_on_hip = is_on_hip(left_wrist, left_hip, left_elbow_angle)
        right_on_hip = is_on_hip(right_wrist, right_hip, right_elbow_angle)
        hip_touch_confirmed = left_on_hip and right_on_hip

    
        top_y = min(nose[1], left_shoulder[1], right_shoulder[1])
        bottom_y = max(left_ankle[1], right_ankle[1])
        pixel_height = bottom_y - top_y

        camera_constant = 700 
        estimated_distance_cm = (camera_constant * height_cm) / pixel_height
        estimated_distance_cm = round(min(max(estimated_distance_cm, 30), 300), 2)  

     
        print({
            "hip_touch_confirmed": hip_touch_confirmed,
            "left_on_hip": left_on_hip,
            "right_on_hip": right_on_hip,
            "left_elbow_angle": round(left_elbow_angle, 2),
            "right_elbow_angle": round(right_elbow_angle, 2),
            "estimated_distance_cm": estimated_distance_cm,
        })

        return jsonify({
            "hip_touch_confirmed": bool(hip_touch_confirmed),
            "left_on_hip": bool(left_on_hip),
            "right_on_hip": bool(right_on_hip),
            "left_elbow_angle": round(float(left_elbow_angle), 2),
            "right_elbow_angle": round(float(right_elbow_angle), 2),
            "estimated_distance_cm": round(float(estimated_distance_cm), 2),
        })



    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
    


@app.route('/measurement3', methods=['POST'])
def validate_side_pose():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        keypoints, _, _ = extract_keypoints(image_file)

        def kp(index): return keypoints[index]

        left_shoulder, right_shoulder = kp(11), kp(12)
        left_hip, right_hip = kp(23), kp(24)
        left_wrist, right_wrist = kp(15), kp(16)

        
        dist_thresh = 80
        horiz_offset_thresh = 30  

        right_front = (
            right_wrist[0] < right_shoulder[0] - horiz_offset_thresh and
            right_wrist[0] < right_hip[0] - horiz_offset_thresh and
            euclidean(right_wrist, right_hip) < dist_thresh
        )

        left_back = (
            left_wrist[0] > left_shoulder[0] + horiz_offset_thresh and
            left_wrist[0] > left_hip[0] + horiz_offset_thresh and
            euclidean(left_wrist, left_hip) < dist_thresh
        )

        pose_valid = right_front and left_back

        print({
            "pose_valid": pose_valid,
            "right_wrist_on_stomach": right_front,
            "left_wrist_on_back": left_back
        })

        return jsonify({
            "pose_valid": bool(pose_valid),
            "right_wrist_on_stomach": bool(right_front),
            "left_wrist_on_back": bool(left_back)
        })

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

    
@app.route('/measurement1', methods=['POST']) 
def upload_image1():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        height_cm = float(request.form.get('height_cm'))
        print(height_cm)
        keypoints, image_height, image_width = extract_keypoints(request.files['image'])

        def kp(index):
            return keypoints[index]

        left_shoulder = kp(11)
        right_shoulder = kp(12)

        
        shoulder_px = calculate_distance(left_shoulder, right_shoulder, image_width, image_height)
        shoulder_cm = calculate_real_distance(left_shoulder, right_shoulder, image_width, image_height, height_cm)+10

        
        neck_px = 0.25 * shoulder_px
        neck_cm = 0.25 * shoulder_cm

        
        


        
        closest_height = min(measurement_chart.keys(), key=lambda x: abs(x - height_cm))
        shirt_range, half_sleeve_range, full_sleeve_range = measurement_chart[closest_height]

       
        shirt_length_cm = sum(shirt_range) / 2
        half_sleeve_cm = sum(half_sleeve_range) / 2
        full_sleeve_cm = sum(full_sleeve_range) / 2

        result = {
            "shoulder_width": {
                "pixels": round(shoulder_px, 2),
                "cm": round(shoulder_cm, 2)
            },
            "neck_width": {
                "pixels": round(neck_px, 2),
                "cm": round(neck_cm, 2),
                "estimated_from_shoulder": True
            },
            "shirt_length": {
                "pixels": None,
                "cm": round(shirt_length_cm, 2),
                "estimated_from_chart": True
            },
            "half_sleeve_length": {
                "pixels": None,
                "cm": round(half_sleeve_cm, 2),
                "estimated_from_chart": True
            },
            "full_sleeve_length": {
                "pixels": None,
                "cm": round(full_sleeve_cm, 2),
                "estimated_from_chart": True
            },
        }

        print(result)
        return jsonify(result)
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500




@app.route('/measurement2', methods=['POST'])
def measure_wrist_distances():
    if 'front_image' not in request.files or 'side_image' not in request.files:
        return jsonify({"error": "Both front and side images are required"}), 400

    try:
        try:
            height_cm = float(request.form.get('height_cm'))
        except ValueError:
            return jsonify({"error": "Invalid height"}), 400
        print(height_cm)
        front_kp, front_h, front_w = extract_keypoints(request.files['front_image'])
        side_kp, side_h, side_w = extract_keypoints(request.files['side_image'])

        front_left_wrist = front_kp[15]
        front_right_wrist = front_kp[16]
        side_left_wrist = side_kp[15]
        side_right_wrist = side_kp[16]

        front_distance_cm = calculate_real_distance(front_left_wrist, front_right_wrist, front_w, front_h, height_cm)
        side_distance_cm = calculate_real_distance(side_left_wrist, side_right_wrist, side_w, side_h, height_cm)
        print({
            "front_wrist_distance_cm": round(front_distance_cm, 2),
            "side_wrist_distance_cm": round(side_distance_cm, 2)
        })
        return jsonify({
            "front_wrist_distance_cm": round(front_distance_cm, 2),
            "side_wrist_distance_cm": round(side_distance_cm, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
