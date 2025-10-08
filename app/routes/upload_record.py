from flask import Blueprint, request, jsonify
from app.model import db, MeasurementRecord

upload_bp = Blueprint("upload", __name__, url_prefix="/upload")

@upload_bp.route('/record', methods=['POST'])
def upload_record():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        name = data.get("name")
        height_cm = data.get("height_cm")
        front_wrist_distance_cm = data.get("front_wrist_distance_cm")
        side_wrist_distance_cm = data.get("side_wrist_distance_cm")
        front_wrist_actual = data.get("front_wrist_actual")
        side_wrist_actual = data.get("side_wrist_actual")

        measurements = data.get("measurements", {})
        actual_values = data.get("actual_values", {})

        record = MeasurementRecord(
            name=name,
            height_cm=height_cm,
            front_wrist_distance_cm=front_wrist_distance_cm,
            side_wrist_distance_cm=side_wrist_distance_cm,
            front_wrist_actual=front_wrist_actual,
            side_wrist_actual=side_wrist_actual,

            shoulder_width=measurements.get("shoulder_width"),
            neck_width=measurements.get("neck_width"),
            shirt_length=measurements.get("shirt_length"),
            half_sleeve_length=measurements.get("half_sleeve_length"),
            full_sleeve_length=measurements.get("full_sleeve_length"),

            shoulder_width_actual=actual_values.get("shoulder_width"),
            neck_width_actual=actual_values.get("neck_width"),
            shirt_length_actual=actual_values.get("shirt_length"),
            half_sleeve_length_actual=actual_values.get("half_sleeve_length"),
            full_sleeve_length_actual=actual_values.get("full_sleeve_length"),
        )

        db.session.add(record)
        db.session.commit()

        return jsonify({"message": "Record uploaded successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@upload_bp.route('/records', methods=['GET'])
def get_records():
    try:
        records = MeasurementRecord.query.all()
        results = []
        for record in records:
            results.append({
                "id": record.id,
                "name": record.name,
                "height_cm": record.height_cm,
                "front_wrist_distance_cm": record.front_wrist_distance_cm,
                "side_wrist_distance_cm": record.side_wrist_distance_cm,
                "front_wrist_actual": record.front_wrist_actual,
                "side_wrist_actual": record.side_wrist_actual,
                "measurements": {
                    "shoulder_width": record.shoulder_width,
                    "neck_width": record.neck_width,
                    "shirt_length": record.shirt_length,
                    "half_sleeve_length": record.half_sleeve_length,
                    "full_sleeve_length": record.full_sleeve_length
                },
                "actual_values": {
                    "shoulder_width": record.shoulder_width_actual,
                    "neck_width": record.neck_width_actual,
                    "shirt_length": record.shirt_length_actual,
                    "half_sleeve_length": record.half_sleeve_length_actual,
                    "full_sleeve_length": record.full_sleeve_length_actual
                }
            })
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

