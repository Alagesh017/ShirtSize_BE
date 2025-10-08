from flask_sqlalchemy import SQLAlchemy
from app import db

class MeasurementRecord(db.Model):
    __tablename__ = 'measurement_records'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    height_cm = db.Column(db.Float)

    front_wrist_distance_cm = db.Column(db.Float)
    side_wrist_distance_cm = db.Column(db.Float)
    front_wrist_actual = db.Column(db.String(50))
    side_wrist_actual = db.Column(db.String(50))


    shoulder_width = db.Column(db.Float)
    neck_width = db.Column(db.Float)
    shirt_length = db.Column(db.Float)
    half_sleeve_length = db.Column(db.Float)
    full_sleeve_length = db.Column(db.Float)

    shoulder_width_actual = db.Column(db.String(50))
    neck_width_actual = db.Column(db.String(50))
    shirt_length_actual = db.Column(db.String(50))
    half_sleeve_length_actual = db.Column(db.String(50))
    full_sleeve_length_actual = db.Column(db.String(50))

    def __repr__(self):
        return f"<MeasurementRecord {self.name}>"
