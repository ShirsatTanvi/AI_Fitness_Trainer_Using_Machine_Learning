import cv2
import mediapipe as mp
import numpy as np
from AudioCommSys import text_to_speech
import time

class Pose:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose            
    
    total_calories_burned = 0
    total_workout_time = 0


    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    @staticmethod
    def countdown():
        countdown_messages = ["5", "4", "3", "2", "1", "Exercise started!"]
        for message in countdown_messages:
            text_to_speech(message)
            time.sleep(1)

    @staticmethod
    def get_performance_bar_color(per):
        color = (0, 205, 205)
        if 0 < per <= 30:
            color = (51, 51, 255)
        if 30 < per <= 60:
            color = (0, 165, 255)
        if 60 <= per <= 100:
            color = (0, 255, 255)
        return color

    @staticmethod
    def position_info_floor_exercise(img, isRightPosition):
        if isRightPosition:
            cv2.putText(
                img,
                "Right Position",
                (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                20,
            )
        else:
            cv2.putText(
                img,
                "Incorrect Position",
                (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                20,
            )

    @staticmethod
    def position_info_standing_exercise(img, isRightPosition):
        if isRightPosition:
            cv2.putText(
                img,
                "Facing Forward",
                (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                20,
            )
        else:
            cv2.putText(
                img,
                "Not Facing Forward",
                (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                20,
            )

    @staticmethod
    def draw_performance_bar(img, per, bar, color, count):
        cv2.rectangle(img, (1600, 100), (1675, 650), color, 3)
        cv2.rectangle(img, (1600, int(bar)), (1675, 650), color, cv2.FILLED)
        cv2.putText(
            img, f"{int(per)} %", (1600, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4
        )

    @staticmethod
    def text_to_speech_count(count):
        text_to_speech(f"{count}")

    @staticmethod
    def distanceCalculate(p1, p2):
        """Calculates the Euclidean distance between two points."""
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
    @staticmethod
    def findAngle(a, b, c, minVis=0.8):
        if hasattr(a, 'visibility') and hasattr(b, 'visibility') and hasattr(c, 'visibility'):
            if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
                bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
                ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
                angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)
                if angle > 180:
                    return 360 - angle
                else:
                    return angle
        return -1


    @staticmethod
    def legState(angle):
        if angle < 0:
            return 0
        elif angle < 105:
            return 1
        elif angle < 150:
            return 2
        else:
            return 3

    # Other methods can be added here as needed...
