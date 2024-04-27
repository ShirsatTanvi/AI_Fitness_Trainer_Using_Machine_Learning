import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from AudioCommSys import text_to_speech
from PoseModule import Pose
from db import configure_db
from flask import Flask
from EmailingSystem import email_user

from flask import session
app = Flask(__name__)
mysql = configure_db(app)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

total_calories_burned = 0
total_workout_time = 0

import math
import numpy as np

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    
    # Get magnitudes
    magA = np.dot(vA, vA) ** 0.5
    magB = np.dot(vB, vB) ** 0.5
    
    # Get cosine value
    cos_ = dot_prod / magA / magB
    
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod / magB / magA)
    
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360
    ang_deg = 180 - ang_deg
    
    if ang_deg - 180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else:
        return ang_deg


mp_holistic = mp.solutions.holistic
pose_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
pose_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)



def countdown():
    countdown_messages = ["5", "4", "3", "2", "1", "Exercise started!"]
    for message in countdown_messages:
        text_to_speech(message)
        time.sleep(1)

def text_to_speech_count(count):
    text_to_speech("Rep number " + str(count))

def send_mail(total_calories_burned, total_workout_time, overall_feedback):
    """Send an email to the user with workout summary and overall feedback."""
    if 'username' in session:
        username = session['username']
        cur = mysql.connection.cursor()
        cur.execute("SELECT email, username FROM users WHERE username = %s", (username,))
        user_info = cur.fetchone()
        cur.close()
        
        if user_info:
            email_user(user_info['email'], user_info['username'], str(total_calories_burned), total_workout_time, overall_feedback)


class Exercises:
    def __init__(self, cap=None, difficulty=1):
        self.cap = cap
        self.pose = Pose()
        self.set_difficulty(difficulty)
        
    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if self.difficulty == '1':
            self.reps = 10
        elif self.difficulty == '2':
            self.reps = 15
        elif self.difficulty == '3':
            self.reps = 20
        

    def push_ups(self):
        global total_calories_burned, total_workout_time
        cap = cv2.VideoCapture(0)  # Open webcam
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()
        countdown_thread.join()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pushUpStart = 0
            pushUpCount = 0
            start = time.process_time()
            calories_burned_per_rep = 0 
            time_elapsed = 0

            good_reps = 0  # Counter for good reps
            bad_reps = 0   # Counter for bad reps

            while pushUpCount < self.reps:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to capture frame.")
                    break

                image_height, image_width, _ = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(image_rgb)

                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark
                    nosePoint = (int(landmarks[0].x * image_width), int(landmarks[0].y * image_height))
                    leftWrist = (int(landmarks[15].x * image_width), int(landmarks[15].y * image_height))
                    rightWrist = (int(landmarks[16].x * image_width), int(landmarks[16].y * image_height))
                    leftShoulder = (int(landmarks[11].x * image_width), int(landmarks[11].y * image_height))
                    rightShoulder = (int(landmarks[12].x * image_width), int(landmarks[12].y * image_height))
                    leftElbow = (int(landmarks[13].x * image_width), int(landmarks[13].y * image_height))
                    rightElbow = (int(landmarks[14].x * image_width), int(landmarks[14].y * image_height))
                    leftHip = (int(landmarks[23].x * image_width), int(landmarks[23].y * image_height))
                    rightHip = (int(landmarks[24].x * image_width), int(landmarks[24].y * image_height))
                    leftKnee = (int(landmarks[25].x * image_width), int(landmarks[25].y * image_height))

                    per = 0  # Placeholder for per variable, assuming it represents the completion percentage

                    # Calculate angles for form feedback
                    elbow_angle = self.pose.findAngle(leftShoulder, leftElbow, leftWrist)
                    shoulder_angle = self.pose.findAngle(leftHip, leftShoulder, rightShoulder)
                    hip_angle = self.pose.findAngle(leftShoulder, leftHip, leftKnee)

                    form = 0

                    # Check for proper form
                    if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                        form = 1

                    # Check for full range of motion for the push-up
                    if form == 1:
                        if per == 0:
                            if elbow_angle <= 90 and hip_angle > 160:
                                feedback = "Up"
                                if direction == 0:
                                    pushUpCount += 0.5
                                    direction = 1
                            else:
                                feedback = "Fix Form"

                        if per == 100:
                            if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                                feedback = "Down"
                                if direction == 1:
                                    pushUpCount += 0.5
                                    direction = 0
                            else:
                                feedback = "Fix Form"

                    # Adjust feedback based on angles
                    if form == 1:
                        feedback = "Keep Going"
                    else:
                        feedback = "Correct your form for better results"

                    # Update good and bad reps counters
                    if feedback == "Fix Form":
                        bad_reps += 1
                    else:
                        good_reps += 1

                # Draw counter rectangle
                cv2.rectangle(frame, (0, 0), (275, 73), (245, 117, 16), -1)
                # Write 'REPS' label
                cv2.putText(frame, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # Write push-up count
                cv2.putText(frame, str(pushUpCount), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                # Write 'STAGE' label
                cv2.putText(frame, 'STAGE', (115, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # Write push-up stage
                cv2.putText(frame, 'UP' if pushUpStart else 'DOWN', (110, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Display time elapsed
                time_elapsed = int(time.process_time() - start)
                cv2.putText(frame, 'Time Elapsed: ' + str(time_elapsed) + ' sec', (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Display feedback
                # Calculate the size of the text
                (text_width, text_height), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw a filled white rectangle as the background for the text
                cv2.rectangle(frame, (15, 160 - text_height - 5), (15 + text_width + 5, 160), (255, 255, 255), -1)
                
                # Write the text in red on top of the white background
                cv2.putText(frame, feedback, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # Display calories burned per rep
                if pushUpCount > 0:
                    cv2.putText(frame, 'Calories Burned: ' + str(calories_burned_per_rep), (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.resize(frame, (1280, 820))

                total_calories_burned += calories_burned_per_rep
                total_workout_time += time_elapsed

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Display the frame
                cv2.imshow('Push Ups', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Analyze performance based on good and bad reps
            if good_reps > bad_reps:
                overall_feedback = "Your Push ups form is good. Keep it up!"
            elif good_reps < bad_reps:
                 overall_feedback = "Pay attention to your push ups form. Focus on lifting and lowering your arm properly."
            else:
                overall_feedback = "You're making progress, but there's room for improvement in your Push ups form."
    
            # Send email with calories burned, time elapsed, and overall feedback
            send_mail(calories_burned_per_rep, time_elapsed, overall_feedback)
    
            # Release resources
            cap.release()
            cv2.destroyAllWindows()


    def squats(self):
        global total_calories_burned, total_workout_time
        cap = cv2.VideoCapture(0)  # Open webcam
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()
        countdown_thread.join()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            counter = 0
            lastState = 9
            stage = None
            start = time.process_time()
            calories_burned_per_rep = 0 
            time_elapsed = 0

            good_reps = 0  # Counter for good reps
            bad_reps = 0   # Counter for bad reps

            bend_down_flag = False  # Flag to indicate if the user has bent down at least once

            while counter < self.reps:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame.")
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    lm = results.pose_landmarks.landmark
                    rAngle = self.pose.findAngle(lm[24], lm[26], lm[28])
                    lAngle = self.pose.findAngle(lm[23], lm[25], lm[27])

                    # Stage Detection
                    if rAngle > 160:
                        stage = "down"
                    if rAngle < 30 and stage == 'down':
                        stage = "up"
                    rState = self.pose.legState(rAngle)
                    lState = self.pose.legState(lAngle)
                    state = rState * lState

                    feedback = ""  # Initialize feedback string

                    if not bend_down_flag and stage == "down":
                        bend_down_flag = True
                        feedback += "Feedback: You've bent down. "  # Show feedback only after the first bend down
                    else:
                        if state == 0:  # One or both legs not detected
                            if rState == 0:
                                feedback += "Right Leg Not Detected. "
                            if lState == 0:
                                feedback += "Left Leg Not Detected. "
                        elif state % 2 == 0 or rState != lState:  # One or both legs still transitioning
                            if lastState == 1:
                                if lState == 2 or lState == 1:
                                    feedback += "Fully extend left leg. "
                                if rState == 2 or lState == 1:
                                    feedback += "Fully extend right leg. "
                            else:
                                if lState == 2 or lState == 3:
                                    feedback += "Fully retract left leg. "
                                if rState == 2 or lState == 3:
                                    feedback += "Fully retract right leg. "
                        else:
                            if state == 1 or state == 9:
                                if lastState != state:
                                    lastState = state
                                    if lastState == 1:
                                        counter += 1
                                        time_elapsed = int(time.process_time() - start)
                                        calories_burned_per_rep = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
                                        speaker_thread = threading.Thread(target=text_to_speech_count, args=(counter,))
                                        speaker_thread.start()
                                        good_reps += 1  # Increment good reps counter

                    if rAngle < 20:
                        feedback = "Feedback: Bend forward. "
                        bad_reps += 1  # Increment bad reps counter
                    elif rAngle > 45:
                        feedback = "Feedback: Bend backward. "
                        bad_reps += 1  # Increment bad reps counter

                    if 50 <= lAngle <= 80:
                        feedback = "Feedback: Lower your hips. "
                        bad_reps += 1  # Increment bad reps counter
                    elif lAngle > 95:
                        feedback = "Feedback: Squat too deep. "
                        bad_reps += 1  # Increment bad reps counter

                    # Render feedback
                    # Calculate the size of the text
                    (text_width, text_height), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Draw a filled white rectangle as the background for the text
                    cv2.rectangle(image, (15, 160 - text_height - 5), (15 + text_width + 5, 160), (255, 255, 255), -1)
                    
                    # Write the text in red on top of the white background
                    cv2.putText(image, feedback, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"An error occurred: {e}")

                # Render counter
                cv2.rectangle(image, (0, 0), (275, 73), (245, 117, 16), -1)
                cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (115, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(stage), (110, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                time_elapsed = int(time.process_time() - start)
                cv2.putText(image, 'Time Elapsed: ' + str(time_elapsed) + ' sec', (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

                if counter > 0:
                    calories_burned_per_rep = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
                    cv2.putText(image, 'Calories Burned: ' + str(calories_burned_per_rep), (15, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.resize(image, (1280, 820))

                total_calories_burned += calories_burned_per_rep
                total_workout_time += time_elapsed

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Display the frame
                cv2.imshow('Squats', frame)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Analyze performance based on good and bad reps
            if good_reps > bad_reps:
                overall_feedback = "Your Squats form is good. Keep it up!"
            elif good_reps < bad_reps:
                 overall_feedback = "Pay attention to your Squats form. Focus on lifting and lowering properly."
            else:
                overall_feedback = "You're making progress, but there's room for improvement in your squats form."

            # Send email with calories burned, time elapsed, and overall feedback
            send_mail(calories_burned_per_rep, time_elapsed, overall_feedback)
            cap.release()
            cv2.destroyAllWindows()
