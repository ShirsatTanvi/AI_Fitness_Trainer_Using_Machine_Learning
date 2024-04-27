import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from AudioCommSys import text_to_speech
from EmailingSystem import email_user
import speech_recognition as sr

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# global total_calories_burned
# global total_workout_time

total_calories_burned = 0
total_workout_time = 0


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def countdown():
    countdown_messages = ["5", "4", "3", "2", "1", "Exercise started!"]
    for message in countdown_messages:
        text_to_speech(message)
        time.sleep(1)

def get_performance_bar_color(per):
    color = (0, 205, 205)
    if 0 < per <= 30:
        color = (51, 51, 255)
    if 30 < per <= 60:
        color = (0, 165, 255)
    if 60 <= per <= 100:
        color = (0, 255, 255)
    return color

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

def draw_performance_bar(img, per, bar, color, count):
    cv2.rectangle(img, (1600, 100), (1675, 650), color, 3)
    cv2.rectangle(img, (1600, int(bar)), (1675, 650), color, cv2.FILLED)
    cv2.putText(
        img, f"{int(per)} %", (1600, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4
    )

def text_to_speech_count(count):
    text_to_speech(f"{count}")

def distanceCalculate(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)
        if angle > 180:
            return 360 - angle
        else:
            return angle
    else:
        return -1

def legState(angle):
    if angle < 0:
        return 0
    elif angle < 105:
        return 1
    elif angle < 150:
        return 2
    else:
        return 3
# def listen_for_stop_command(recognizer):
#     """Listen for voice command to stop exercise."""
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
    
#     try:
#         command = recognizer.recognize_google(audio).lower()
#         if 'stop exercise' in command:
#             return True
#     except sr.UnknownValueError:
#         pass
    
#     return False

def push_ups():
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
        time_elapsed=0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

                if distanceCalculate(rightShoulder, rightWrist) < 130:
                    pushUpStart = 1
                elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) > 250:
                    pushUpCount += 1
                    pushUpStart = 0
                    time_elapsed = int(time.process_time() - start)
                    calories_burned_per_rep = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
                    speaker_thread = threading.Thread(target=text_to_speech_count, args=(pushUpCount,))
                    speaker_thread.start()

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

            # Display calories burned per rep
            if pushUpCount > 0:
                cv2.putText(frame, 'Calories Burned: ' + str(calories_burned_per_rep), (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            frame = cv2.resize(frame, (1280, 820))

            total_calories_burned += calories_burned_per_rep
            total_workout_time +=time_elapsed

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Display the frame
            cv2.imshow('Push Ups', frame)

            total_workout_time +=time_elapsed

            total_calories_burned += calories_burned_per_rep

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def squats():
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
        time_elapsed=0

        while cap.isOpened():
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
                rAngle = findAngle(lm[24], lm[26], lm[28])
                lAngle = findAngle(lm[23], lm[25], lm[27])

                # Stage Detection
                if rAngle > 160:
                    stage = "down"
                if rAngle < 30 and stage == 'down':
                    stage = "up"
                rState = legState(rAngle)
                lState = legState(lAngle)
                state = rState * lState

                if state == 0:  # One or both legs not detected
                    if rState == 0:
                        print("Right Leg Not Detected")
                    if lState == 0:
                        print("Left Leg Not Detected")
                elif state % 2 == 0 or rState != lState:  # One or both legs still transitioning
                    if lastState == 1:
                        if lState == 2 or lState == 1:
                            print("Fully extend left leg")
                        if rState == 2 or lState == 1:
                            print("Fully extend right leg")
                    else:
                        if lState == 2 or lState == 3:
                            print("Fully retract left leg")
                        if rState == 2 or lState == 3:
                            print("Fully retract right leg")
                else:
                    if state == 1 or state == 9:
                        if lastState != state:
                            lastState = state
                            if lastState == 1:
                                # print("GOOD!")
                                
                                counter += 1
                                time_elapsed = int(time.process_time() - start)
                                calories_burned_per_rep = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
                                speaker_thread = threading.Thread(target=text_to_speech_count, args=(counter,))
                                speaker_thread.start()

            except Exception as e:
                print(f"An error occurred: {e}")

            # Render curl counter
            cv2.rectangle(image, (0, 0), (275, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (115, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (110, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            time_elapsed = int(time.process_time() - start)
            cv2.putText(image, 'Time Elapsed: ' + str(time_elapsed) + ' sec', (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if counter > 0:
                calories_burned_per_rep = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
                cv2.putText(image, 'Calories Burned: ' + str(calories_burned_per_rep), (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            image = cv2.resize(image, (1280, 820))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Squats', image)

            total_workout_time +=time_elapsed

            total_calories_burned += calories_burned_per_rep

            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Call your exercise functions here...
    text_to_speech_count("Welcome! Let's start exercising. Choose an exercise: Enter 1 for Bicep Curls or 2 for Push-ups or 3 for Squats.")
    while True:
        exercise = input("Choose an exercise (1 for Bicep Curls, 2 for Push-ups, 3 for squats): ")
        if exercise == '1':
            text_to_speech_count("Starting Bicep Curls exercise.")
            bicep_curls()
            break

        elif exercise == '2':
            advisory_message_thread = threading.Thread(target=text_to_speech_count, args=("Before starting Push-ups exercise, please be advised that push-up exercises may not be suitable for individuals with certain medical conditions, such as heart conditions. It is important to consult with a healthcare professional before engaging in any physical activity, especially if you have any underlying health concerns. Proceed with push-up exercises at your own risk, and stop immediately if you experience any discomfort or adverse symptoms. If you are unsure about whether push-up exercises are appropriate for you, please seek guidance from a qualified healthcare provider.",))
            advisory_message_thread.start()
            advisory_message_thread.join()  # Wait for the advisory message to finish
        
            # Starting push-up exercise statement
            text_to_speech_count("Starting Push-ups exercise.")
            push_ups()
            break

        elif exercise == '3':
            text_to_speech_count("Starting squats exercise.")
            squats()
            break
        else:
            text_to_speech_count("Invalid choice. Please enter 1 or 2.")
    # bicep_curls()
    # push_ups()
    # squats()

    # Email user with the total calories burned and total workout time
    email_user("varunjadhav284@gmail.com", "User", str(total_calories_burned), total_workout_time)

if __name__ == "__main__":
    main()