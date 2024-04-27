
from ExercisesModule import Exercises
from PoseModule import Pose
from flask import Flask, render_template, request, redirect, url_for, Response
from flask import session
import cv2
from db import configure_db






app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'AI2024SSPM-AI-FItness_app'


mysql = configure_db(app)




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        userDetails = request.form
        username = userDetails['username']
        email = userDetails['email']
        password = userDetails['password']
        age = userDetails['age']  # Get age from form data
        weight = userDetails['weight']  # Get weight from form data
        gender = userDetails['gender']  # Get gender from form data
        
        cur = mysql.connection.cursor()
        
        try:
            cur.execute("INSERT INTO users(username, email, password, age, weight, gender) VALUES(%s, %s, %s, %s, %s, %s)", (username, email, password, age, weight, gender))
            mysql.connection.commit()
            cur.close()
            # return 'Registration successful!'
            return redirect(url_for('login'))
        except Exception as e:
            return f'Registration failed: {str(e)}'
    else:
        return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userDetails = request.form
        identifier = userDetails['identifier']  # Field to enter either username or email
        password = userDetails['password']
        
        cur = mysql.connection.cursor()
        
        try:
            # Check if the identifier matches either username or email
            result = cur.execute("SELECT * FROM users WHERE (username = %s OR email = %s) AND password = %s", (identifier, identifier, password))
            if result > 0:
                # Retrieve the user's username from the database
                user = cur.fetchone()
                username = user['username']
                
                # Store the username in the session
                session['username'] = username
                return redirect(url_for('dashboard'))
                
            else:
                # Display error message in the placeholder span
                return render_template('index.html', login_error="Login failed. Invalid username or password.", show_alert=True)
        except Exception as e:
            return f'Login failed: {str(e)}'
    else:
        return render_template('index.html')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videoframe')
def videoframe():
    if 'username' in session:
        # User is logged in, continue with dashboard logic
        user_details = get_user_details()  # Implement this function according to your database structure
    
        
        return render_template('videoframe.html', user_details=user_details)
    else:
        # User is not logged in, redirect to login page
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session upon logout
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        # User is logged in, continue with dashboard logic
        user_details = get_user_details()  # Implement this function according to your database structure
    
        return render_template('dashboard.html', user_details=user_details)
        # return render_template('dashboard.html')
    else:
        # User is not logged in, redirect to login page
        return redirect(url_for('login'))

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Open webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Open webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         else:
#             frame = cv2.resize(frame, (1280, 820))
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_exercise/<exercise>')
def start_exercise(exercise):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Set the selected exercise in session
    session['selected_exercise'] = exercise
    
    # Redirect to the exercise difficulty selection page
    return redirect(url_for('exercise_difficulty'))



@app.route('/start_selected_exercise_ajax')
def start_selected_exercise_ajax():
    if 'username' not in session:
        return 'Unauthorized', 401

    # Get the selected exercise from the request parameters
    exercise = request.args.get('exercise')
    
    # Set the difficulty level and start the selected exercise
    difficulty = request.args.get('difficulty')
    if exercise == 'bicep_curls':
        
        Pose.text_to_speech_count("Starting Bicep Curls exercise in.")
        exercises = Exercises(cap=None, difficulty=difficulty)
        exercises.bicep_curls()
    elif exercise == 'push_ups':
        Pose.text_to_speech_count("Starting Push Ups exercise.")
        exercises = Exercises(cap=None, difficulty=difficulty)
        exercises.push_ups()
    elif exercise == 'squats':
        Pose.text_to_speech_count("Starting Squats exercise.")
        exercises = Exercises(cap=None, difficulty=difficulty)
        exercises.squats()
    elif exercise == 'pull_ups':
        Pose.text_to_speech_count("Starting pull ups exercise.")
        exercises = Exercises(cap=None, difficulty=difficulty)
        exercises.pull_ups()
    elif exercise == 'plank':
        Pose.text_to_speech_count("Starting plank exercise.")
        exercises = Exercises(cap=None, difficulty=difficulty)
        exercises.plank()

    return 'Exercise started successfully'



def get_user_details():
    try:
        # Get the username from the session
        username = session.get('username')

        if username:
            cur = mysql.connection.cursor()
            # Assuming you already have a database cursor named 'cur' from the current session
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()  # Fetch the first row

            if user:
                # Extract user details from the fetched row
                username = user['username']
                email = user['email']
                age = user['age']
                weight = user['weight']
                gender = user['gender']

                # Return user details as a dictionary
                return {
                    'username': username,
                    'email': email,
                    'age': age,
                    'weight': weight,
                    'gender': gender
                }
            else:
                return None  # User not found
        else:
            return None  # No active session or username not found in session
    except Exception as e:
        # Handle exceptions (e.g., database connection error)
        print(f"Error fetching user details: {e}")
        return None



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
