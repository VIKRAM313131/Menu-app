from flask import Flask, request, render_template, Response, url_for, redirect,send_file,jsonify,send_from_directory
import pywhatkit as kit
import pyautogui as py
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
from serpapi import GoogleSearch
import pyttsx3

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import requests
import base64

import pythoncom
import threading
import cv2
import numpy as np


from werkzeug.utils import secure_filename
import os
from io import BytesIO

from PIL import Image

import io
#Flask App Initialization 
app = Flask(__name__)







##################################################      Function to send Whatsapp message     ##############################################
def send_whatsapp_message(phone_number, message, hour, minute):
    try:
        kit.sendwhatmsg(phone_number, message, hour, minute)
        return f"Message scheduled to be sent to {phone_number} at {hour:02d}:{minute:02d}"
    except Exception as e:
        return str(e)





################################################      Function To SEnd SMS By connecting Phone     #######################################
def send_message_via_phone_link(phone, msg): 
    try:
        # Open the Phone Link app
        py.press('win')
        time.sleep(3)
        py.typewrite("Phone Link")
        time.sleep(5)
        py.press('enter')
        time.sleep(8)  # Adjust based on system performance

        # Locate and click the compose button
        compose = py.locateOnScreen("static/compose.png", confidence=0.8)
        if compose:
            py.click(compose)
        else:
            return "Compose button not found."
        time.sleep(2)

        # Locate and click the number field
        number_field = py.locateOnScreen("static/number.png", confidence=0.8)
        if number_field:
            py.click(number_field)
            py.typewrite(phone)
            time.sleep(3)
            py.press('enter')
            time.sleep(6)
        else:
            return "Number field not found."

        # Locate and click the message field
        message_field = py.locateOnScreen("static/message.png", confidence=0.8)
        py.click(message_field)
        py.typewrite(msg)
        time.sleep(6)
        # Locate and click the send button
        send_button = py.locateOnScreen('static/send.png', confidence=0.8)
        if send_button:
            py.click(send_button)
        else:
            return "Send button not found."

        return "Your message has been sent successfully."
    except Exception as e:
        return str(e)









#################################################   Function for Sending Email      ####################################################
def send_email(receiver_email, subject, message):
    try:
        email = "vikramrajawat1611@gmail.com"
        text = f"Subject : {subject}\n\n{message}"
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email, "cdrwnbctzetqiyhn")
        server.sendmail(email, receiver_email, text)
        server.quit()
        date = datetime.date.today().strftime("%Y-%m-%d")
        return f"Email sent to {receiver_email} successfully on {date}"
    except Exception as e:
        return str(e)
    









###############################################     Function for Sending Email in bulk     #############################################
def send_bulk_emails(email_list, subject, message):
    # Email configuration
    sender_email = "vikramrajawat1611@gmail.com"  # Replace with your email
    sender_password = "cdrwnbctzetqiyhn"      # Replace with your password or app password

    # Connect to SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
    except Exception as e:
        print(f"Failed to connect to SMTP server: {str(e)}")
        return f"Failed to connect to SMTP server: {str(e)}"

    for receiver_email in email_list:
        try:
            # Create a new message container for each recipient
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            # Add message body
            msg.attach(MIMEText(message, 'plain'))

            # Send the email
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print(f"Email sent to {receiver_email} successfully")
        except Exception as e:
            print(f"Error sending email to {receiver_email}: {str(e)}")
    
    server.quit()










############################################     Function for Query Search on Google     ################################################
def google_search(query):
    api_key = "f9dc3a810e1e3f15cec84beff8232b327d66ed776ed0c3f0c1f980992994b62d"
    params ={
        "engine": "google",
        "q": query,
        "api_key": api_key
        }
    search = GoogleSearch(params)
    results = search.get_dict()
    top_results = results.get('organic_results', [])[:5]
    return top_results





#############################################    Function for Speaking String   ########################################################

# Function to speak using pyttsx3
def speak(text, rate=150):
    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()
    finally:
        pythoncom.CoUninitialize()

# Function to print and speak
def print_and_speak(text):
    print(text)
    speak(text)






###############################################   Function For Controlling or Setting Volume   ############################################
# Function to set volume
def set_volume(volume_level):
    try:
        CoInitialize()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume_level = float(volume_level)
        # Volume level must be between 0.0 and 1.0
        volume.SetMasterVolumeLevelScalar(volume_level / 100, None)
        CoUninitialize()
        return f"Volume set to {volume_level}%"
    except Exception as e:
        CoUninitialize()
        return str(e)









#################################################     Function for finding Geo-Cordinates     #############################################
# Function to get geo coordinates and location
def get_geo_location(query=''):
    try:
        if query:
            response = requests.get(f"https://ipinfo.io/{query}")
        else:
            response = requests.get("https://ipinfo.io")
        data = response.json()
        location = data.get('loc', 'Location not found')
        city = data.get('city', 'City not found')
        region = data.get('region', 'Region not found')
        country = data.get('country', 'Country not found')
        return f"Coordinates: {location}, Location: {city}, {region}, {country}"
    except Exception as e:
        return str(e)







##############################################     Function for Cropping Face during live video    ####################################

camera = None  # Global variable to hold the camera instance

def detect_and_generate():
    global camera
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cropped_faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            cropped_faces.append(face)

        if cropped_faces:
            # Resize all faces to a fixed height (adjust as needed)
            max_height = max(face.shape[0] for face in cropped_faces)
            for i in range(len(cropped_faces)):
                face = cropped_faces[i]
                new_height = max_height
                new_width = int(face.shape[1] * (new_height / face.shape[0]))
                cropped_faces[i] = cv2.resize(face, (new_width, new_height))

            combined_faces = cv2.hconcat(cropped_faces)
            combined_faces = cv2.resize(combined_faces, (frame.shape[1], combined_faces.shape[0]))
            frame_with_faces = cv2.vconcat([frame, combined_faces])
        else:
            frame_with_faces = frame

        ret, jpeg = cv2.imencode('.jpg', frame_with_faces)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()








########################################    Function For Applying Filters     #########################################################

# Initialize Cascade Classifiers
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Load Filters
filters = {
    'none': None,  # No filter
   
    'glass': cv2.imread('static/filters/glass.png', cv2.IMREAD_UNCHANGED),
 
    'star': cv2.imread('static/filters/star.png', cv2.IMREAD_UNCHANGED)
}

# Global Variables
cap = None
selected_filter = None

def generate_filtered_video():
    global cap, selected_filter
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            if selected_filter is not None:
                selected_filter_resized = cv2.resize(selected_filter, (w, h))
                
                for i in range(y, y + h):
                    for j in range(x, x + w):
                        if selected_filter_resized[i - y, j - x, 3] != 0:  # Check alpha channel for transparency
                            frame[i, j] = selected_filter_resized[i - y, j - x, :3]  # Assign RGB values

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        



###############################       Function to apply  a Filter on image     ###################################################

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_sepia_filter(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img

def apply_color_filter(img, color):
    b, g, r = color
    colored_img = np.zeros_like(img)
    colored_img[:, :, 0] = b
    colored_img[:, :, 1] = g
    colored_img[:, :, 2] = r
    filtered_img = cv2.addWeighted(img, 0.5, colored_img, 0.5, 0)
    return filtered_img



























UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS











###############     Routing Code Start Here ----------------------------------------------->>



#Routing on Index page by Default
@app.route('/')
def index():
    return render_template('index.html')





#Routing On Whatsapp template 
@app.route('/whatsapp', methods=['GET', 'POST'])
def whatsapp():
    if request.method == 'POST':
        phone_number = request.form['phone_number']
        message = request.form['message']
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])

        result = send_whatsapp_message(phone_number, message, hour, minute)
        return render_template('result.html', result=result)

    return render_template('whatsapp.html')





#Routing on Phone_link template
@app.route('/phone_link', methods=['GET', 'POST'])
def phone_link():
    if request.method == 'POST':
        phone = request.form['phone']
        msg = request.form['msg']

        result = send_message_via_phone_link(phone, msg)
        return render_template('result.html', result=result)

    return render_template('phone_link.html')






#Routing on Email template 
@app.route('/email', methods=['GET', 'POST'])
def email():
    if request.method == 'POST':
        receiver_email = request.form['receiver_email']
        subject = request.form['subject']
        message = request.form['message']

        result = send_email(receiver_email, subject, message)
        return render_template('result.html', result=result)
    return render_template('email.html')






#Routing On Send_Bulk_Email template
@app.route('/send_bulk_email', methods=['GET', 'POST'])
def send_bulk_email():
    if request.method == 'POST':
        # Retrieve form data
        email_list = request.form.get('email_list').split(',')
        email_list = [email.strip() for email in email_list]  # Strip whitespace from each email
        subject = request.form['subject']
        message = request.form['message']

        # Send bulk emails
        result = send_bulk_emails(email_list, subject, message)

        # Prepare result message
        if result:
            result_message = result
        else:
            result_message = f"Bulk emails sent to {', '.join(email_list)} successfully."

        return render_template('result.html', result=result_message)

    return render_template('send_bulk_email.html')








#Routing on Google_search template
@app.route('/google_search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = google_search(query)
        return render_template('search_results.html', results=results)
    return render_template('google_search.html')





#Routing on Speak template
@app.route('/speak', methods=['GET', 'POST'])
def speak_text():
    if request.method == 'POST':
        text = request.form['text']
        print_and_speak(text)
        return render_template('result.html', result=f'Text "{text}" spoken successfully')
    return render_template('speak.html')







#Routing on Volume templatee
@app.route('/volume', methods=['GET', 'POST'])
def volume():
    if request.method == 'POST':
        volume_level = request.form['volume_level']
        result = set_volume(volume_level)
        return render_template('result.html', result=result)
    return render_template('volume.html')







# Routing On Geo-location template
@app.route('/geo_location', methods=['GET', 'POST'])
def geo_location():
    if request.method == 'POST':
        result = get_geo_location()
        return render_template('result.html', result=result)
    return render_template('geo_location.html')

@app.route('/geo_location', methods=['GET', 'POST'])
def geo_location_route():
    if request.method == 'POST':
        location_input = request.form.get('location_input', '')
        result = get_geo_location(location_input)
        return render_template('result.html', result=result)
    return render_template('geo_location.html')








# Routing on Face-detection1(crop) template
@app.route('/face_detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
    return "Camera stopped."









# Routing on Face filter template
@app.route('/face_detection_filters')
def face_detection_filters():
    return render_template('face_detection_filters.html')

@app.route('/video_feed_with_filters')
def video_feed_with_filters():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    return Response(generate_filtered_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_video_filter/<filter_name>')
def select_video_filter(filter_name):
    global selected_filter
    if filter_name in filters:
        selected_filter = filters[filter_name]
        print(f"Filter selected: {filter_name}")
    return redirect(url_for('face_detection_filters'))

@app.route('/stop_filtered_camera')
def stop_filtered_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return redirect(url_for('index'))
    







#Routing for Filter image or apply

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg'))
            return redirect(url_for('filter_image'))
    return render_template('upload_image.html')

@app.route('/filter_image')
def filter_image():
    filters = ['none', 'sepia']
    return render_template('filter_image.html', filters=filters)

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    filter_name = request.json.get('filter')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
    img = cv2.imread(img_path)

    if filter_name == 'sepia':
        img = apply_sepia_filter(img)
    
    processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(processed_img_path, img)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_str})

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



































import joblib
# Load the trained model

model = joblib.load('linear_regression_model.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    weight = None
    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            # Predict height
            predicted_height = model.predict(np.array([[weight]]))
            result = predicted_height[0]
        except ValueError:
            result = "Invalid input. Please enter a numeric value."
    
    return render_template('predict.html', result=result, weight=weight)






from color_detection import detect_color

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            color_name = detect_color(file_path)
            if color_name:
                return render_template('color_result.html', color_name=color_name, image_path=file_path)
            else:
                return "Error processing image."
    return render_template('upload.html')

































import google.generativeai as genai
from IPython.display import Markdown


GOOGLE_API_KEY = 'AIzaSyCBni1ZR0Bx3JzkkZNcQaVcO4b5YxIwFkA'
genai.configure(api_key=GOOGLE_API_KEY)

# Function to handle Gemini API article generation
def generate_article_text(prompt_text):
    try:
        # Choose a Gemini API model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

        # Prompt the model with the text
        response = model.generate_content([prompt_text])

        return response.text

    except Exception as e:
        print(f"Error generating article: {e}")
        return str(e)

# Route for Gemini API article generation form and processing
@app.route('/generate_article', methods=['GET', 'POST'])
def generate_article():
    if request.method == 'POST':
        prompt_text = request.form['prompt_text']

        if prompt_text:
            response_text = generate_article_text(prompt_text)

            return render_template('article_result.html', prompt_text=prompt_text, response_text=response_text)

    return render_template('generate_article.html')

# Route for displaying the result
@app.route('/article_result')
def article_result():
    return render_template('article_result.html')



if __name__ == "__main__":

    app.run(debug=True)

