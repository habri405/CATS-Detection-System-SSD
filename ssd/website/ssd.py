from flask import Blueprint, Flask, request, render_template, send_from_directory, flash, redirect, url_for, current_app
from flask import Flask, render_template, request, redirect, send_file, Response
from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
import os
import argparse
import io
import tempfile
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import yaml

ssd = Blueprint('ssd', __name__)

catstrafficsign_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load your pre-trained model (adjust path as necessary)
model_path = os.path.join(catstrafficsign_dir, 'website', 'customTF2', 'data', 'inference_graph', 'saved_model')  # Example: '/path/to/local/model'
detector = tf.saved_model.load(model_path)

# Assuming you have a list of class names that matches the labels from training
class_names = ['Billboard', 'Green- GO', 'Green- GO Left', 'Green- GO Right', 'Green- GO Straight', 'Hump', 'Obstacles Ahead', 'Pass either side', 'Red- STOP', 'Red- STOP Left', 'Red- STOP Right', 'Red- STOP Straight', 'Sign Board', 'Stop', 'car', 'person']  # Example class names
# Function to load the model from a given path
def load_model(model_path):
    return tf.saved_model.load(model_path)
model = load_model(model_path)

upload_folder = os.path.join(catstrafficsign_dir, 'website', 'static', 'uploads')
detection_folder = os.path.join(catstrafficsign_dir, 'website', 'static', 'detections')

@ssd.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            detection_path = os.path.join(detection_folder, "processed_" + filename)
            if filename.lower().endswith(('.mp4', '.avi')):
                # Process video
                process_video(filepath, detection_path, model, class_names)
            else:
                # Process image
                detect_and_draw_image(filepath, detection_path, model)

            return send_from_directory(detection_folder, "processed_" + filename)

        flash('No valid file selected')
        return redirect(request.url)

    return render_template('ssdprocess.html')

# Function to load the model from a given path
def load_model(model_path):
    return tf.saved_model.load(model_path)

def detect_and_draw_image(image_path, detection_path, model_path):
    # Load the model from the path
    # model = load_model(model_path)
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)  # Ensure dtype matches model's expected input

    # Run detection
    detections = model(input_tensor)

    # Process detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']
    detection_classes = detections['detection_classes']

    # Draw each detection with class name and percentage
    for i in range(num_detections):
        score = detection_scores[i]
        if score >= 0.5:  # Adjust threshold as needed
            box = detection_boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            box = box.astype(int)
            class_id = int(detection_classes[i]) - 1  # Adjust for 0-based indexing
            if class_id < len(class_names):  # Check if class_id is within the range of class_names list
                class_name = class_names[class_id]  # Get the class name using the class ID
                label = f"{class_name}: {score*100:.2f}%"  # Label format
            
                # Draw rectangle and label on the image
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                cv2.putText(image, label, (box[1], box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else:
                # Handle unexpected class_id values
                print(f"Warning: Detected class_id {class_id+1} is out of range from the defined class_names.")


    # Save or display the resulting image
    cv2.imwrite(detection_path, image)
    # Optionally, display the image
    # cv2.imshow('Detected Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_video(video_path, detection_path, model, class_names):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(detection_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor([frame_rgb], dtype=tf.uint8)
            detections = model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detection_boxes = detections['detection_boxes']
            detection_scores = detections['detection_scores']
            detection_classes = detections['detection_classes'].astype(int)

            for i in range(num_detections):
                score = detection_scores[i]
                if score >= 0.5:  # Adjust threshold as needed
                    box = detection_boxes[i] * np.array([frame_height, frame_width, frame_height, frame_width])
                    box = box.astype(int)
                    class_id = int(detection_classes[i]) - 1  # Adjust for 0-based indexing
                    if class_id < len(class_names):  # Check if class_id is within the range of class_names list
                        class_name = class_names[class_id]  # Get the class name using the class ID
                        label = f"{class_name}: {score*100:.2f}%"  # Label format
                    
                        # Draw rectangle and label on the frame
                        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                        cv2.putText(frame, label, (box[1], box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        # Handle unexpected class_id values
                        print(f"Warning: Detected class_id {class_id+1} is out of range from the defined class_names.")

            out.write(frame)  # Write out the frame
        else:
            break

    cap.release()
    out.release()

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'MP4'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@ssd.route("/faq")
def faq():
    return render_template("faq.html")

@ssd.route("/about_us")
def about_us():
    return render_template("about_us.html")


