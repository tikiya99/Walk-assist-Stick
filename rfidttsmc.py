import cv2
import torch
import numpy as np
import mediapipe as mp
from scipy.interpolate import RectBivariateSpline
import serial
import pyttsx3
import time
import csv

# Initialize serial communication
ser = serial.Serial('/dev/ttyACM0', 9600)  # Update to the appropriate port
time.sleep(2)  # Wait for the serial connection to initialize

# Initialize text-to-speech
engine = pyttsx3.init()

# Initialize the body landmarks detection module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Download the model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Process image
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

# Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

# Define depth to distance
def depth_to_distance(depth_value, depth_scale):
    return 1.0 / (depth_value * depth_scale)

# Read messages from CSV file
messages_dict = {}
with open('rfid_messages.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader)  # Skip the header
    for rows in reader:
        uid = rows[0].strip()
        message = rows[1].strip()
        messages_dict[uid] = message

# Function to handle RFID data
rfid_messages = []
def handle_rfid_data(data):
    if "RFID UID:" in data:
        uid = data.split(":")[1].strip().replace(" ", "")
        message = messages_dict.get(uid, "No message found for this UID.")
        rfid_messages.append(f"UID: {uid} - {message}")
        if len(rfid_messages) > 10:  # Keep only the last 10 messages
            rfid_messages.pop(0)
        engine.say(message)
        engine.runAndWait()

# Main Loop
cap = cv2.VideoCapture(0)  # Change to use the camera input
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the body landmarks in the frame
    results = pose.process(img)

    if results.pose_landmarks is not None:
        waist_landmarks = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]]

        mid_point = ((waist_landmarks[0].x + waist_landmarks[1].x) / 2,
                     (waist_landmarks[0].y + waist_landmarks[1].y) / 2,
                     (waist_landmarks[0].z + waist_landmarks[1].z) / 2)
        mid_x, mid_y, mid_z = mid_point

        imgbatch = transform(img).to('cpu')

        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        h, w = output_norm.shape
        x_grid = np.arange(w)
        y_grid = np.arange(h)

        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_mid_filt = spline(mid_y, mid_x)
        depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
        depth_mid_filt = (apply_ema_filter(depth_midas) / 10)[0][0]

        depth_colormap = cv2.applyColorMap((output_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, depth_colormap, 0.4, 0)

        # Divide the screen into 3 sections
        section_width = w // 3
        left_section = output_norm[:, :section_width]
        center_section = output_norm[:, section_width:2*section_width]
        right_section = output_norm[:, 2*section_width:]

        # Calculate the mean depth value for each section
        mean_left_depth = np.mean(left_section)
        mean_center_depth = np.mean(center_section)
        mean_right_depth = np.mean(right_section)

        # Choose the segment with the lowest mean depth value
        mean_depths = [mean_left_depth, mean_center_depth, mean_right_depth]
        lowest_mean_section = mean_depths.index(min(mean_depths))

        # Calculate the center of each segment
        left_center = (section_width // 2, h // 2)
        center_center = (section_width + section_width // 2, h // 2)
        right_center = (2 * section_width + section_width // 2, h // 2)

        segment_centers = [left_center, center_center, right_center]

        # Starting point is the bottom middle of the middle segment
        start_point = (w // 2, h - 1)
        end_point = segment_centers[lowest_mean_section]

        # Create a visualization of the sections
        segments_visualization = overlay.copy()
        cv2.line(segments_visualization, (section_width, 0), (section_width, h), (0, 255, 0), 2)
        cv2.line(segments_visualization, (2*section_width, 0), (2*section_width, h), (0, 255, 0), 2)

        if lowest_mean_section == 0:  # Left section
            cv2.rectangle(segments_visualization, (0, 0), (section_width, h), (255, 0, 0), 3)
            ser.write(b'1')  # Send command
        elif lowest_mean_section == 1:  # Center section
            cv2.rectangle(segments_visualization, (section_width, 0), (2*section_width, h), (255, 0, 0), 3)
        else:  # Right section
            cv2.rectangle(segments_visualization, (2*section_width, 0), (w, h), (255, 0, 0), 3)
            ser.write(b'2')  # Send command

        # Draw the path from start to end point
        cv2.arrowedLine(segments_visualization, start_point, end_point, (255, 255, 255), 2, tipLength=0.05)

        # Display the mean depth value on top of each segment
        cv2.putText(segments_visualization, f"Mean Depth: {mean_left_depth:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(segments_visualization, f"Mean Depth: {mean_center_depth:.2f}", (section_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(segments_visualization, f"Mean Depth: {mean_right_depth:.2f}", (2*section_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(overlay, "Depth in unit: " + str(np.format_float_positional(depth_mid_filt, precision=3)),
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.imshow('Walking', overlay)
        cv2.imshow('Segments', segments_visualization)

    # Read RFID data from the serial port
    if ser.in_waiting > 0:
        rfid_data = ser.readline().decode('utf-8').strip()
        handle_rfid_data(rfid_data)

    # Create a black image to display RFID messages
    rfid_display = np.zeros((300, 400, 3), dtype=np.uint8)

    # Display RFID messages
    for i, msg in enumerate(rfid_messages):
        y_pos = 30 + i * 25
        cv2.putText(rfid_display, msg, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('RFID Messages', rfid_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()  
