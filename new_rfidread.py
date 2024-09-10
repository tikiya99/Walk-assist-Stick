import socket
import cv2
import numpy as np
import time
import pyttsx3
import csv
from pynput import keyboard
import multiprocessing
import os

#export QT_QPA_PLATFORM=xcb

# ESP32 IP address and port
ESP32_IP = "192.168.10.209"  # Replace with your ESP32 IP address
ESP32_PORT = 80

# Initialize text-to-speech
engine = pyttsx3.init()

# Global variables for inter-process communication
frame_queue = multiprocessing.Queue()
command_queue = multiprocessing.Queue()
path_request_queue = multiprocessing.Queue()
path_result_queue = multiprocessing.Queue()
rfid_queue = multiprocessing.Queue()

# Function to handle key press events
def on_press(key):
    try:
        if key.char == 'q':
            print("Terminating script...")
            os._exit(0)
    except AttributeError:
        pass


# Function to load RFID data from CSV
def load_rfid_data(csv_file):
    rfid_data = {}
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # Only process rows with at least 2 columns
                    rfid_id, name = row[0], row[1]
                    rfid_data[rfid_id] = name
    except Exception as e:
        print(f"Error loading RFID data: {e}")
    return rfid_data

# Check RFID against CSV and use TTS to announce name
def check_rfid_and_announce(rfid_id, rfid_data):
    if rfid_id in rfid_data:
        person_name = rfid_data[rfid_id]
        message = f"Access granted to {person_name}."
        engine.say(message)
        engine.runAndWait()
        print(message)
    else:
        message = "Access denied."
        engine.say(message)
        engine.runAndWait()
        print(message)

# Function to draw grid on the frame
def draw_grid(frame, rows, cols, color=(0, 255, 0), thickness=2):
    height, width, _ = frame.shape
    row_height = height // rows
    col_width = width // cols

    for i in range(1, cols):
        x = i * col_width
        cv2.line(frame, (x, 0), (x, height), color, thickness)
    for i in range(1, rows):
        y = i * row_height
        cv2.line(frame, (0, y), (width, y), color, thickness)
    for r in range(rows):
        for c in range(cols):
            cell_center_x = c * col_width + col_width // 2
            cell_center_y = r * row_height + row_height // 2
            x_coord = c
            y_coord = rows - 1 - r
            cv2.putText(frame, f"({x_coord},{y_coord})", (cell_center_x - 30, cell_center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

# Function to handle RFID card input from ESP32
def handle_rfid_input():
    os.sched_setaffinity(0, {3})  # Pin this process to core 3
    rfid_data = load_rfid_data('rfid_data.csv')  # Assuming your CSV file is named 'rfid_data.csv'
    
    while True:
        rfid_id = rfid_queue.get()  # Get RFID ID from queue
        check_rfid_and_announce(rfid_id, rfid_data)

# Function to receive RFID card ID from ESP32
def receive_rfid_from_esp():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((ESP32_IP, ESP32_PORT))
            client_socket.settimeout(1.0)  # Set a timeout for reading data
            
            while True:
                try:
                    rfid_id = client_socket.recv(1024).decode()  # Receive RFID card ID from ESP32
                    if rfid_id:  # Check if any data is received
                        print(f"Received RFID ID: {rfid_id}")
                        rfid_queue.put(rfid_id)  # Put RFID ID in queue for processing
                except socket.timeout:
                    # No data received, continue looping
                    continue
    except Exception as e:
        print(f"Error receiving RFID ID: {e}")

# Function to generate matrix
def generate_matrix(rows, cols, default_value=0):
    return np.full((rows, cols), default_value)

# Function to capture frames from the camera (same as original)
def capture_frames(camera_index, resolution, rows, cols):
    os.sched_setaffinity(0, {0})  # Pin this process to core 0
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        frame_queue.put(frame)
        time.sleep(0.033)  # Capture at ~30 FPS

    cap.release()

# Function to process frames and detect ArUco markers (same as original)
def process_frames(resolution, rows, cols):
    os.sched_setaffinity(0, {1})  # Pin this process to core 1
    matrix = generate_matrix(rows, cols)
    if not hasattr(cv2, 'aruco'):
        print("Error: OpenCV installed without aruco module.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    marker_to_label = {
        203: 'Start',
        23: 'Obstacle1',
        124: 'Obstacle2',
        62: 'Obstacle3',
        40: 'Obstacle4',
        98: 'FinalDestination'
    }

    while True:
        frame = frame_queue.get()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        start_position = None
        end_position = None
        obstacles = set()
        marker_positions = {}

        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                top_left_corner = tuple(corner[0][0].astype(int))
                marker_id = ids[i][0]
                label = marker_to_label.get(marker_id, f"ID {marker_id}")
                cv2.putText(frame, label, (top_left_corner[0], top_left_corner[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Store the position of the marker
                marker_positions[label] = top_left_corner

                grid_x = top_left_corner[0] // (resolution[0] // cols)
                grid_y = top_left_corner[1] // (resolution[1] // rows)
                grid_position = (grid_x, rows - 1 - grid_y)

                if label == 'Start':
                    start_position = grid_position
                elif label == 'FinalDestination':
                    end_position = grid_position
                else:
                    obstacles.add(grid_position)

        if start_position is not None and end_position is not None:
            path_request_queue.put((start_position, end_position, obstacles))

        # Try to get the planned path from the path_result_queue
        try:
            path = path_result_queue.get_nowait()
            for i in range(len(path) - 1):
                start_pixel = (path[i][0] * resolution[0] // cols, (rows - 1 - path[i][1]) * resolution[1] // rows)
                end_pixel = (path[i+1][0] * resolution[0] // cols, (rows - 1 - path[i+1][1]) * resolution[1] // rows)
                cv2.line(frame, start_pixel, end_pixel, (0, 0, 255), 2)

                current_step = path[i + 1]
                previous_step = path[i]

                if current_step[0] > previous_step[0]:
                    command_queue.put('1')
                elif current_step[0] < previous_step[0]:
                    command_queue.put('2')
                elif current_step[1] > previous_step[1]:
                    command_queue.put('3')
                elif current_step[1] < previous_step[1]:
                    command_queue.put('4')
        except:
            pass

        frame = draw_grid(frame, rows, cols)
        cv2.imshow('Camera with Grid and ArUco Markers', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Function to perform path planning
def path_planning(rows, cols):
    os.sched_setaffinity(0, {2})  # Pin this process to core 2
    while True:
        start_position, end_position, obstacles = path_request_queue.get()
        path = find_path(start_position, end_position, obstacles, rows, cols)
        path_result_queue.put(path)

# Function to send commands to ESP32
def send_commands():
    os.sched_setaffinity(0, {3})  # Pin this process to core 3
    while True:
        command = command_queue.get()
        send_command(command)
        time.sleep(0.2)  # Adjust sleep time as necessary

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    camera_index = 2
    resolution = (1280, 720)
    rows = 6
    cols = 6

    # RFID processing
    rfid_process = multiprocessing.Process(target=handle_rfid_input)
    rfid_receive_process = multiprocessing.Process(target=receive_rfid_from_esp)

    capture_process = multiprocessing.Process(target=capture_frames, args=(camera_index, resolution, rows, cols))
    process_process = multiprocessing.Process(target=process_frames, args=(resolution, rows, cols))
    planning_process = multiprocessing.Process(target=path_planning, args=(rows, cols))
    command_process = multiprocessing.Process(target=send_commands)

    rfid_process.start()
    rfid_receive_process.start()
    capture_process.start()
    process_process.start()
    planning_process.start()
    command_process.start()

    rfid_process.join()
    rfid_receive_process.join()
    capture_process.join()
    process_process.join()
    planning_process.join()
    command_process.join()
