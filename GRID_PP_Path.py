import socket
import cv2
import numpy as np
import time
from pynput import keyboard
import multiprocessing
import os

# ESP32 IP address and port
ESP32_IP = "192.168.10.209"  # Replace with your ESP32 IP address
ESP32_PORT = 80

# Global variables for inter-process communication
frame_queue = multiprocessing.Queue()
command_queue = multiprocessing.Queue()
path_request_queue = multiprocessing.Queue()
path_result_queue = multiprocessing.Queue()

# Initialize Bluetooth serial communication
def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((ESP32_IP, ESP32_PORT))
            client_socket.send(command.encode())
            response = client_socket.recv(1024)
            print("Received from ESP32:", response.decode())
    except Exception as e:
        print(f"Error: {e}")

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

# Function to generate matrix
def generate_matrix(rows, cols, default_value=0):
    return np.full((rows, cols), default_value)

# Updated function to find path from start to end avoiding obstacles
def find_path(start, end, obstacles, rows, cols):
    path = [start]
    current_position = start

    while current_position != end:
        next_position = (current_position[0] + np.sign(end[0] - current_position[0]),
                         current_position[1] + np.sign(end[1] - current_position[1]))

        if next_position in obstacles:
            next_position = (current_position[0] + np.sign(end[0] - current_position[0]),
                             current_position[1])
            if next_position in obstacles:
                next_position = (current_position[0],
                                 current_position[1] + np.sign(end[1] - current_position[1]))

        path.append(next_position)
        current_position = next_position
        if current_position == end:
            break

    return path

# Function to handle key press events
def on_press(key):
    try:
        if key.char == 'q':
            print("Terminating script...")
            os._exit(0)
    except AttributeError:
        pass

# Function to capture frames from the camera
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

# Function to process frames and detect ArUco markers
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

    capture_process = multiprocessing.Process(target=capture_frames, args=(camera_index, resolution, rows, cols))
    process_process = multiprocessing.Process(target=process_frames, args=(resolution, rows, cols))
    planning_process = multiprocessing.Process(target=path_planning, args=(rows, cols))
    command_process = multiprocessing.Process(target=send_commands)

    capture_process.start()
    process_process.start()
    planning_process.start()
    command_process.start()

    capture_process.join()
    process_process.join()
    planning_process.join()
    command_process.join()