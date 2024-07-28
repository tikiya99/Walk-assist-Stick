import socket
import cv2
import numpy as np
import time
from pynput import keyboard

# ESP32 IP address and port
ESP32_IP = "192.168.10.209"  # Replace with your ESP32 IP address
ESP32_PORT = 80

# Initialize Bluetooth serial communication
def send_command(command):
    print(f"Sending command: {command}")

    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # Connect to the server
            client_socket.connect((ESP32_IP, ESP32_PORT))

            # Send the command
            client_socket.send(command.encode())

            # Receive the response
            response = client_socket.recv(1024)
            print("Received from ESP32:", response.decode())
    except socket.error as e:
        print(f"Socket error: {e}")
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

# Function to find path from start to end avoiding obstacles
def find_path(start, end, obstacles, rows, cols):
    path = []
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        
        if current == end:
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        open_set.remove(current)
        
        for neighbor in get_neighbors(current, rows, cols):
            if neighbor in obstacles:
                continue
            
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                open_set.add(neighbor)
    
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position, rows, cols):
    x, y = position
    neighbors = []
    if x > 0: neighbors.append((x - 1, y))
    if x < cols - 1: neighbors.append((x + 1, y))
    if y > 0: neighbors.append((x, y - 1))
    if y < rows - 1: neighbors.append((x, y + 1))
    return neighbors

# Function to handle key press events
def on_press(key):
    try:
        if key.char == 'q':
            print("Terminating script...")
            return False  # Returning False stops the listener
    except AttributeError:
        pass

# Main function
def main(camera_index=0, resolution=(1280, 720), rows=6, cols=6):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Generate the matrix
    matrix = generate_matrix(rows, cols)
    print("Generated matrix:")
    print(matrix)
    
    # Check if aruco module is available
    if not hasattr(cv2, 'aruco'):
        print("Error: OpenCV installed without aruco module.")
        return
    
    # Load the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # Dictionary to store ArUco marker IDs and their corresponding obstacle labels
    marker_to_label = {}
    
    # Function to update the custom label for a given ArUco marker ID
    def update_label(marker_id, label):
        marker_to_label[marker_id] = label
    
    # Example custom labels (you can modify these as needed)
    update_label(203, 'Start')
    update_label(23, 'Obstacle1')
    update_label(124, 'Obstacle2')
    update_label(62, 'Obstacle3')
    update_label(40, 'Obstacle4')
    update_label(98, 'FinalDestination')
    
    # Start the keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    while listener.running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        start_position = None
        end_position = None
        obstacles = set()
        
        # Draw detected markers
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Manually draw the IDs on the markers with custom labels
            for i, corner in enumerate(corners):
                top_left_corner = tuple(corner[0][0].astype(int))
                marker_id = ids[i][0]
                label = marker_to_label.get(marker_id, f"ID {marker_id}")
                cv2.putText(frame, label, (top_left_corner[0], top_left_corner[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
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
            path = find_path(start_position, end_position, obstacles, rows, cols)
            
            # Draw the path on the frame
            for i in range(len(path) - 1):
                start_pixel = ((path[i][0] * resolution[0] // cols), 
                               ((rows - 1 - path[i][1]) * resolution[1] // rows))
                end_pixel = ((path[i+1][0] * resolution[0] // cols), 
                             ((rows - 1 - path[i+1][1]) * resolution[1] // rows))
                cv2.line(frame, start_pixel, end_pixel, (0, 0, 255), 2)
                
                # Send commands based on the path
                current_step = path[i + 1]
                previous_step = path[i]
                
                if current_step[0] > previous_step[0]:
                    send_command('1')  # Move right
                elif current_step[0] < previous_step[0]:
                    send_command('2')  # Move left
                elif current_step[1] > previous_step[1]:
                    send_command('3')  # Move down
                elif current_step[1] < previous_step[1]:
                    send_command('4')  # Move up
                
                time.sleep(0.5)  # Adjust sleep time as necessary
        
        frame = draw_grid(frame, rows, cols)
        cv2.imshow('Camera with Grid and ArUco Markers', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
