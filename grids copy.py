import cv2
import numpy as np

def draw_grid(frame, rows, cols, color=(0, 255, 0), thickness=2):
    height, width, _ = frame.shape
    row_height = height // rows
    col_width = width // cols
    
    # Draw grid lines
    for i in range(1, cols):
        x = i * col_width
        cv2.line(frame, (x, 0), (x, height), color, thickness)
        
    for i in range(1, rows):
        y = i * row_height
        cv2.line(frame, (0, y), (width, y), color, thickness)
        
    # Display Cartesian coordinates (x, y) based on grid indices (r, c)
    for r in range(rows):
        for c in range(cols):
            # Calculate the center of the cell
            cell_center_x = c * col_width + col_width // 2
            cell_center_y = r * row_height + row_height // 2
            
            # Display Cartesian coordinates (x, y)
            x_coord = c
            y_coord = rows - 1 - r
            cv2.putText(frame, f"({x_coord},{y_coord})", (cell_center_x - 30, cell_center_y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def generate_matrix(rows, cols, default_value=0):
    return np.full((rows, cols), default_value)

def is_point_in_rect(point, rect):
    x, y = point
    (x1, y1), (x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        start_position = None
        final_position = None
        obstacle_positions = []
        
        # Draw detected markers
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Manually draw the IDs on the markers with custom labels
            for i, corner in enumerate(corners):
                # Get the top-left corner of the marker
                top_left_corner = tuple(corner[0][0].astype(int))
                bottom_right_corner = tuple(corner[0][2].astype(int))
                marker_id = ids[i][0]
                
                # Get the label for the marker ID, defaulting to the ID if not found
                label = marker_to_label.get(marker_id, f"ID {marker_id}")
                
                # Draw the label near the top-left corner
                cv2.putText(frame, label, (top_left_corner[0], top_left_corner[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Store the positions of the start, final destination, and obstacles
                if label == 'Start':
                    start_position = top_left_corner
                elif label == 'FinalDestination':
                    final_position = top_left_corner
                elif 'Obstacle' in label:
                    obstacle_positions.append((top_left_corner, bottom_right_corner))
        
        frame = draw_grid(frame, rows, cols)
        
        if start_position is not None and final_position is not None:
            current_position = np.array(start_position)
            destination = np.array(final_position)
            obstacle_encountered = False
            line_color = (0, 255, 0)  # Green line initially
            
            while np.linalg.norm(destination - current_position) > 10:
                direction_vector = destination - current_position
                distance = np.linalg.norm(direction_vector)
                direction_vector = direction_vector / distance  # Normalize the vector
                
                end_position = current_position + 50 * direction_vector
                end_position = tuple(end_position.astype(int))
                
                collision = False
                for obstacle in obstacle_positions:
                    if is_point_in_rect(end_position, obstacle):
                        collision = True
                        break
                
                if collision:
                    line_color = (0, 0, 255)  # Change line color to red when obstacle encountered
                    cv2.line(frame, tuple(current_position.astype(int)), end_position, line_color, 2)
                    current_position = np.array(end_position)
                    obstacle_encountered = True
                else:
                    line_color = (0, 255, 0)  # Green line when no obstacle
                    cv2.line(frame, tuple(current_position.astype(int)), end_position, line_color, 2)
                    current_position = np.array(end_position)
                    obstacle_encountered = False
                
                if np.linalg.norm(destination - current_position) < 50:
                    break

                if obstacle_encountered:
                    # Change direction and line color
                    direction_vector = np.array([-direction_vector[1], direction_vector[0]])
                    obstacle_encountered = False
                else:
                    direction_vector = destination - current_position

                distance = np.linalg.norm(direction_vector)
                direction_vector = direction_vector / distance  # Normalize the vector
                
        cv2.imshow('Camera with Grid and ArUco Markers', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
