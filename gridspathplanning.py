import heapq
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def _init_(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node (heuristic)
        self.f = 0  # Total cost

    def _lt_(self, other):
        return self.f < other.f

def astar(maze, start, end):
    # Initialize start and end node
    start_node = Node(start)
    end_node = Node(end)

    # Initialize open and closed lists
    open_list = []
    closed_list = set()

    # Heapify the open_list and add the start node
    heapq.heappush(open_list, start_node)

    # Loop until the open list is empty
    while open_list:
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # Check if we reached the goal
        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Generate children
        (x, y) = current_node.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Adjacent squares

        for next_position in neighbors:
            # Make sure within range
            if next_position[0] > (len(maze) - 1) or next_position[0] < 0 or next_position[1] > (len(maze[0]) - 1) or next_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[next_position[0]][next_position[1]] != 0:
                continue

            # Create new node
            neighbor = Node(next_position, current_node)

            # If the neighbor is in the closed list, skip it
            if neighbor.position in closed_list:
                continue

            # Calculate f, g, and h
            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - end_node.position[0]) * 2) + ((neighbor.position[1] - end_node.position[1]) * 2)
            neighbor.f = neighbor.g + neighbor.h

            # If the neighbor is already in the open list with a lower f, skip it
            if add_to_open(open_list, neighbor):
                heapq.heappush(open_list, neighbor)

    return None

def add_to_open(open_list, neighbor):
    for node in open_list:
        if neighbor == node and neighbor.f >= node.f:
            return False
    return True

def visualize_maze(maze, path=None):
    maze_np = np.array(maze)
    fig, ax = plt.subplots()
    ax.imshow(maze_np, cmap=plt.cm.Dark2)
    
    # Draw the path
    if path:
        for position in path:
            ax.plot(position[1], position[0], "s", color="red")

    plt.show()

# Define the maze as a 2D list
# 0: walkable, 1: obstacle
maze = [
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0],
]

start = (0, 0)
end = (4, 6)

path = astar(maze, start, end)
print("Path:", path)

# Visualization of the maze and the path
visualize_maze(maze, path)