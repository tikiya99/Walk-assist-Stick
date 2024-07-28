import socket
import random
import time
from pynput import keyboard

ESP32_IP = "192.168.10.209"  # Replace with your ESP32 IP address
ESP32_PORT = 80

def send_random_command():
    commands = ['1', '2', '3']
    command_to_send = random.choice(commands)
    print(f"Sending command: {command_to_send}")

    try:
        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((ESP32_IP, ESP32_PORT))

        # Send data
        client_socket.send(command_to_send.encode())

        # Receive response
        response = client_socket.recv(1024)
        print("Received from ESP32:", response.decode())

        # Close the connection
        client_socket.close()
    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def on_press(key):
    try:
        if key.char == 'q':
            print("Terminating script...")
            return False  # Returning False stops the listener
    except AttributeError:
        pass

if __name__ == "__main__":
    print("Press 'q' to quit the script")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while listener.running:
        send_random_command()
        time.sleep(5)  # Wait for 5 seconds before sending the next command
