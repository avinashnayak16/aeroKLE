import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.137.1', 8000))  # Replace with main host IP

while True:
    message = "Object detected!"
    client_socket.send(message.encode())
