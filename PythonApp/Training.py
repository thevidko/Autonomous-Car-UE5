import socket
import csv
import os
import time
import signal

HOST = "127.0.0.1"  # Listen on all available interfaces
PORT = 65432
FILENAME = 'received_messages.csv'
BUFFER_SIZE = 1024

def signal_handler(sig, frame):
    print("Interrupt received, closing server...")
    server.close()
    os.remove(FILENAME) if os.path.exists(FILENAME) else None  # Remove partially written file
    exit(0)

signal.signal(signal.SIGINT, signal_handler)  # Register signal handler for graceful termination

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen()
    print(f"Server listening on {HOST}:{PORT}...")

    while True:
        conn, addr = server.accept()
        with conn:
            print(f'Connected by {addr}')
            try:
                with open(FILENAME, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    while True:
                        data = conn.recv(BUFFER_SIZE).decode('utf-8')
                        if not data:
                            break
                        # Validate CSV format for robustness
                        try:
                            reader = csv.reader(data.splitlines())
                            for row in reader:
                                writer.writerow(row)
                        except csv.Error as e:
                            print(f"Invalid CSV data received: {e}")
                            conn.sendall("Invalid CSV format. Please send valid CSV messages only.".encode('utf-8'))

            except ConnectionError as e:
                print(f"Connection error: {e}")
            finally:
                conn.close()
