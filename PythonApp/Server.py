import socket

def start_server(host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        try:
            while True:  # Keep the server running to accept new connections
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        print(f"Received: {data.decode()}")
                        conn.sendall(data)  # Echo back the received data
        except KeyboardInterrupt:
            print("Server is shutting down.")

if __name__ == "__main__":
    start_server()