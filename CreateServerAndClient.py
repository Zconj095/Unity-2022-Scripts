import socket

def create_server(host='0.0.0.0', port=12345):
    """ Creates a simple server that listens for incoming connections. """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)

def create_client(host='localhost', port=12345):
    """ Creates a simple client that connects to a server and sends data. """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b'Hello, world')
        data = s.recv(1024)
    print(f"Received {data}")

# To run the server
# create_server()

# To run the client (connect to the server)
# create_client()
