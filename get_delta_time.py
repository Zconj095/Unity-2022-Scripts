import time

delta_time = 0
start_time = time.time()

def get_delta_time():
    global delta_time, start_time
    current_time = time.time()
    delta_time = current_time - start_time
    start_time = current_time
    return delta_time