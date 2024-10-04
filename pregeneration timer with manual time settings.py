import time

def pregeneration_timer(start_time, threshold=10, borrowing_value=5):
    current_time = start_time
    last_borrowed_at = -1  # Initialize to a value that cannot be a valid threshold

    while current_time > 0:
        print(f"Remaining Time: {current_time} seconds")
        time.sleep(1)  # Wait for one second
        current_time -= 1

        # Apply subtractive borrowing at the threshold
        if current_time // threshold > last_borrowed_at and current_time != 0:
            last_borrowed_at = current_time // threshold
            if current_time % threshold == 0:
                print(f"Threshold Reached: {current_time}")
                current_time -= borrowing_value
                print(f"Subtractive Borrowing Applied: New Time: {current_time} seconds")

    print("Timer Completed.")

# Example usage: Set a timer for 30 seconds
pregeneration_timer(60)
