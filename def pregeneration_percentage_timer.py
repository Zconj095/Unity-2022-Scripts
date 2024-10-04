import time
def pregeneration_percentage_timer(duration, threshold_percentage=10, borrowing_percentage=5):
    elapsed_time = 0
    percentage_completion = 0
    last_borrowed_at = -1  # Initialize to a value that cannot be a valid threshold

    while percentage_completion < 100:
        print(f"Completion: {percentage_completion}%")
        time.sleep(duration / 100)  # Wait proportionally to reach 100% in the specified duration
        elapsed_time += duration / 100
        percentage_completion = (elapsed_time / duration) * 100

        # Apply subtractive borrowing at the threshold
        if percentage_completion // threshold_percentage > last_borrowed_at and percentage_completion < 100:
            last_borrowed_at = percentage_completion // threshold_percentage
            if percentage_completion % threshold_percentage == 0:
                print(f"Threshold Reached: {percentage_completion}%")
                percentage_completion -= borrowing_percentage
                elapsed_time = (percentage_completion / 100) * duration
                print(f"Subtractive Borrowing Applied: New Completion: {percentage_completion}%")

    print("Timer Completed. 100% Reached.")

# Example usage: Set a timer duration for 60 seconds
pregeneration_percentage_timer(60)
