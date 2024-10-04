def data_calibrator_with_borrowing(threshold=10, borrowing_value=5, max_value=100):
    current_value = 0
    last_borrowed_at = -1  # Initialize to a value that cannot be a valid threshold

    while current_value <= max_value:
        print(f"Current Value: {current_value}")
        current_value += 1

        # Check if the threshold is reached or exceeded, and apply subtractive borrowing only once per threshold
        if current_value // threshold > last_borrowed_at:
            last_borrowed_at = current_value // threshold
            if current_value % threshold == 0 or current_value == max_value:
                print(f"Threshold Reached or Exceeded: {current_value}")
                current_value -= borrowing_value
                print(f"Subtractive Borrowing Applied: New Value: {current_value}")

    print("Calibration Completed.")

# Run the updated calibrator
data_calibrator_with_borrowing()
