def median(left, right=None):
    if right is None:
        return left
    else:
        return left if left < right else right

def recursive_recall(words):
    if len(words) == 1:
        return words[0]
    else:
        middle = len(words) // 2
        left_median = recursive_recall(words[:middle])
        right_median = recursive_recall(words[middle:])
        return median(left_median, right_median)

# Helper function to test and demonstrate
def find_median(words):
    words.sort()  # Sort the words to ensure we can find the median correctly
    return recursive_recall(words)

# Example usage
words = ["banana", "apple", "cherry", "date", "fig", "grape"]
print("Median word:", find_median(words))

