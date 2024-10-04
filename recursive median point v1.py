def recursive_median_point(points):
    if len(points) <= 1:
        return points[0]

    middle = len(points) // 2
    left_points = points[:middle]
    right_points = points[middle:]

    left_median = recursive_median_point(left_points)
    right_median = recursive_median_point(right_points)

    if len(left_points) > len(right_points):
        return left_median
    else:
        return right_median

def recursive_recall(words):
    if len(words) == 1:
        return words[0]
    else:
        middle = len(words) // 2
        left_median = recursive_recall(words[:middle])
        right_median = recursive_recall(words[middle:])
        return median(left_median, right_median)

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def factorial_relay(n):
    if n == 0:
        return 1
    else:
        def factorial_helper(n):
            if n == 0:
                return 1
            else:
                return n * factorial_helper(n - 1)

        return factorial_helper(n)
