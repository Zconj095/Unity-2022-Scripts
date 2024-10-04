import math

def ease_of_use(command_complexity, recursive_breakdown):
    """Calculate ease of use"""
    return (1 - command_complexity) * recursive_breakdown

def flexibility(user_intent, system_adaptability):
    """Calculate flexibility"""
    return user_intent * system_adaptability

def power(system_complexity, recursive_processing):
    """Calculate power"""
    return system_complexity * recursive_processing

def speed(recursive_breakdown_depth, processing_time):
    """Calculate speed"""
    return (1 - recursive_breakdown_depth) * processing_time

def accuracy(user_intent, system_understanding):
    """Calculate accuracy"""
    return user_intent * system_understanding

# Example usage:
command_complexity = 0.5  # Medium complexity command
recursive_breakdown = 0.8  # High recursive breakdown
user_intent = 0.9  # Strong user intent
system_adaptability = 0.7  # High system adaptability
system_complexity = 0.8  # High system complexity
recursive_processing = 0.9  # High recursive processing
recursive_breakdown_depth = 0.6  # Medium recursive breakdown depth
processing_time = 0.5  # Medium processing time
system_understanding = 0.8  # High system understanding

print("Ease of use:", ease_of_use(command_complexity, recursive_breakdown))
print("Flexibility:", flexibility(user_intent, system_adaptability))
print("Power:", power(system_complexity, recursive_processing))
print("Speed:", speed(recursive_breakdown_depth, processing_time))
print("Accuracy:", accuracy(user_intent, system_understanding))