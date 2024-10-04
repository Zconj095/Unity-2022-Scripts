def generate_story(prompt, depth, max_depth):
    if depth > max_depth:
        return prompt
    
    # Simple text generation example (this can be replaced with a more complex model)
    new_prompt = prompt + " And then something interesting happened."
    
    return generate_story(new_prompt, depth + 1, max_depth)

# Example usage
initial_prompt = "Once upon a time, in a land far away, there was a recursive text prompt."
story = generate_story(initial_prompt, 0, 5)
print(story)

import random

def generate_next_sentence():
    sentences = [
        " The hero continued their journey.",
        " A dragon appeared on the horizon.",
        " They discovered an ancient artifact.",
        " A mysterious figure offered guidance.",
        " They faced a challenging obstacle."
    ]
    return random.choice(sentences)

def generate_story(prompt, depth, max_depth):
    if depth > max_depth:
        return prompt
    
    new_prompt = prompt + generate_next_sentence()
    
    return generate_story(new_prompt, depth + 1, max_depth)

# Example usage
initial_prompt = "Once upon a time, in a land far away, there was a recursive text prompt."
story = generate_story(initial_prompt, 0, 5)
print(story)

def generate_story(prompt, depth, max_depth):
    if depth >= max_depth:
        return prompt + " The end."

    new_prompt = prompt + generate_next_sentence()
    
    return generate_story(new_prompt, depth + 1, max_depth)

# Example usage
initial_prompt = "Once upon a time, in a land far away, there was a recursive text prompt."
story = generate_story(initial_prompt, 0, 5)
print(story)
