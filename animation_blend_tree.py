def animation_blend_tree(blend_states: list) -> str:
    blend_states_dict = {
        "fade": "Fade",
        "ease_in": "Ease In",
        "ease_out": "Ease Out",
        "linear": "Linear",
        "spline": "Spline",
        "step": "Step"
    }
    blend_tree = ""
    for state in blend_states:
        if state in blend_states_dict:
            blend_tree += f"{blend_states_dict[state]}, "
        else:
            blend_tree += f"Unknown state: {state}, "
    return blend_tree.strip().rstrip(", ")

# Example usage:
print(animation_blend_tree(["fade", "ease_out", "linear"]))  # Output: Fade, Ease Out, Linear
print(animation_blend_tree(["spline", "step"]))  # Output: Spline, Step
print(animation_blend_tree(["unknown", "ease_in"]))  # Output: Unknown state: unknown, Ease In