def get_input_event():
    if keyboard.is_pressed('esc'):
        return 'quit'
    elif keyboard.is_pressed('w'):
        return 'move_forward'
    elif keyboard.is_pressed('s'):
        return 'move_backward'
    elif keyboard.is_pressed('a'):
        return 'move_left'
    elif keyboard.is_pressed('d'):
        return 'move_right'
    else:
        return None