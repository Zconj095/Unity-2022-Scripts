import bpy
import csv
import tensorflow as tf
import pandas as pd

def init_joysticks():
    joysticks = bpy.context.window_manager.gamepad_devices
    
    if len(joysticks) == 0:
        print("No joysticks found.")
        return None, None
    
    if len(joysticks) == 1:
        print(f"One joystick found: {joysticks[0].name}")
        return joysticks[0], None
    
    print(f"Two joysticks found: {joysticks[0].name} and {joysticks[1].name}")
    return joysticks[0], joysticks[1]

def read_joystick(joystick):
    axes = joystick.axis_values
    left_stick = (axes[0], axes[1])
    right_stick = (axes[2], axes[3]) if len(axes) > 3 else (0.0, 0.0)
    return left_stick, right_stick

def collect_data(scene):
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "joystick1" not in obj:
        obj["joystick1"], obj["joystick2"] = init_joysticks()
    
    joystick1 = obj["joystick1"]
    joystick2 = obj["joystick2"]
    
    if joystick1:
        left_stick1, right_stick1 = read_joystick(joystick1)
        # Record the data
        with open('joystick_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([left_stick1[0], left_stick1[1], right_stick1[0], right_stick1[1]])

# Run this function to collect data
collect_data(bpy.context.scene)

# Training the model
def train_model():
    # Load data
    data = pd.read_csv('joystick_data.csv', header=None)
    inputs = data[[0, 1, 2, 3]].values
    outputs = data[[0, 1, 2, 3]].values  # Adjust as needed for desired outputs

    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(inputs, outputs, epochs=10)

    # Save the model
    model.save('joystick_model.h5')

# Run this function to train the model
train_model()

def apply_model(scene):
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "joystick1" not in obj:
        obj["joystick1"], obj["joystick2"] = init_joysticks()
    
    joystick1 = obj["joystick1"]
    joystick2 = obj["joystick2"]
    
    if "model" not in obj:
        obj["model"] = tf.keras.models.load_model('joystick_model.h5')
    
    model = obj["model"]
    
    if joystick1:
        left_stick1, right_stick1 = read_joystick(joystick1)
        inputs = [[left_stick1[0], left_stick1[1], right_stick1[0], right_stick1[1]]]
        adjusted_inputs = model.predict(inputs)[0]
        
        move_x1, move_y1, look_x1, look_y1 = adjusted_inputs
        
        obj.location.x += move_x1 * 0.1
        obj.location.y += move_y1 * 0.1

    if joystick2:
        left_stick2, right_stick2 = read_joystick(joystick2)
        inputs = [[left_stick2[0], left_stick2[1], right_stick2[0], right_stick2[1]]]
        adjusted_inputs = model.predict(inputs)[0]
        
        move_x2, move_y2, look_x2, look_y2 = adjusted_inputs
        
        obj.location.x += move_x2 * 0.1
        obj.location.y += move_y2 * 0.1

# Ensure this script runs continuously in the game engine
bpy.app.handlers.frame_change_post.append(apply_model)
