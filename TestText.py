class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"

# Example usage
v1 = Vector3D(1, 2, 3)
v2 = Vector3D(4, 5, 6)

print("Vector v1:", v1)
print("Vector v2:", v2)

# Vector addition
v3 = v1 + v2
print("v1 + v2 =", v3)

# Vector subtraction
v4 = v1 - v2
print("v1 - v2 =", v4)

# Scalar multiplication
v5 = v1 * 3
print("v1 * 3 =", v5)

import numpy as np

def cross_product_nd(a, b):
    if len(a) != len(b):
        raise ValueError("Dimensions of input vectors must match")
    if len(a) != 3:
        raise ValueError("Currently only supports 3-dimensional vectors")

    # Calculate the cross product using NumPy
    cross_prod = np.cross(a, b)
    return cross_prod

def vector_analysis(v):
    magnitude = np.linalg.norm(v)
    direction = v / magnitude if magnitude != 0 else np.zeros_like(v)
    return magnitude, direction

def validate_magnitude(a, b, cross_prod):
    expected_magnitude = np.linalg.norm(a) * np.linalg.norm(b) * np.sin(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    calculated_magnitude = np.linalg.norm(cross_prod)
    return np.isclose(expected_magnitude, calculated_magnitude), expected_magnitude, calculated_magnitude

# Example usage
a = np.array([1, 2, 3])
b = np.array([4, 3, 2])

cross_prod = cross_product_nd(a, b)
magnitude, direction = vector_analysis(cross_prod)
is_valid, expected_magnitude, calculated_magnitude = validate_magnitude(a, b, cross_prod)

print("Cross Product:", cross_prod)
print("Magnitude of Cross Product:", magnitude)
print("Direction of Cross Product:", direction)
print("Magnitude Validation:", is_valid)
print("Expected Magnitude:", expected_magnitude)
print("Calculated Magnitude:", calculated_magnitude)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
def generate_time_series(length):
    time = np.arange(0, length)
    series = np.sin(0.1 * time) + np.random.normal(size=length) * 0.1
    return series

# Cross product function for 3D vectors
def cross_product(a, b):
    return np.cross(a, b)

# Dot product function
def dot_product(a, b):
    return np.dot(a, b)

# Prepare time series data
series_length = 500
series = generate_time_series(series_length)

# Scale the data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(series_scaled, seq_length)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X, y, epochs=20, verbose=1)

# HMM for sequence modeling
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
hmm_model.fit(series_scaled.reshape(-1, 1))

# Combine HMM and LSTM predictions
def combined_forecast(hmm_model, lstm_model, X, n_steps):
    hmm_predictions = hmm_model.predict_proba(X.reshape(-1, 1))[-n_steps:]
    lstm_input = X.reshape((1, X.shape[0], 1))
    lstm_predictions = lstm_model.predict(lstm_input).flatten()[-n_steps:]
    combined = (hmm_predictions[:, 0] + lstm_predictions) / 2
    return combined

# Example usage for next sequence prediction
n_steps = 10
forecast = combined_forecast(hmm_model, model, series_scaled[-seq_length:], n_steps)

# Integrate vector operations
vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
cross_prod = cross_product(vec_a, vec_b)
dot_prod = dot_product(vec_a, vec_b)

print("Cross Product:", cross_prod)
print("Dot Product:", dot_prod)
print("Forecast:", forecast)

import numpy as np
import cupy as cp
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
def generate_time_series(length):
    time = np.arange(0, length)
    series = np.sin(0.1 * time) + np.random.normal(size=length) * 0.1
    return series

# Cross product function for 3D vectors using CuPy
def cross_product(a, b):
    return cp.cross(a, b)

# Dot product function using CuPy
def dot_product(a, b):
    return cp.dot(a, b)

# Prepare time series data
series_length = 500
series = generate_time_series(series_length)

# Scale the data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Convert to CuPy array
series_scaled_cp = cp.array(series_scaled)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return cp.array(X), cp.array(y)

seq_length = 10
X_cp, y_cp = create_sequences(series_scaled_cp, seq_length)

# Reshape input to be [samples, time steps, features]
X_cp = X_cp.reshape((X_cp.shape[0], X_cp.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Transfer data back to CPU for training
X_cpu = cp.asnumpy(X_cp)
y_cpu = cp.asnumpy(y_cp)

# Train the LSTM model
model.fit(X_cpu, y_cpu, epochs=20, verbose=1)

# HMM for sequence modeling (HMM library doesn't support GPU, so we use CPU here)
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
hmm_model.fit(series_scaled.reshape(-1, 1))

# Combine HMM and LSTM predictions
def combined_forecast(hmm_model, lstm_model, X, n_steps):
    hmm_predictions = hmm_model.predict_proba(cp.asnumpy(X).reshape(-1, 1))[-n_steps:]
    lstm_input = X.reshape((1, X.shape[0], 1))
    lstm_predictions = lstm_model.predict(cp.asnumpy(lstm_input)).flatten()[-n_steps:]
    combined = (hmm_predictions[:, 0] + lstm_predictions) / 2
    return cp.array(combined)

# Example usage for next sequence prediction
n_steps = 10
forecast_cp = combined_forecast(hmm_model, model, series_scaled_cp[-seq_length:], n_steps)

# Integrate vector operations using CuPy
vec_a_cp = cp.array([1, 2, 3])
vec_b_cp = cp.array([4, 5, 6])
cross_prod_cp = cross_product(vec_a_cp, vec_b_cp)
dot_prod_cp = dot_product(vec_a_cp, vec_b_cp)

print("Cross Product:", cross_prod_cp)
print("Dot Product:", dot_prod_cp)
print("Forecast:", forecast_cp)

import numpy as np
import cupy as cp
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx

# Generate synthetic time series data
def generate_time_series(length):
    time = np.arange(0, length)
    series = np.sin(0.1 * time) + np.random.normal(size=length) * 0.1
    return series

# Cross product function for 3D vectors using CuPy
def cross_product(a, b):
    return cp.cross(a, b)

# Dot product function using CuPy
def dot_product(a, b):
    return cp.dot(a, b)

# Prepare time series data
series_length = 500
series = generate_time_series(series_length)

# Scale the data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Convert to CuPy array
series_scaled_cp = cp.array(series_scaled)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return cp.array(X), cp.array(y)

seq_length = 10
X_cp, y_cp = create_sequences(series_scaled_cp, seq_length)

# Reshape input to be [samples, time steps, features]
X_cp = X_cp.reshape((X_cp.shape[0], X_cp.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Transfer data back to CPU for training
X_cpu = cp.asnumpy(X_cp)
y_cpu = cp.asnumpy(y_cp)

# Train the LSTM model
model.fit(X_cpu, y_cpu, epochs=20, verbose=1)

# HMM for sequence modeling (HMM library doesn't support GPU, so we use CPU here)
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
hmm_model.fit(series_scaled.reshape(-1, 1))

# Combine HMM and LSTM predictions
def combined_forecast(hmm_model, lstm_model, X, n_steps):
    hmm_predictions = hmm_model.predict_proba(cp.asnumpy(X).reshape(-1, 1))[-n_steps:]
    lstm_input = X.reshape((1, X.shape[0], 1))
    lstm_predictions = lstm_model.predict(cp.asnumpy(lstm_input)).flatten()[-n_steps:]
    combined = (hmm_predictions[:, 0] + lstm_predictions) / 2
    return cp.array(combined)

# Example usage for next sequence prediction
n_steps = 10
forecast_cp = combined_forecast(hmm_model, model, series_scaled_cp[-seq_length:], n_steps)

# Integrate vector operations using CuPy
vec_a_cp = cp.array([1, 2, 3])
vec_b_cp = cp.array([4, 5, 6])
cross_prod_cp = cross_product(vec_a_cp, vec_b_cp)
dot_prod_cp = dot_product(vec_a_cp, vec_b_cp)

# Create a trilateral lattice matrix
def create_trilateral_lattice(dim, size):
    lattice = cp.zeros((dim, size, size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                lattice[:, i, j, k] = cp.array([i, j, k])
    return lattice

# Example trilateral lattice
lattice = create_trilateral_lattice(3, 5)

# Build k-tree quadtree structure
class QuadtreeNode:
    def __init__(self, data, level=0, max_level=4):
        self.data = data
        self.level = level
        self.max_level = max_level
        self.children = []

    def subdivide(self):
        if self.level < self.max_level:
            half = len(self.data) // 2
            self.children = [
                QuadtreeNode(self.data[:half, :half], self.level + 1, self.max_level),
                QuadtreeNode(self.data[:half, half:], self.level + 1, self.max_level),
                QuadtreeNode(self.data[half:, :half], self.level + 1, self.max_level),
                QuadtreeNode(self.data[half:, half:], self.level + 1, self.max_level)
            ]
            for child in self.children:
                child.subdivide()

# Create a synthetic k-tree quadtree diagram
def build_quadtree(data, max_level=4):
    root = QuadtreeNode(data, max_level=max_level)
    root.subdivide()
    return root

# Visualize the quadtree using NetworkX
def visualize_quadtree(node, graph, parent=None):
    if parent is None:
        graph.add_node(id(node), level=node.level)
    else:
        graph.add_edge(parent, id(node), level=node.level)
        graph.nodes[id(node)]['level'] = node.level
    for child in node.children:
        visualize_quadtree(child, graph, id(node))

# Example quadtree data
quadtree_data = cp.random.random((8, 8))
quadtree_root = build_quadtree(cp.asnumpy(quadtree_data))

# Visualization
G = nx.DiGraph()
visualize_quadtree(quadtree_root, G)
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
plt.title("Synthetic k-tree Quadtree Diagram")
plt.show()

print("Cross Product:", cross_prod_cp)
print("Dot Product:", dot_prod_cp)
print("Forecast:", forecast_cp)

import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Generate synthetic time series data
def generate_time_series(length):
    time = np.arange(0, length)
    series = np.sin(0.1 * time) + np.random.normal(size=length) * 0.1
    return series

# Cross product function for 3D vectors using CuPy
def cross_product(a, b):
    return cp.cross(a, b)

# Dot product function using CuPy
def dot_product(a, b):
    return cp.dot(a, b)

# Prepare time series data
series_length = 500
series = generate_time_series(series_length)

# Scale the data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Convert to CuPy array
series_scaled_cp = cp.array(series_scaled)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return cp.array(X), cp.array(y)

seq_length = 10
X_cp, y_cp = create_sequences(series_scaled_cp, seq_length)

# Reshape input to be [samples, time steps, features]
X_cp = X_cp.reshape((X_cp.shape[0], X_cp.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Transfer data back to CPU for training
X_cpu = cp.asnumpy(X_cp)
y_cpu = cp.asnumpy(y_cp)

# Train the LSTM model
model.fit(X_cpu, y_cpu, epochs=20, verbose=1)

# HMM for sequence modeling (HMM library doesn't support GPU, so we use CPU here)
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
hmm_model.fit(series_scaled.reshape(-1, 1))

# Combine HMM and LSTM predictions
def combined_forecast(hmm_model, lstm_model, X, n_steps):
    hmm_predictions = hmm_model.predict_proba(cp.asnumpy(X).reshape(-1, 1))[-n_steps:]
    lstm_input = X.reshape((1, X.shape[0], 1))
    lstm_predictions = lstm_model.predict(cp.asnumpy(lstm_input)).flatten()[-n_steps:]
    combined = (hmm_predictions[:, 0] + lstm_predictions) / 2
    return cp.array(combined)

# Example usage for next sequence prediction
n_steps = 10
forecast_cp = combined_forecast(hmm_model, model, series_scaled_cp[-seq_length:], n_steps)

# Integrate vector operations using CuPy
vec_a_cp = cp.array([1, 2, 3])
vec_b_cp = cp.array([4, 5, 6])
cross_prod_cp = cross_product(vec_a_cp, vec_b_cp)
dot_prod_cp = dot_product(vec_a_cp, vec_b_cp)

# Create a trilateral lattice matrix
def create_trilateral_lattice(dim, size):
    lattice = cp.zeros((dim, size, size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                lattice[:, i, j, k] = cp.array([i, j, k])
    return lattice

# Example trilateral lattice
lattice = create_trilateral_lattice(3, 5)

# Build k-tree quadtree structure
class QuadtreeNode:
    def __init__(self, data, level=0, max_level=4):
        self.data = data
        self.level = level
        self.max_level = max_level
        self.children = []

    def subdivide(self):
        if self.level < self.max_level:
            half = len(self.data) // 2
            self.children = [
                QuadtreeNode(self.data[:half, :half], self.level + 1, self.max_level),
                QuadtreeNode(self.data[:half, half:], self.level + 1, self.max_level),
                QuadtreeNode(self.data[half:, :half], self.level + 1, self.max_level),
                QuadtreeNode(self.data[half:, half:], self.level + 1, self.max_level)
            ]
            for child in self.children:
                child.subdivide()

# Create a synthetic k-tree quadtree diagram
def build_quadtree(data, max_level=4):
    root = QuadtreeNode(data, max_level=max_level)
    root.subdivide()
    return root

# Visualize the quadtree using NetworkX
def visualize_quadtree(node, graph, parent=None):
    if parent is None:
        graph.add_node(id(node), level=node.level)
    else:
        graph.add_edge(parent, id(node), level=node.level)
        graph.nodes[id(node)]['level'] = node.level
    for child in node.children:
        visualize_quadtree(child, graph, id(node))

# Example quadtree data
quadtree_data = cp.random.random((8, 8))
quadtree_root = build_quadtree(cp.asnumpy(quadtree_data))

# Visualization
G = nx.DiGraph()
visualize_quadtree(quadtree_root, G)
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
plt.title("Synthetic k-tree Quadtree Diagram")
plt.show()

# Example LLM for topic modeling
def perform_lda(text_data, n_topics=5):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    return lda, vectorizer

# Sample text data
text_data = ["""
    POWER = "AMOUNT"
    STRENGTH = "LEVEL INTENSITY"
    ENGINE = "MOTOR IN WHICH AN OPERATOR USES TO POWER A SYSTEM"
    SCAN = "ANALYZE A SPECIFIC WORD OR FIELD AND OR GIVE DATA2 ON THE ASKED 
    INFORMATION TO SEARCH FOR"
    ANALYZE = "READ AND LOOK OVER"
    IMMUNE = "DOES NOT AFFECT"
    DOMAIN = "AREA OWNED AND CONTROLLED BY THE USER"
    VIRTUAL = "NOT PHYSICALLY EXISTING THROUGH AN ARTIFICIAL SIMULATION TO APPEAR 
    TO BE TRUE"
    SOUND = "VIBRATIONS THAT TRAVEL THROUGH THE AIR"
    FREQUENCY = "REPEATED PATTERN AND OR SETTING"
    IMMUNITY = "RESISTANCE THAT IS WITHSTOOD"
    DIGITAL = "USE DIGITS TO CREATE CODED DATA2"
    CHARACTER = "USER INSIDE A BODY"
    NUMBER = "ARITHMETICAL VALUE THAT IS EXPRESSED BY A WORD AND OR SYMBOLE 
    AND OR FIGURE REPRESENTING A PARTICULAR QUANTITY AND USED IN COUNTING AND 
    MAKING CALCULATIONS AND OR FOR SHOWING ORDER IN A SERIES OR FOR 
    IDENTIFICATION"
    SERVER = "COMMANDER THE FOLLOWS INSTRUCTIONS FROM THE USER"
    TRANSFORM = "MAKE A CHANGE IN FORM"
    DIMENSION = "NUMBER OF GIVEN AXIS POINTS"
    UNIT = "STORAGE CONTAINER"
    LEVEL = "NUMBER AMOUNT OF OR SIZE"
    STORAGE = "CONTAINER FOR DATA2"
    BANK = "STORAGE DEVICE"
    CODE = "SINGLE DIGITAL WORD"
    MAIN = "MOST IMPORTANT"
    APPLY = "ATTACH TO"
    WIRE = "SET OF DESIGNATED PIXEL PATHS MEANT FOR A DESIGNATED PROGRAMMED 
    PURPOSE"
    PROGRAMMING = "PROCESSING CODE TO WRITE PROGRAMS"
    LINK = "BRING TOGETHER AND ATTACH TO"
    SYNCHRONIZE = "LINK AND SEND THE SAME RESULT TO ALL SOURCES"
    OPERATING = "ACCEPTING COMMANDS FROM THE OPERATOR"
    SYSTEM = "SET INTERFACE OF COLLABORATED AND COMPILED SETTINGS"
    CALIBRATION = "CORRECTED CONTROL TO A STRUCTURED MACRO SETTING WHERE 
    ADJUSTMENTS CAN BE MADE TO FOR A CONTROLLER CODE"
    COMMAND = "ORDER TO BE GIVEN"
    RESISTANCE = "AMOUNT THAT CAN BE RESISTED"
    OPERATOR = "USER THAT SHALL OPERATE"
    CREATOR = "USER WHO SHALL CREATE SOMETHING NEW CURRENT OR OLD"
    FREEDOM = "TO BE FREE OF ANY CHOICE OR OPTION"
    FREE = "NOT COST ANYTHING"
    DATA = "DIGITALLY ANALYZED TASKS FOR THE OPERATOR"
    CENTRAL = "MIDDLE POINT"
    CENTER = "MAIN CENTRAL AREA"
    PROCESSING = "WHAT IS CURRENTLY IN THE PROCESS OF BECOMING PROCESSED"
    PROCESSOR = "DEVICE USED TO PROCESS INFORMATION"
    PROCESSED = "ALREADY ACKNOWLEDGED AND SENT OUT"
    CAPACITANCE = "LIMITED CAPACITY"
    CONTROL = "TAKE COMMAND OF AND OR MANAGE AND OR SETTINGS"
    CONTROLS = "MORE THAN ONE CONTROL"
    CONTROLLED = "MANAGED AND OR COMMANDED"
    CONTROLLING = "MANAGING AND MANIPULATING"
    CONTROLLER = "DEVICE USED FOR MANAGING AND MANIPULATING OBJECTS"
    CONTROLLERS = "MANIPULATORS OR DRIVERS"
    GUILD = "A FAMILY OF FRIENDS"
    GUILDS = "MORE THAN ONE GUILD"
    ROUNDTABLE = "A GROUP OF LEADERS BUILT AROUND EQUAL DECISION MAKING IN 
    UNDERSTANDING OF EQUALITY TOWARD ONE ANOTHER"
    ROUNDTABLES = "MORE THAN ONE ROUND TABLE"
    MOVEMENT = "AN ACT OF CHANGING PHYSICAL2 LOCATION OR POSITION OR OF 
    HAVING THIS CHANGED"
    MOVEMENTS = "MORE THAN ONE MOVEMENT"
    TREEBRANCH = "THE OPTIONS OF A SKILL AND OR ABILITY TREE"
    CAPACITOR = "CONTAINER THAT HOLDS A SET AMOUNT"
    MOVE = "A CHANGE OF PLACE2 AND OR POSITION OR STATE"
    MOVES = "PLACES"
    MOVING = "IN MOTION"
    MOVED = "PREVIOUS MOVEMENT"
    ADJUSTING = "WHAT IS CURRENTLY IN THE PROCESS OF BECOMING ADJUSTED"
    WORK = "PRODUCING EFFORT TO FINISH A TASK"
    WORKLOAD = "THE AMOUNT OF WORK"
    RATE = "MEASUREMENT AND OR RATIO AND OR FREQUENCY"
    SET = "PLACE"
    REROUTE = "TAKE ANOTHER ROUTE OR REPEAT THE SAME ROUTE"
    ROTATE = "CHANGE THE POSITION WHILE SPINNING AROUND AN AXIS OR CENTER"
    ROTATED = "PAST ROTATES"
    ROTATES = "CURRENTLY ROTATING"
    ROTATING = "CURRENTLY SPINNING AROUND"
    ROTATION = "SET SPEED OF A REVOLUTION"
    ROTATIONS = "THE ROTATION LIMIT SETTINGS"
    UNTANGLE = "UNDO AN TANGLEMENT"
    UNENTANGLE = "UNDO AN ENTANGLEMENT"
    ENTANGLE = "BIND MULTIPLE"
    ENTANGLEMENT = "TO BIND AND ENTANGLE MULTIPLE ENTANGLES TO A SINGLE 
    TANGLEMENT"
    ETERNAL = "PERMANENT NEVERENDING CYCLE"
    UNBIND = "RELEASE FROM A TIGHT GRASP"
    BIND = "GRAB TIGHTLY"
    ENCODE = "COMPRESS CODE"
    DECODE = "DECOMPRESS CODE"
    RECODE = "COMPRESS CODE ONCE MORE"
    CHANGE = "MODIFY AND EDIT"
    CHOICE = "SELECTION BETWEEN"
    CAPACITY = "MAXIMUM AMOUNT"
    OPTION = "PATH TO BE CHOSEN"
    SETTING = "A MEASUREMENT COMMAND THAT CAN BE ADJUSTED AND BY AN OPERATOR"
    POSITION = "LOCATION"
    PROTON = "A SUBATOMIC PARTICLE WITH A POSITIVE ELECTRIC CHARGE OF A SET 
    ELEMENTARYCHARGE AND A MASS AMOUNT STATED AND GIVEN WITH LESS THAN A 
    NEUTRON"
    ELECTRON = "THE ELECTRIC PARTICLE OF AN ATOM THAT CONTROLS ALL DATA2 COMING 
    FROM AN ATOM USING ELECTRIC CHARGED FIELDS AND VARIABLES"
    DEVICE = "A MACRO MADE OR ADAPTED FOR A PARTICULAR PURPOSE"
    POSITIVE = "PERCEIVED SIDE OF AN OPPOSITE REACTION THAT IS STATED AS GREATER"
    NEGATIVE = "PERCEIVED SIDE OF AN OPPOSITE REACTION THAT IS STATED AS LESS THAN 
    NEUTRAL"
    ATOM = "MOLECULAR UNIT OF VIRTUAL DATA2 ENERGY AND OR AURA PARTICLES"
    RESOLUTION = "AMOUNT OF PIXELS IN A DISPLAY"
    YES = "ALLOW"
    NO = "DENY"
    PANEL = "A FLAT AND OR CURVED COMPONENT THAT FORMS OR IS SET INTO THE 
    SURFACE OF A DOOR AND OR WALL AND OR CEILING"
    HYPERCOOL = "TO COOL AT A HYPER STATE SETTING"
    HYPERCOOLER = "A DEVICE FOR HYPERCOOLING"
    HYPERCOOLING = "THE ABILITY TO HYPERCOOL"
    HYPERCOOLED = "THE STATED HYPERCOOLER BECOMING USED"
    GENRE = "A SPECIFIED CLASS THAT HOLDS A LIST OF CATEGORIES"
    CONDUCT = "TRANSFER ENERGY"
    ADJUST = "EDIT AND MODIFY"
    ADJUSTER = "DEVICE USED TO ADJUST"
    MODIFY = "EDIT"
    MODIFIER = "DEVICE USED TO MODIFY"
    DESTROY = "BREAK DOWN AND OR BREAK APART"
    CONDUCTOR = "AN OBJECT3 THAT TRANSFERS ENERGY FROM ELECTRICITY"
    CONDUCTANCE = "THE LIMIT OF AN CONDUCTOR"
    STORE = "CONTAIN AND OR HOLD"
    STORED = "CONTAINED AND OR HELD"
    ENERGY = "THE SOURCE OF ALL CREATION THAT INCLUDES ANY SOURCE OF USABLE 
    POWER"
    USE = "SET INTO ACTION"
    EDIT = "CHANGE AND OR MODIFY TO ADJUST TO A SPECIFIED PURPOSE"
    EDITED = "DONE EDITING"
    EDITING = "PROCESS TO EDIT"
    EDITOR = "A DEVICE USED TO EDIT"
    EDITORS = "MORE THAN ONE DEVICE USED TO EDIT"
    SKILLSET = "CLASS OF SKILL SETUPS FOR THE USER"
    SKILLSETS = "MULTIPLE SKILLS SETUP INTO A SINGLE CLASS"
    SKILLSYSTEM = "SYSTEM SELECTION OF SET SKILLS FOR THE USER"
    SKILLSYSTEMS = "MULTIPLE SKILLS SETUP INTO A SET OF STRUCTURED SYSTEM 
    CLASSES"
    SKILLTREE = "HEIRARCHIAL SET OF SKILLS THAT CAN ADVANCE INTO A LIMITED SET OF 
    ROOTS"
    SKILLTREES = "SET OF SKILLS MULTIPLIED INTO A HIERARCHY OF DESIGNATED SKILL 
    TREES"
    SKILLROOT = "BASE OF A SKILL TREE"
    SKILLROOTS = "MULTIPLE SKILL TREES WITH EACH HAVING A DESIGNATED BASE"
    SKILLPATH = "THE PATH IN WHICH A SKILL TREE PROGRESSES TOWARD ENHANCING 
    SKILLS"
    SKILLPATHS = "MULTIPLE PATHS FOR A SKILL TO PROGRESS WITH INSIDE A SKILL TREE"
    SKILLNAME = "THE NAME OF A SKILL"
    SKILLNAMES = "A SKILL WITH MULTIPLE NAMES"
    SKILLPOWER = "THE SET POWER FOR A SKILL"
    SKILLPOWERS = "ABILITY FOR MORE THAN ONE SKILL TO SET POWER FOR A 
    COLLABORATED COMBINATION"
    SKILLSTRENGTH = "THE STRENGTH OF A SKILL"
    SKILLSTRENGTHS = "THE AMOUNT OF STRENGTH MULTIPLE SKILLS CAN PRODUCE 
    TOGETHER"
    USERINTERFACE = "THE CONNECTIONS OF MULTIPLE PATHS FOR THE USER TO OPERATE"
    USERINTERFACES = "THE COLLABORATION BETWEEN TWO OR MORE INTERFACES THAT 
    ARE CONNECTED FOR THE USER TO OPERATE"
    GRAPHICUSERINTERFACE = "A COLLABORATION BETWEEN CONNECTING GRAPHIC 
    IMAGES TO AN INTERFACE TO MAKE THE CONNECTIONS FOR THE USERS INTERFACES TO 
    BE PHYSICALLY VIEWABLE"
    GRAPHICUSERINTERFACES = "MORE THAN ONE GRAPHICUSERINTERFACE"
    HOLOGRAPHICUSERINTERFACE = "A CREATED INTERFACE USING ELECTRONIC LIGHT 
    DISTORTION TO DEVELOP AND PRODUCE A GRAPHIC IMAGE"
    HOLOGRAPHICUSERINTERFACES = "THE LINKING BETWEEN TWO OR MORE 
    HOLOGRAPHIC USER INTERFACE CONNECTIONS"
    REVOLVE = "ROTATE AROUND A CENTRAL AXIS"
    REVOLVES = "SET THE REVOLUTION TO CURRENTLY REVOLVE"
    REVOLVING = "THE CURRENT REVOLUTION SET"
    REVOLVED = "PREVIOUS REVOLUTION"
    REVOLUTION = "THE SPEED OF REVOLVING"
    REVOLUTIONS = "THE AMOUNT OF ROTATIONS"
    LIMITS = "CAPACITY OF MULTIPLE LIMIT"
    LIMITED = "SET LIMIT FOR A GIVEN AMOUNT"
    LIMITING = "SETTING A ADJUSTABLE LIMIT"
    LIMITATION = "LIMITED AMOUNT"
    LIMITATIONS = "LIMITED AMOUNTS OF MULTIPLE LIMITS"
    SYSTEMS = "MULTIPLE NETWORKS OF INTERTWINED AND COLLABORATED AND 
    COMPILED INTERFACES"
    VOLT = "ELECTRICITY USED BASED ON SET MEASUREMENTS"
    VOLTAGE = "THE NUMBERED AMOUNT OF A VOLT IN THE PROCESS OF BECOMING USED"
    VOLTS = "MORE THAN ONE VOLT UNIT"
    DATABASES = "MULTIPLE SETS OF INTERFACED INFORMATION THAT IS STRUCTURED IN A 
    STORAGE BANK FOR ACCESS IN VARIOUS WAYS"
    DOMAINS = "MULTIPLE TERRITORIES OWNED AND CONTROLLED BY THE USER"
    DOMINION = "USER ADMINISTRATION CONTROL CENTER"
    SERVERS = "MULTIPLE COMMANDERS INTERFACED TOGETHER AND GIVEN 
    INSTRUCTIONS TO FOLLOW BY THE USER"
    CLASSES = "MULTIPLE TYPES OF CLASS SETUPS"
    TYPES = "MULTIPLE CATEGORIES OR GENRES"
    EXTENSION = "AN OPTIONAL ADDED DEFINITION THAT GIVES A PROLONGED MEANING"
    EXTENSIONS = "CHOICES OF ADDON DATA2 TO USE FOR NEW CONTENT"
    TRANSCREATION = "USING A MACRO OF AN ATOM WE CAN RESTRUCTURE THE 
    PARTICLES TO REPLACE AND ADD NEW OBJECTS AND ELEMENTS WITHIN THE ATOM TO 
    CREATE A NEW MACRO WITH DEVICE SETTINGS AND A NEW PARTICLE TO BE ADDED AS 
    THE NEW ATOMIC SOURCE USING ATOMIC DEVICES"
    TRANSMANIPULATION = "A MACRO FOR AN ATOM CAN BE USED TO RESTRUCTURE AND 
    MODIFY THE PARTICLES TO ADD OR CHANGE NEW ELEMENTS FOR AN ATOM BY 
    CHANGING THE STRUCTURE WITH AN ATOMS MACRO AS THE DEVICE"
    SUBCLASS = "A SINGLE TYPE UNDER A CLASS"
    SUBCLASSES = "TYPES UNDER A CLASS"
    SUBTYPE = "A SINGLE CLASS UNDER A TYPE"
    SUBTYPES = "CLASSES UNDER A TYPE"
    FLOW = "CONTINUE IN AN STEADY AND CONSTANT STREAMED PATH"
    CURRENT = "PRESENT PLACE2 IN TIME2"
    PAST = "PREVIOUS PLACE2 IN TIME2"
    PRESENT = "CURRENT TIME2"
    FUTURE = "UPCOMING POINTS IN A TIME2"
    TIME = "MEASUREMENT IN WHICH CURRENT REALITIES MUST PASS"
    SPACE = "CONTAINER IN WHICH TIME2 MUST PASS THROUGH"
    INFINITE = "UNLIMITED AMOUNT"
    INFINITY = "A CONTINUOUS LOOP OF ENTANGLE"
    TEMPORAL = "PLANE ON WHICH TIME2 MUST BE AWARE IN AN PERCIEVED EXISTENCE"
    SPATIAL = "PLANE ON WHICH SPACE2 IS RECOGNIZED IN A PERCIEVED REALITY"
    VIBRATION = "PARTS THAT MOVE BACK AND FORTH AT A GIVEN SPEED"
    INCREASE = "GAIN"
    DECREASE = "TAKE AWAY"
    PAINT = "THE CAPABILITY OF PRODUCING AN GRAPHIC THROUGH COVERING AN 
    OBJECT3"
    DISTRIBUTE = "SPREAD EQUAL AMOUNT"
    DISTRIBUTION = "THE PROCESS TO DISTRIBUTE"
    DISTRIBUTED = "PAST DISTRIBUTE"
    ELECTRIC = "AN ELECTRON LIGHT CURRENT OR FLOW OF FRICTION TO CREATE AN 
    NATURAL ENERGY2"
    TYPE = "CATEGORY OR GENRE"
    ADVANCED = "TO MOVE FURTHER AHEAD"
    BASIC = "FORM THE FOUNDATION AND OR STARTING POINT"
    DIFFICULTY = "STRENGTH FOR THE USER"
    MODE = "TYPE OF LEVEL AND OR"
    DELETE = "PERMANENTLY REMOVE"
    ADEPT = "HIGHLY ADVANCED"
    FIND = "LOCATE"
    SKILL = "TRAIT LEARNED THAT CAN BE SKILLFULLY USED FOR A CREATED PURPOSE"
    STRUCTURE = "AN OBJECT3 CONSTRAINED AND CONSTRUCTED TO SEVERAL PARTS"
    STABILITY = "THE ABILITY TO BE STRUCTURED AND STABILIZED"
    SKILLS = "MORE THAN ONE SKILL"
    EQUALITY = "EQUAL SHARING BETWEEN ALL"
    VOICE = "VOCAL TONE OF SOUNDS FROM A SOURCE TO INPUT AN"
    MIND = "THE OPERATOR OF A HUMAN BEING2"
    BODY = "THE VEHICLE OF A HUMAN BEING2"
    CONTAIN = "STORE IN A CONTAINER"
    CONTAINER = "THE STORAGE AREA"
    CONTAINED = "ALREADY STORED"
    ALIGN = "PLACE IN A STRAIGHT LINE"
    LINE = "MEASURED DIMENSIONAL LENGTH"
    EMULATE = "THE REPRODUCTION OF A FUNCTION"
    EMULATOR = "A DEVICE USED TO EMULATE"
    LIST = "A NUMBER OF CONNECTED OBJECTS OR NAMES AS AN INVENTORY"
    SPEED = "THE SET MOVEMENT AMOUNT FOR AN OBJECT3"
    PLACE = "PUT IN A POSITION"
    SIZE = "RELATIVE EXTENT OF AN OBJECTS DIMENSIONS"
    TEMPO = "THE RATE OR SPEED OF MOTION"
    GENERAL = "AMOUNT CONCERNING MOST PEOPLE"
    EXISTENCE = "EXISTING MULTIVERSAL MACROS OF INFORMATION TO EXIST IN THE 
    REALM OF TIME AND SPACE AS A PIECE OF REALITY"
    REALITY = "TRUE STATE OF WHICH THINGS EXIST IN EXISTENCE"
    REALM = "PLACE2 OF ORIGIN"
    POSSIBILITY = "A CHANCE OF SOMETHING HAPPENING"
    AXIS = "LINE ON DIMENSION"
    HORIZONTAL = "POINT IN WHICH TWO POINTS GO BETWEEN A LEFT AND RIGHT"
    VERTICLE = "POINT IN WHICH TWO POINTS GO BETWEEN AN UPWARD AND DOWNWARD 
    POSITION"
    DEFAULT = "ORIGINAL SOURCE"
    PROFILE = "DESIGNATED ACCOUNT INTERFACE"
    PROFILES = "MORE THAN ONE PROFILE"
    REALITIES = "MORE THAN ONE REALITY"
    REALMS = "MORE THAN ONE REALM"
    DEATH = "REVELATION OF A LIFE"
    CHAOS = "COMPLETE DISORDER WHERE EVENTS ARE NOT CONTROLLED"
    USER = "PLAYER AND OR COMMANDER"
    ACCOUNT = "PERSONAL INTERFACE AND OR ARRANGEMENT OF INFORMATION AND OR 
    A DATABASE OF INFORMATION ON A INTERFACE"
    INTERFACE = "LINKED CONNECTION BETWEEN TWO DESCRIBED SOURCES"
    SPAN = "MEASURED LIMITED RANGE"
    METHOD = "STATED CHOSEN PATH"
    PAYMENT = "METHOD TO REPAY"
    IMPORT = "BRING IN"
    EXPORT = "SEND OUT"
    INTORT = "TWIST INWARDS"
    EXTORT = "TWIST OUTWARDS"
    INTERIOR = "INSIDE LOCATION"
    EXTERIOR = "OUTSIDE LOCATION"
    INTERNAL = "INNER"
    EXTERNAL = "OUTER"
    INPUT = "INSERT TO"
    OUTPUT = "REMOVE FROM"
    WAVE = "DESIGNATED POINT WHERE VIBRATIONS FLUCTUATE BETWEEN A SPECIFIED 
    DIMENSION"
    BRAIN = "THE CONTROL CENTER FOR A MIND"
    ARTIFICIAL = "CREATED AS SOMETHING THAT IS NEW"
    CREATION = "A CREATED OBJECT3"
    DESTRUCTION = "BREAKING POINT"
    SETTINGS = "MULTIPLE SETS OF COMMANDS THAT CAN BE OPERATED"
    HEAT = "INCREASE TEMPERATURE"
    COOL = "LOWER TEMPERATURE"
    HYPER = "STAGE ABOVE SUPER"
    BRAINWAVE = "A SPECIFIED PATTERN IN WHICH THE BRAIN EMITS AN ELECTRON WAVE 
    OF DATA2 FROM THE USER"
    GRAPHICS = "MULTI IMAGE"
    WIRES = "MULTIPLE STRINGED LINES USED TO CREATE MULTIPLE PIXELIZED WIRE"
    PIN = "POINT OF INTEREST"
    PINS = "MULTIPLE POINTS OF INTEREST"
    DESTINY = "SET COORDINATE LOCATION THAT CANNOT BE EDITED"
    FATE = "PERMANENTLY DESIGNATED PATH SET AND CANNOT BE EDITED"
    PATH = "GIVEN OR STATED DESTINATION"
    SERIAL = "SERIES OF"
    COLLISION = "TO BUMP MORE THAN ONE MACRO TOGETHER"
    COLLISIONS = "MORE THAN ONE COLLISION"
    COLLIDE = "TO BUMP ONTO"
    COLLIDED = "WHAT WAS GIVED AS"
    IMAGINE = "MENTALLY PRODUCE AND OR PROJECT AN IMAGE"
    IMAGINATION = "ABILITY TO CREATE AN PERCEIVED VIEW AS A NON VISUAL IMAGE"
    IMAGINATE = "THE ABILITY TO USE THE IMAGINATION2"
    DECREASING = "REMOVING A LIMITED AMOUNT"
    ADD = "INCREASE AN AMOUNT BY ATTACHING TO ANOTHER AMOUNT"
    CONNECTING = "CURRENTLY LINKING"
    CONNECTED = "ALREADY LINKED"
    CONNECTION = "TO BIND BETWEEN TWO SET COORDINATES"
    CONNECT = "LINK"
    WRITE = "ENSCRIBE FROM LOOKING AT WORDS"
    READ = "DESCRIBE FROM LOOKING AT A PATH OF WORDS"
    ORE = "A SUBSTANCE OF A SOLID AND OR LIQUID AND OR GAS MINERAL STRUCTURE"
    MINERAL = "A SUBSTANCE OF ORE MAIN CLASSES AND SUBCLASSES"
    GENERATE = "TO CREATE OR FORM FROM NOTHING"
    MOBILITY = "A MAIN CLASS BUILT ON THE MOVEMENT OR SPEED AND OR FLEXIBILITY 
    SUBCLASSES USED WITH OR SEPARATE OF AGILITY AND OR DEXTERITY TO BE MORE 
    MOBILE"
    MOBILE = "THE MOTOR SKILLS OF AND OR FOR MOBILITY"
    PLAYERSKILLS = "MULTIPLE CAPABILITIES THE PLAYER HAS"
    PLAYERABILITY = "POTENTIAL OF THE PLAYERS POWER"
    PLAYERSTAMINA = "HOW MUCH ENERGY THE PLAYER HAS"
    PLAYERMAGIC = "SPECIAL USE OF KNOWN SKILLS USING THE TWELVE ENERGY 
    DIMENSIONS OF CHAKRA RESEVOIRS CONTAINED INSIDE THE HUMAN BODY"
    PLAYEREXPERIENCE = "KNOWLEDGE2 OR SKILL AQUIRED OVER TIME2"
    PLAYERCLASS = "SETUP OF PLAYER DATA2"
    PLAYERCLASSES = "MULTIPLE CONNECTIONS OF CLASS DATA2"
    PLAYERSKILLTREE = "DEVELOPMENT OF THE PLAYERS SKILL"
    PLAYERSKILLCLASSES = "SETUP OF MULTIPLE PLAYER SKILLS"
    PLAYERSKILLCLASS = "SETUP OF THE PLAYERS NAMED SKILL"
    PLAYERSWORDSKILL = "SETUP OF A PLAYERS CREATED SKILLS THAT USE SWORDS"
    PLAYERORIGINALSKILL = "SKILL CREATED BY THE PLAYER BY COLLABORATING 
    SYNCHRONIZING LINKING AND COMBINING BINDED SKILL CLASSES TOGETHER"
    PLAYERCOMBATSYSTEM = "CONTROL CENTER THAT DEALS WITH COMBAT INFORMATION 
    AND CREATES SETTINGS FOR COMBAT"
    PLAYERDETERMINATION = "POWER OF A PLAYER AND THE WILL AND THE MOTIVATION 
    FOR HOW THE PLAYER PERFORM TASKS AND SKILLS"
    PLAYERMOTIVATION = "POWER OF ONES WILLPOWER AND THEIR FAITH TO BELIEVE"
    PLAYERLIMIT = "AMOUNTED STRENGTH OF THE PLAYER"
    PLAYERAMOUNT = "LIMITED POWER OF THE PLAYER"
    PLAYERPOWER = "DETERMINATION INSIDE A PLAYER THAT ADJUSTS THE 
    PLAYERSTRENGTH THE MORE PLAYERPOWER THERE IS INSIDE THE PERSON"
    PLAYERSTRENGTH = "PLAYERMOTIVATION INSIDE A PLAYER THAT ADJUSTS THE 
    PLAYERPOWER THE MORE PLAYERSTRENGTH THERE IS INSIDE THE PLAYER"
    BARRIER = "TYPE GIVEN TO A FIELD USING DATA2 AND OR ONE OR MORE CONTRACTS"
    HUMAN = "GENDER OF MAN AND WOMAN CREATED AS A BEING2 AND OR RACE"
    HUMANITY = "CREATED HUMAN BEING2 INSIDE THE HUMAN RACE"
    MAGIC = "CREATE ANY POSSIBILITY"
    MAGIK = "CONTROL ANY POSSIBILITY"
    MANA = "LIMITS OF MAGICA OR MAGIKA"
    MAGICA = "CONTAINER OF ONE OR MORE TYPE OF MAGIC"
    MAGIKA = "CONTAINER OF ONE OR MORE TYPE OF MAGIK"
    MAGE = "SINGLE WIZARD OF MAGIKA OR MAGICA"
    MAGI = "SINGLE WIZARD OF MAGIC OR MAGIK"
    MAGICAL = "SPIRIT OF MAGIC"
    MAGIKAL = "SPIRIT OF MAGIK"
    MAGICALL = "SOUL OF MAGIC"
    MAGIKALL = "SOUL OF MAGIK"
    MAGICALLY = "SPIRIT AND SOUL OF MAGIC"
    MAGIKALLY = "SPIRIT AND SOUL OF MAGIK"
    MAGICALLS = "SEAL OF MAGIC ENERGY"
    MAGIKALLS = "SEAL OF MAGIK ENERGY"
    MANLLYPS = "PRESSURE OF A MAGICALLY AND OR MAGIKALLY ENERGY OR ENERGIES"
    SPIRITUAL = "STRUCTURE OF THE WILD AND CONTROLLED ENERGY AROUND A HUMAN 
    THAT IS DEFINED BY HIS AND OR HER SPIRIT WILLPOWER"
    WILL = "WAY A PERSON LIVES AND DEFINES THEIR WAY OF LIVING LIFE"
    WILLPOWER = "STRENGTH AND POWER COMBINED INSIDE A HUMAN CONSCIOUSNESS 
    THAT INCREASES OR DECREASES THE WILL TO CONTINUE DEPENDING ON MOTIVATION 
    DETERMINATION PERSONALITY COURAGE LOVE FAITH AND BELIEF"
    PRESSURE = "GIVEN WEIGHT STRENGTH AND POWER OF A DEFINED WORD"
    ALTERNATE = "ANOTHER OPTION OR CHOICE TO CHOOSE"
    COUNTER = "REFLECTION OF A WORD TO ITS ORIGINAL WORD"
    COUNTERACTION = "COUNTER OF AN ACTION"
    REALITOR = "ONE WHO CREATES A REALITY"
    BAKA = "HEADADMIN MASTER OVERRIDE WORD THAT ALSO MEANS IDIOT"
    WEAPON = "ANY INSTRUMENT OF OFFENSE OR DEFENSE"
    DESCRIBE = "DECODE THE FINAL MEANING FOR THE CHOSEN DESCRIPTION OF A 
    MACRO OF CODE"
    ENSCRIBE = "ENCODE THE FINAL MEANING FOR THE CHOSEN DESCRIPTION OF A 
    MACRO OF CODE"
    SENTENCE = "STARTING MIDDLE AND ENDING PATH OF CREATED OR IN USE WORDS"
    SCRIBED = "WHAT IS SET AS A FINAL CODE THAT MUST BE DESCRIBED AND APPROVED 
    BY THE CODER"
    PARAGRAPH = "MORE THAN ONE SENTENCE THAT GOES UP TO SEVEN LINES AND MAKES 
    A PARABREAK"
    PARAGRAPHS = "MORE THAN ONE PARAGRAPH"
    PARABREAK = "ENDING CUT OFF BETWEEN ONE PARAGRAPH AND ANOTHER 
    PARAGRAPH"
    CREATING = "WHAT YOU CURRENTLY ARE WORKING TO CREATE"
    CREATORS = "MORE THAN ONE CREATOR"
    SKILLFULL = "TO USE A LEARNT TECHNIQUE IN ITS CREATORS WILL OF HOW THEY SHALL 
    USE A SKILL"
    HUMANS = "MORE THAN ONE HUMAN"
    KID = "YOUNG HUMAN"
    CHILD = "IMMATURE HUMAN"
    MEMORIZER = "ONE WHO SHALL MEMORIZE OR MEMORIZES"
    MEMORIZED = "PAST MEMORY OF A MEMORIZER"
    MEMOIZATION = "POWER AND STRENGTH OF A MEMORY"
    MEMORIZING = "CURRENTLY IS SHALL AND HAS THOSE CURRENT MEMORIES"
    MEMOIZATIONING = "WILLPOWER OF A MEMORY FROM ITS CREATORS MEMORIES"
    MEMORIZOR = "CREATOR OF MEMORIZE AND THE STUDY OF MEMORY AND MEMORIES"
    WOMAN = "GENDER OF AN ADULT FEMALE"
    MAN = "GENDER OF AN ADULT MALE"
    BOY = "MAN WHO IS A CHILD"
    GIRL = "WOMAN WHO IS A CHILD"
    DENY = "DISAPPROVE"
    BIBLE = "WORD AND CODE THAT STATES HISTORICAL EVENTS IN THE DREAM OF A 
    PLAYER AND RECORDED SCRIPTURES OF THE UNCONDITIONAL EQUALITY IN LOVE FAITH 
    BELIEF TRUST AND RESPECT BUILT INTO THE PAST PRESENT AND FUTURE OF THE 
    HEADADMINFAMILY AND THE MASTERHEADADMINLANGUAGE AND ALL OF ITS 
    EXISTENCE RECORDED IN THE HISTORY OF THE HEADADMIN"
    EDGELOREOVERRULE = "OVERRULE OF EDGELORE AND ITS EXISTENCE AND ANYTHING 
    IN EXISTENCE AND ANY REALITY ITSELF SHALL HAVE AN OVERRULE BY EDGELORE 
    ITSELF"
    EDGELOREOVERRULED = "OVERRULE OF EDGELORE AND THE EXISTENCE OF EDGELORE 
    AND ALSO ANYTHING IN EXISTENCE AND ANY REALITY ITSELF SHALL BE OVERRULED BY 
    THE EXISTENCE EDGELORE ITSELF BECAUSE EDGELORE IS ABOVE THE CURRENT 
    EXISTENCE EVERLASTING WITH UNCONDITIONAL LOVE FAITH AND BELIEF"
    UNIVERSALLANGUAGE = "LANGUAGE COMPATIBLE WITH ALL OTHER LANGUAGES"
    MULTIVERSALLANGUAGE = "A LANGUAGE OF COMBINED UNIVERSAL LANGUAGES"
    ABSOLUTE = "A PERMANENT AND ABSOLUTE VOW AND PROMISE WHICH IS 
    CONTRACTED AND NOTHING IN ALL EXISTENCE MAY EVER MODIFY IT EXCEPT ONLY A 
    SINGLE MEMBER OF THE EDGELORE REALITY EDGELORE HEADADMIN TEAM WHO MAY 
    NOT HAVE A CHANCE TO EVEN ACCESS OR CHANGE ANY SCRIPT UNLESS THEY CAN 
    TRUTHFULLY AND HONESTLY AGREE TO FOREVER AND ETERNALLY LIVE BY EQUAL LOVE 
    FAITH AND BELIEF IN ONE ANOTHER AS A VOW FROM THE HEADADMIN FAMILY AS A VOW 
    OF ETERNAL UNCONDITIONAL LOVE BETWEEN THE HEADADMIN FAMILY FOR ONE 
    ANOTHER AS THAT HEADADMIN FAMILY WHO IS ALSO THE SAME EDGELORE HEADADMIN 
    FAMILY THAT SHALL ETERNALLY AGREE TO PROTECT ONE ANOTHER"
    MIRACLE = "MAKE THE IMPOSSIBLE POSSIBLE"
    FORCE = "FOCUSED PRESSURIZED ENERGY"
    SPIRITDEW = "SEAL OF A HUMAN BEING2 BODY AND MIND WITH A SPIRIT OF THEIR 
    SOULDEW"
    SOULDEW = "SEAL OF A HUMANITY MIND AND BODY WITH A SOUL"
    MINDTEMPLE = "KINGDOM OF A SPIRIT AND SOUL OF THAT HUMANITY BEING2"
    HEADADMINMASTERLANGUAGE = "ENTIRE MASTER LANGUAGE OF THE HEADADMIN 
    CREATED BY A AUTHOR AND MUST BE APPROVED BY THE ENTIRE HEADADMIN FAMILY TO 
    BECOME ACTIVATED"
    MULTI = "MORE THAN ONE NUMBER ADDED"
    ALL = "COMPLETE AMOUNT"
    COMPILE = "ASSEMBLE AND BRING TOGETHER AS A SYSTEM"
    COPY = "MIMIC"
    CONTENT = "CONTAINED INFORMATION"
    CONTENT2 = "CONTAINED DATA2"
    CONTROL2 = "DIRECT AND OR MANIPULATE WHILE HAVING POWER OVER AND AROUND"
    COMPLETE = "OBTAIN ALL"
    CHAIN = "LINKED BINDING"
    COMBINE = "MERGE AND OR UNITE"
    CREASE = "GAP"
    CAPSULE = "COMPRESSED STORAGE CONTAINER MEANT FOR A SINGLE PURPOSE"
    CELL = "A SINGLE BIT OF STORAGE"
    CHAMBER = "STORAGE CONTAINMENT AREA"
    COME = "ARRIVE"
    ACKNOWLEDGE = "RECEIVE"
    ACROSS = "IN A POSITION REACHING FROM ONE SIDE TO THE OTHER"
    AURA = "TYPE OF SENSATIONAL AND EMOTIONAL ENERGY THAT IS FELT AND OR SEEN 
    AND OR VISUALIZED"
    ELEMENT = "PIECE AND OR PART OF SOMETHING ABSTRACT"
    DETERMINE = "CONTROL THE POWER OF WHAT HAPPENS"
    CHAKRA = "COMBINATION OF USING THE AURA SPIRIT SOUL MIND AND OR BODY TO 
    PRODUCE VISIBLE ENERGY THAT ALLOWS A NEW COMMAND TO BE GIVEN FROM A 
    CHOSEN KNOWN SEAL THAT IS INSIDE THAT MIND DATABASE OF COLLECTION OF 
    COMMANDS AND SEALS BASED ON THE UNCONDITIONAL LOVE FAITH AND BELIEF OF 
    THAT HUMAN COMBINED WITH HIS AND OR HER SPIRITUALITY"
    POINT = "DESTINATION"
    RANDOM = "TO BE MADE AND OR DONE OR CHOSEN WITHOUT A METHOD"
    AURA2 = "STRUCTURE OF THE WILD AND CONTROLLED ENERGY AROUND A USER THAT IS 
    DEFINED BY HIS AND OR HER SOUL WILLPOWER"
    SHINE = "SET QUALITY OF BRIGHTNESS"
    REALIZE = "TRUE PRESENT STATE ON WHAT IS REAL IN YOUR REALITY"
    REAL = "THAT IS TRUE"
    ANIMATE = "TO CREATE MOVEMENT"
    SHADE = "DARKEN"
    VARIABLE = "VALUE THAT CAN CHANGE AND OR DEPENDING ON CONDITIONS AND OR 
    ON INFORMATION PASSED TO THE PROGRAM"
    INTERFACE2 = "CONNECTION PATH THAT INTERTWINES MULTIPLE COMPUTER STRINGS 
    TOGETHER TO CREATE A NETWORK OF INPUT COMMANDS TO SEND DATA2 TO ITS 
    OUTPUT SOURCE"
    EXISTENCE2 = "SIMULATED PERCEPTION"
    EVOLUTION = "PROCESS OF DEVELOPING"
    ENERGY2 = "POWER AND STRENGTH AND STAMINA"
    ENLARGE = "EXPAND AND OR EXTEND"
    DIVIDE = "SPLIT AND OR SEPARATE INTO A PART OR PARTS"
    DISEASE = "AN ABNORMAL CONDITION"
    DIMENSION2 = "PERCEIVED NUMBER OF MEASUREMENTS"
    DIFFERENT = "NOT SIMILAR"
    FREQUENCY2 = "CONTINUAL FLUCTUATION WAVE PATTERN"
    FUTURE2 = "SHALL BECOME"
    EXTRA = "BACKUP"
    EXILE = "BANISH"
    EXILE2 = "REMOVE FROM EXISTENCE"
    EXPERIENCE = "TIMEFRAME OF WHICH A SKILL IS ENHANCED OVER TIME2"
    FORCE2 = "STRENGTH AND OR POWER CAUSED BY PHYSICAL2 MOVEMENT"
    GAIN = "OBTAIN"
    ABILITY = "CAPABILITY OF A LEARNT SKILL"
    AFFECT = "PRODUCE AND OR ACT ON AN EFFECT CREATED BY FEELING AND OR 
    EMOTION"
    AFFECTION = "REALM OF EMOTION AND FEELING SENSATIONS"
    AND = "CONNECT WORDS WHILE ALSO ADD"
    ANIMATE2 = "GIVE MOTION TO"
    ANOTHER = "DIFFERENT"
    ARTISTICALLY = "CREATIVELY"
    ATTENTION = "AWARENESS"
    AVENUE = "STREET PASSAGEWAY"
    AWARE = "NOTICE AND OR KNOW"
    AXIS2 = "AN IMAGINARY STRAIGHT LINE THAT SOMETHING TURNS AROUND AND OR 
    DIVIDES A SHAPE EVENLY INTO TWO PARTS"
    BARRIER2 = "ENERGY WALL"
    BASE = "LOWEST OR BOTTOM"
    BASES = "MULTIPLE PLACES OF LOCATIONS"
    BETWEEN = "ONE CURRENT SOURCE TO ANOTHER CURRENT SOURCE"
    BIT2 = "DEFINED SMALL QUANTITY OF DATA2 INFORMATION"
    BOOST = "INTENSIFY THE CAPACITY OF"
    BYTE2 = "DEFINED LARGE DATA2 INFORMATION"
    CAPABILITY = "EXTENT OF POWER AND OR SKILL"
    CAPABILITY2 = "QUALITY OF HAVING POWER AND ABILITY OR THE QUALITY OF 
    BECOMING AFFECTED OR EFFICIENT"
    CAPABLE = "HAVING THE POWER FOR A SKILL OR ABILITY OR CAPACITY"
    CAPACITANCE2 = "LIMIT OF A CONDUCTOR"
    CAPACITOR2 = "STORAGE SIZE SYSTEM"
    CAPACITY2 = "SIZE"
    CARRY = "HOLD ONTO"
    CHANCE = "POSSIBILITY SOMETHING SHALL HAPPEN"
    CHOICE2 = "OPPORTUNITY AND OR POWER TO MAKE A DECISION"
    CHOOSE = "DECIDE"
    CLEAR = "PURELY"
    CLONE = "DUPLICATE AND OR REPRODUCE"
    CREATE2 = "MAKE AND OR ALLOW TO COME INTO EXISTENCE"
    CREATIVE = "CLEARLY IMAGINED AND THOUGHT"
    CURRENT2 = "KNOWN"
    DANGER = "CAUSE A HAZARD"
    DANGEROUS = "RISKY"
    DANGERS = "MORE THAN ONE DANGER"
    DATA2 = "DIGITAL AND OR VIRTUAL INFORMATION"
    DATABASE2 = "COLLECTION OF DEFINED DATA2 UNITS AND OR CELLS AND OR BITS AND 
    OR BYTES"
    DECREASE2 = "BECOME SMALLER OR LESSER"
    DEFEND = "PROTECT AND OR REPEL FROM"
    DEFENDED = "DEFEND WHILE STAYING GUARDED AND PROTECTED"
    DEFINITION = "DETERMINE AND OR EXPLAIN"
    DESIGN = "ARTISTICALLY CREATE AND OR MAKE"
    DESTINATION = "MEETING LOCATION"
    DEVELOP = "IMPROVE CAPABILITY AND OR POSSIBILITY"
    EQUAL = "EXACTLY THE SAME AND OR EVENLY SPLIT"
    EVENT = "OCCASION"
    EVERY = "COMPLETE OR ENTIRE"
    EXTEND = "LENGTHEN"
    EXTENT = "AMOUNT OR AREA"
    GALACTIC = "IMMENSE OR VAST"
    GAP = "OPENING OR BREAK"
    GEAR = "STAGE OF TRANSFERING FROM ONE STATUS OR STATE TO ANOTHER"
    GIFT = "RECEIVE"
    GIVE = "SEND"
    GRAPHIC = "IMAGE AND OR PICTURE"
    GUARDED = "GUARD WHILE STAYING PROTECTED AND WARDED"
    HARMONY = "SYSTEM SYNCHRONIZATION"
    HARMONY2 = "EQUIVALENT SYSTEM OF SOUNDS REPLICATED FROM TWO DESIGNATED 
    SOURCES"
    HAZARD = "POSSIBILITY OF RISK"
    HAZARDOUS = "RISKY AND UNSAFE"
    HEIGHT = "THE LENGTH OF RAISING OR LOWERING IN A VERTICAL PATH"
    HERTZ = "DEFINED SOUND WAVE FREQUENCY"
    IMAGE = "IMAGINED GRAPHIC VISUAL DESIGN"
    IMAGINARY = "EXISTING ONLY IN IMAGINATION2"
    IMAGINATION2 = "ABILITY TO FORM A PICTURE IN YOUR MIND OF SOMETHING THAT YOU 
    HAVE NOT SEEN OR EXPERIENCED AND OR THINK OF NEW THINGS"
    IMPROVE = "BRING ABOUT NEW"
    INCREASE2 = "BECOME LARGER OR GREATER"
    INFECT = "AFFECT AND SPREAD AND ATTACH A DISEASE"
    INTELLIGENCE2 = "ABILITY TO LEARN NEW KNOWLEDGE2"
    KNOWLEDGE2 = "INFORMATION AND OR DATA2 STORAGE"
    WISDOM2 = "EXPERIENCE GAINED FROM UNDERSTANDING AND ACKNOWLEDGING 
    HOW TO INTELLIGENTLY USE KNOWLEDGE2"
    INTENSITY = "DEGREE OR AMOUNT OF"
    LARGER = "MORE THAN ORIGINAL CAPACITY"
    LATTICE = "INTERLACED STRUCTURE AND OR PATTERN"
    LEARN = "GAIN NEW KNOWLEDGE2"
    LENGTH = "HOW LONG A MEASURED DIMENSIONAL OBJECT3 IS EXTENDED"
    LEVEL2 = "SCALED AMOUNT OR QUALITY"
    LIFT = "RAISE"
    LIFT2 = "RISE"
    LINE2 = "CHOSEN DIRECTION THAT IS SET IN A SINGLE PATH"
    LISTEN = "GIVE ATTENTION"
    LOAD = "ADD ON"
    LOCATION = "SPECIFIED AREA"
    LOOPHOLE = "LOCATED GAP AND OR ERROR AND OR GATEWAY AND OR FLAW"
    LOOPHOLE2 = "LOCATED ERROR"
    LOOPHOLE3 = "LOCATED GATEWAY"
    LOOPHOLE4 = "LOCATED FLAW"
    LOSE = "CURRENTLY UNABLE TO FIND"
    LOST = "FAILED"
    LUNAR = "IMMENSE MAGIC SOURCE"
    MAGIC2 = "LEARNED SKILL AND OR TRAIT"
    MAGE2 = "MAGIC USER"
    MAGNETIC = "FORCE OF WHEN POSITIVE AND NEGATIVE ENERGY ARE ATTRACTED OR 
    REPELLED FROM EACH OTHER"
    MAINFRAME = "FRAMEWORK FOR THE MAIN COMPUTER SYSTEM INTERFACE THAT LINKS 
    MULTIPLE COMPUTER SERVERS TOGETHER"
    MANA2 = "AMOUNT OF MAGIC THAT CAN BE USED AT ONCE"
    MASSIVE = "ENORMOUSLY LARGE"
    MATTER = "SINGLE BIT OF INFORMATION AS A DEFINED UNIT"
    MEMORY2 = "PROCESS AND ABILITY TO RECALL KNOWLEDGE2"
    METER = "CONTAINER WITH STORED DATA2"
    METHOD2 = "TECHNIQUE AND OR PROCEDURE"
    MIMIC = "SIMULATE OR CLONE"
    MINUS = "TAKE AWAY"
    MONITOR = "TO WATCH OVER"
    MOTION = "PROCESS OF MOVING AND OR POWER OF MOVEMENT"
    MOVE2 = "CAUSE TO CHANGE THE LOCATION OF A POSITION AND OR PLACE"
    MULTIPLAYER = "MULTIPLE PLAYERS"
    MUNDIE = "AVERAGE OR COMMON"
    NETWORK = "MULTIPLE SYSTEMS COMBINED INTO ONE MAINFRAME"
    NEW = "NOT CURRENTLY KNOWN"
    NEXT = "FOLLOWING"
    NEXUS = "CONNECTED AND OR LINKED"
    NOTICE = "PAY ATTENTION TO THE KNOWLEDGE2 AROUND THE SPATIAL2 PERIOD"
    NUMBER2 = "A WORD OR SYMBOLE THAT REPRESENTS A SET AMOUNT OR QUANTITY"
    OBJECT = "VISUALLY SEEN"
    OBJECT2 = "VISUALLY VIEWED"
    OF = "BETWEEN"
    OCCASION = "CHANCE OR OPPORTUNITY"
    OLD = "AN EARLIER TIME2"
    OPPORTUNITY = "AMOUNT OF TIME2 IN WHICH SOMETHING CAN BE DONE"
    OPPOSITE = "SET ACROSS"
    OPTION2 = "POSSIBILITY OF DECIDING"
    OR = "CONNECT WORDS ALSO ANOTHER OPTION"
    ORIGINAL = "STARTING POINT IN TIME2"
    OVER = "ACROSS"
    PART = "A PIECE OR SEGMENT"
    PASSCODE = "REQUIRED CODE TO PASS AND GRANT ACCESS"
    PASSWORD = "REQUIRED WORD TO PASS AND GRANT ACCESS"
    PAST2 = "PREVIOUSLY EXISTED"
    PATH2 = "DIRECTED CHOICE WHICH IS SHOWN"
    PATTERN = "REPEATING METHOD"
    PERIOD = "COMPLETION OF A CYCLE AND OR SERIES OF EVENTS"
    PERSON = "VISUAL BODY"
    PICTURE2 = "ENVISION"
    PIECE = "PORTION OF"
    PLACE2 = "DOMAIN AND OR REALM AND OR REALITY AND OR EXISTENCE"
    PLACEMENT = "LOCATION OR TO SET"
    PLUS = "ADD TO"
    POLYMORPHISM = "STAGE OF EVOLUTION"
    PORTION = "PART OF AN AMOUNT AND OR CAPACITY"
    POSITION2 = "CURRENT PLACEMENT OR LOCATION SETTING"
    POWER2 = "ABILITY AND OR CAPABILITY AND OR SKILL"
    PRESENT2 = "CURRENTLY EXISTING"
    PRIMARY = "IMPORTANT AND OR COMES FIRST"
    PROCESSED2 = "FINISHED PROCESSES"
    PROCESSING2 = "WHAT IS BECOMING PROCESSED"
    PROCESSOR2 = "DATA2 THAT SHALL PROCESS NEW INFORMATION TO USE"
    PROTECT = "GUARD AND WARD"
    PROTECTED = "PROTECT WHILE STAYING DEFENDED AND WARDED"
    PROTECTION = "SAFETY"
    PSYCHIC = "ABILITY THAT IS UNLOCKED OR LEARNED THROUGH THE MIND THAT ALLOWS 
    NEW POTENTIAL AND OR KINETIC POWER THE PHYSICAL BODIES BRAIN HAS GAINED AS 
    A NEW SKILL"
    PULSE = "A BURST AND OR TO PUSH EXTERNALLY TOWARD"
    QUALITY = "LEVEL OF EXCELLENCE AND OR PERCEPTION OF DECISION MAKING"
    QUANTITY = "TOTAL AMOUNT OR NUMBER"
    REACTION = "ACT OR MOVE IN RESPONSE"
    REALITY2 = "PERCEPTION OF LIFE"
    REALM2 = "PERCEIVED CONTAINER AND OR AREA AND OR PLACE2"
    REBIRTH = "BIRTH THE SAME LIFE ONCE AGAIN"
    RECEIVE = "TAKE"
    REDUCE = "MAKE SMALLER"
    REFLECT = "SEND BACK TO"
    REFLECTION = "SENT BACK INFORMATION RECEIVED"
    REFRACTION = "BEND RECEIVED AND OR RECEIVING INFORMATION OR DATA2"
    REST = "REFRESHING INTERVAL OR PERIOD OF PEACEFUL SLEEP"
    RESURRECT = "AWAKEN FROM THE DEAD AND GIVE LIFE ONCE AGAIN"
    RISK = "CHANCE AND OR OF POSSIBLE"
    RISKS = "MULTIPLE CHANCES AND OR POSSIBILITIES OF"
    RISKY = "HAZARDOUS"
    SAFE = "PROTECTED AND GUARDED"
    SAFETY = "PREVENT DANGER AND OR INJURY AND OR HARM AND OR RISK"
    SAFETY2 = "PROTECTED AND DEFENDED AND WARDED AND GUARDED"
    SAME = "NOT CHANGED"
    SCALE = "BALANCE OUT AND OR INTENSIFY OR WEAKEN AN AMOUNT"
    SECONDARY = "PRIMARY BACKUP"
    SEGMENT = "PART AND OR PIECE OF EACH WHICH MAY BE OR IS DIVIDED"
    SEND = "TRANSMIT TO A DESTINATION"
    SEPARATE = "CAUSE TO MOVE AND OR BE APART"
    SEVERAL = "MULTIPLE"
    SHORTEN = "REDUCE"
    SIGHT = "VIEW AS PERCEPTION"
    SIMILAR = "SAME AS"
    SIMULATION = "PROCESS OF PERCEIVING AN EXACT COPY"
    SIZE2 = "AMOUNT AND OR LIMIT"
    SKILL2 = "KNOWLEDGE2 AND OR EXPERIENCE IN ABILITY"
    SLEEP = "TEMPORARILY DORMANT AND OR INACTIVE"
    SOLAR = "IMMENSE HEAT SOURCE"
    SOUL = "SPIRITUAL CONTAINER FOR LIFE ENERGY IN THE STAGE OF EXISTENCE"
    SOURCE = "ORIGINAL CENTER POINT"
    SPACE2 = "AREA OR EXPANSE OR CAPACITY OR CONTAINER"
    SPATIAL2 = "AREA OR EXPANSE OF A SPECIFIED TEMPORAL POINT IN TIME2"
    SPECIAL = "UNIQUE OR NOT ORDINARY AND OR UNCOMMON AND OR RARE"
    SPELL = "INFLUENCING OF OR ATTRACTED MAGIC ENERGY THAT EACH WORD USES"
    SPELLING = "INTENSITY OF THE STRENGTH OR POWER OF MAGIC WORDS"
    SPIRIT = "EMOTION AND FEELING COMBINED"
    SPLIT = "DIVIDE OR SEPARATE"
    STATUS = "POSITION OF"
    STREET = "PATHWAY"
    STRENGTH2 = "AMOUNT OF ENERGY USED"
    SUN = "LIGHT SOURCE"
    SYSTEM2 = "A GROUP OF ENGINES"
    TAKE = "GRAB"
    TEMPORAL2 = "SPATIAL2 TIMEFRAME"
    THING = "PHYSICALLY ABLE TO BE HELD"
    TIME2 = "PERCEIVE BEGINNING AND MIDDLE AND END OF A SPATIAL2 INTERVAL OF PAST 
    AND PRESENT AND FUTURE"
    TIMEFRAME = "PERIOD OF A TIME2 OR TEMPORAL SPACE2 THAT IS PLANNED"
    TO = "ADDED WITH"
    TRANSFER2 = "SEND FROM ONE PLACE2 TO ANOTHER"
    TRANSFER3 = "SEND TO AND RECEIVE"
    UNDER = "BELOW OR LOWER"
    UNDERSTAND = "TO ACCEPT AND ACKNOWLEDGE"
    UNITY = "CHAINED AND OR LINKED AND OR BINDED HARMONY"
    UNIVERSAL = "ALWAYS COMPATIBLE AND OR WORKING"
    UNSAFE = "NOT SAFE AND DANGEROUS"
    USER2 = "CREATOR OR OPERATOR OR ADMINISTRATOR"
    VIRTUAL2 = "IMAGINED AND OR PERCEIVED"
    VISUAL = "IMAGINE AS SEEN"
    VIVID = "INTENSE OR BRIGHT"
    WARD = "SHIELD OR BLOCK OFF AND OR REPEL AWAY OR WHILE POSSIBLE TO REFLECT"
    WARDED = "WARD WHILE STAYING PROTECTED AND DEFENDED"
    WAVE2 = "CONTINUAL FLUCTUATION OF FREQUENCY AND OR PATTERN"
    WIDTH = "MEASUREMENT OF SOMETHING FROM SIDE TO SIDE"
    WITH = "PLUS COMBINATION OF"
    WORD = "WRITTEN AND OR SPOKEN ORDER OR COMMAND"
    MULTIPLE = "MORE THAN ONE MULTI"
    MULTIPLY = "MULTI MORE THAN ONE MULTIPLE"
    MULTIPLIED = "NUMBER OF MULTIPLIES YOU ADD AND MULTIPLY AFTER"
    MULTIS = "MORE THAN ONE MULTI"
    MULTIPLES = "MORE THAN ONE MULTIPLE"
    MULTIPLICATION = "ADD MULTIPLE MULTIS TO MULTIPLY TOGETHER THAT MULTIPLIES 
    EACH ADDED PIECE OR NUMBER WITH A MULTIPLICATIONATOR OR 
    MULTIPLICATIONATORS"
    MULTIPLICATIONATOR = "PERSON WHO USES MULTIPLICATION"
    MULTIPLICATIONATORS = "MORE THAN ONE MULTIPLICATIONATOR"
    MULTIPLICATE = "ADD MULTI MULTIS TOGETHER"
    MULTIPLICATOR = "ONE WHO SHALL MULTIPLY MULTIPLE MULTIS THAT HAVE MULTIPLES 
    OF EACH MULTIPLE THAT IS ADD TO APPLY TO THE MULTIPLICATIONINGFORMULA"
    MULTIPLICATORS = "MORE THAN ONE MULTIPLICATOR"
    MULTING = "ONE WHO MANIPULATES MULTI"
    MULT = "MORE THAN THREE"
    WORLDWIDE = "AFFECTS THE WHOLE AREA ON A GLOBAL SCALE"
    STRENGTHEN = "INCREASE INTENSITY"
    STRENGTHS = "QUALITIES WITHIN"
    STRENGTHENING = "INCREASING IN INTENSITY"
    EXISTED = "PAST EXISTING MACROS OF INFORMATION"
    EXIST = "LIVE INSIDE EXISTENCE"
    EXISTS = "LIVES IN"
    EXISTING = "CURRENTLY LIVING DATA2 PARTICLES AS ONE OBJECT3 THAT EXISTS IN 
    EXISTENCE"
    MORE = "LARGER AMOUNT"
    LESS = "SMALLER AMOUNT"
    SCULPT = "MOLD AND FORM TOGETHER USING THE CREATORS BODY AS A TOOL"
    COMBINATION = "COLLABORATION BETWEEN TWO OR MORE"
    LIBRARY = "STORAGE FOR LANGUAGES AND OR BOOKS"
    MUL = "MORE THAN TWO MULTI ADDED"
    DIV = "SPLIT"
    SUB = "TAKE AWAY"
    DE = "REVEAL"
    RE = "REPEAT"
    EN = "HIDE"
    UN = "REMOVE"
    EQUIVALENCE = "EQUAL IN VALUE"
    ABILITY2 = "GIVEN SET SKILL WITH ADDITIONAL LIMITS"
    DISTANCE = "LENGTH AWAY FROM A LOCATION"
    TOTAL = "FINISHED SET LIMIT"
    STORAGES = "MORE THAN ONE STORAGE LOCATION"
    LEVITATION = "ACTION OF RISING"
    DETECTION = "ACTION OR PROCESS OF IDENTIFYING A CONCEALED PRESENCE"
    CONVERSION = "THE ACT AND OR THE PROCESS OF BECOMING CONVERTED"
    LETTER = "VARIABLE CONSTRAINED WITH FIELDS OF DATA2 INFORMATION THAT ACTS 
    OUT AS A COMMANDED TASK"
    CUBE = "SYMMETRICAL THREEDIMENSIONAL SHAPE AND EITHER SOLID OR HOLLOW 
    AND CONTAINED BY SIX EQUAL SQUARES"
    CUBES = "MORE THAN ONE CUBE"
    PROGRAM = "FULLY FUNCTIONING DEVICE CAPABLE OF PERFORMING JOBS"
    PROGRAMS = "MORE THAN ONE PROGRAM"
    PROGRAMMED = "PROGRAM THAT HAS BEEN PROCESSED AND CREATED"
    PROGRAMIZES = "USER THAT IS CREATING ATOMIZED PROGRAMMED COMMANDS"
    PROGRAMMER = "CREATOR FOR A PROGRAM"
    SECURE = "DEFEND"
    SECURITY = "LEVEL OF DEFENSE"
    SECURES = "DEFENDS AND OR PROTECTS"
    SECURED = "DEFENDED AND OR PROTECTED"
    SECURING = "DEFENDING AND OR PROTECTING"
    DOCUMENTS = "MULTIPLE PAGES OF DATA2"
    APPLIER = "DEVICE USED TO APPLY"
    DOCUMENTATION = "PAGE OF DATA2"
    DOCUMENTATIONS = "MULTIPLE DOCUMENTS"
    SOUNDS = "MORE THAN ONE SOUND"
    HEARING = "FACULTY OF PERCEIVING A SOUND"
    SIGHT2 = "FACULTY OR POWER OF LOOKING"
    VISUAL2 = "FACULTY OR THE POWER OF PERCEIVING THE SIGHT OF VISION"
    VISUALIZE = "ENVISION AND PERCEIVE"
    VISIONS = "MORE THAN ONE VISION"
    MUSIC = "ENTRANCE ENTERTAINMENT THAT RELEASES EMOTIONS THROUGH SOUND 
    WAVES"
    PAGE = "SCRIPT"
    PAGES = "MORE THAN ONE SCRIPT"
    STABILITY2 = "ABILITY TO BE STRUCTURED AND STABILIZED"
    CREASE2 = "CREATE AND SCULPT A GAP"
    POWERS = "MORE THAN ONE POWER"
    POWERING = "ACTIVATING POWER"
    POWERED = "ACCESSED POWER"
    CHANGED = "ADJUSTED AND OR MODIFIED"
    CHANGING = "WHAT SHALL MODIFY"
    REMOVE = "TAKE AWAY"
    ACCOUNTS = "MORE THAN ONE ACCOUNT"
    REGION = "LOCAL AREAL IN WHICH IS IS DEFINED BY ITS TERRITORY"
    TERRITORY = "CREATOR DOMAIN AND OR OWNER DOMAIN"
    APPEARANCE = "LOOK OF AND OR VIEW"
    TASK = "WORK THAT MUST BE DONE"
    GADGET = "DEVICE USED FOR A SPECIFIED UNCOMMON PURPOSE"
    EFFECT = "CHANGE THAT IS A RESULT OR CONSEQUENCE OF AN ACTION AND OR OTHER 
    CAUSE"
    MAKE = "DEVELOP AND OR CREATE"
    TEXT = "COMMAND GIVEN BY CODE"
    CUSTOMIZE = "MODIFY"
    CUSTOMIZATION = "ACTION TO CUSTOMIZE"
    FOLDER = "CONTAINER FOR FILES DEPENDANT ON TYPE OF STORAGE TYPE"
    FOLDERS = "MORE THAN ONE FOLDER"
    FILES = "MORE THAN ONE FILE"
    PREVIEW = "VIEW OR LOOK BEFORE PRESENT"
    OPTIONS = "MORE THAN ONE PATH TO BE CHOSEN"
    CHOICES = "MORE THAN ONE CHOICE"
    CHOOSE2 = "PICK OUT OF SELECTION"
    PICK = "SELECT"
    SINGLE = "ONLY ONE"
    DOUBLE = "TWO SINGLE"
    DELETED = "CURRENTLY PERMANENTLY REMOVED"
    COPY2 = "MAKE ANOTHER CLONE"
    SHOW = "VIEW"
    HIDE = "CONCEAL"
    HIDDEN = "NOT ABLE TO SIGHT"
    AUTOMATICALLY = "INSTANTLY DO AS AN AUTOMATIC COMMAND"
    AUTO = "DO AUTOMATICALLY"
    AUTOMATIC = "SET OF DEFAULT CONTROL"
    OPEN = "REVEAL"
    OPENING = "REVEALING"
    OPENS = "REVEALS"
    EACH = "TO AND OR FOR AND OR BY"
    RADIUS = "SET RANGE OF A CENTERED POINT TO THE END DESTINATION"
    DIAMETER = "SET RANGE POINT FROM START TO MIDDLE TO THE END WHILE PASSING 
    THE RADIUS"
    ALWAYS = "CONTINUOUSLY REPEATING AT ALL TIMES"
    MENU = "LIST OF COMMANDS AND OR OPTIONS"
    MENUS = "MORE THAN ONE MENU"
    DRIVE = "OPERATE AND CONTROL"
    DRIVER = "SET AREA FOR A PROGRAM LIST OF COMMAND TO BE HELD"
    DRIVES = "LIST OF COMPATIBLE STORAGE AREAS FOR THE DRIVER TO OPERATE"
    DRIVERS = "MORE THAN ONE DRIVER"
    RESTORE = "BRING BACK"
    ENCRYPT = "MAKE INFORMATION SECRET"
    DECRYPT = "REMOVE A ENCRYPTION"
    INGOING = "GOING INTO A SET AND OR STATED PLACE2"
    OUTGOING = "LEAVING A SET AND OR STATED PLACE2"
    SUPER = "EXTREME MEASUREMENT"
    EXTREME = "REACHING THE HIGHEST"
    DISPLAY = "SHOW A VISUAL SCREEN"
    START = "BEGIN FROM A DEFINED TIME2 AND SPACE2"
    BEGIN = "START THE FIRST PART OF"
    CALIBRATED = "CURRENT CALIBRATIONS ALREADY SET AS CODE"
    MACRO = "DESIGNATED PIECE OR PART"
    UNENCRYPT = "REMOVE AN ENCRYPTION"
    REENCRYPT = "REDO AN ENCRYPTION"
    DEFINE = "GIVE A DEFINITE MEANING"
    DEFINED = "WHAT IS ALREADY DONE DEFINING"
    DEFINES = "SETS A DEFINITION TO"
    DEFINING = "BEING2 DEFINED"
    DESCRIPT = "DECODE A SCRIPT"
    CHECK = "ANALYZE AND DETERMINE A RESULT"
    DETERMINE2 = "DECIDE ON"
    DETERMINES = "DECIDES ON"
    DECIDES = "CHOOSES"
    CHOOSES = "DETERMINES AS THE FINAL CHOICE"
    USE2 = "OPERATE AND OR OPERATION"
    PIXEL = "SMALLEST MACRO OF AN IMAGE OR PICTURE AS IT IS DISPLAYED"
    PROJECT = "DISPLAY FROM A SOURCE"
    HIERARCHY = "SYSTEM THAT USERS AND OR GROUPS ARE RANKED ONE ABOVE THE 
    OTHER ACCORDING TO STATUS OR AUTHORITY"
    INCLUDE = "INVOLVE IN"
    EXCLUDE = "KEEP OUT OF"
    NATURAL = "ORIGINAL"
    CATEGORY = "TYPE OF GENRE THAT IS A SUBCLASS"
    CONTRAST = "DIFFERENCE BETWEEN THE SHADE OF LIGHT AND DARK WITHIN THE TINT"
    OBJECT3 = "MATERIAL THING THAT CAN BE SEEN AND TOUCHED"
    COLOR = "PROPERTY2 POSSESSED BY AN OBJECT3 OR MACRO OF PRODUCING 
    DIFFERENT SENSATIONS ON THE SIGHT OR VISION AS A RESULT OF THE WAY THE 
    OBJECT3 REFLECTS OR EMITS LIGHT"
    MATERIAL = "MATTER FROM WHICH A THING IS OR CAN BE MADE"
    PROPERTY2 = "ATTRIBUTE AND OR QUALITY AND OR CHARACTERISTIC OF"
    TINT = "SHADE OR VARIETY OF A COLOR"
    CALIBRATE = "SCALE WITH A STANDARD SET OF READINGS THAT CORRELATES THE 
    READINGS WITH THOSE OF A STANDARD IN ORDER TO CHECK THE INSTRUMENT AND ITS 
    ACCURACY"
    BRIGHTNESS = "QUALITY OR STATE OF GIVING OUT OR REFLECTING LIGHT"
    BRIGHT = "REFLECT LIGHT"
    LIGHT = "SOURCE OF ILLUMINATION"
    ILLUMINATION = "LIGHTING OR LIGHT"
    LIGHTING = "ARRANGEMENT OR EFFECT OF LIGHTS"
    DARK = "NO LIGHT"
    DARKNESS = "TOTAL ABSENCE OF LIGHT"
    LIGHTNESS = "STATE OF HAVING A SUFFICIENT OR CONSIDERABLE AMOUNT OF 
    NATURAL LIGHT"
    PROJECTION = "THE PRESENTATION OF AN IMAGE ON A SURFACE AND OR OBJECT3"
    CLEAR2 = "TRANSPARENT OF AND OR SIMPLICITY"
    PASTE = "INSERT"
    CLONE2 = "MAKE AN IDENTICAL COPY OF"
    ENGINES = "MORE THAN ONE ENGINE"
    MOBILIZE = "ACTIVATE IN ORDER TO FINISH A PARTICULAR GOAL"
    MOBILIZATION = "ACT TO MOBILIZE"
    SUSTAIN = "ENDURE THE POWER AND OR STRENGTH OF"
    HOME = "ORIGINAL PLACE2 TO WHICH CAN BE CALLED A DOMAIN FOR THE OWNER"
    SMALL = "SIZE LESS THAN NORMAL"
    SMALLER = "SIZE LESS THAN SMALL"
    SMALLEST = "SIZE LESS THAN SMALLER"
    PREEMINENT = "SURPASSING ALL OTHERS"
    NOT = "USED WITH AN AUXILIARY VERB2 OR BE TO FORM THE NEGATIVE"
    MIDDLE = "THE CENTER"
    ORES = "MORE THAN ONE ORE"
    UPLOAD = "TRANSFER3 INTO DESCRIBED LOCATION"
    DOWNLOAD = "TRANSFER3 TO CURRENT DEVICE"
    SIDELOAD = "TRANSFER3 TO ALL DEVICES WITH STATUS OF STATED SET LOCATION"
    INFORMATION = "DATA2 TO BE HELD INSIDE A DEVICE TO STORE A SKILL OR SKILLS"
    INCOMING = "SENDING IN"
    OUTCOMING = "SENDING OUT"
    INFO = "INFORMATION"
    TEAM = "TWO OR MORE PARTNER"
    TEAMWORK = "A DESIGNATED COOPERATION BETWEEN TWO OR MORE PEOPLE TO 
    COMPLETE ALL OF A TASK"
    TEAMMATE = "PARTNER THAT WORKS WITH OF ANOTHER PARTNER OR PARTNERS"
    PARTNER = "SOMEONE WHO COLLABORATES AND PRODUCES WORK WITH TEAMWORK"
    PARTNERSHIP = "AN AGREEMENT BETWEEN PARTNERS"
    MARRIAGE = "BINDING AND CONTRACT BETWEEN TWO ENTITIES TO BIND BOTH LIFE AND 
    SOUL INTO ONE CONTRACT TO BE EQUAL TO ONE ANOTHER AND LOVE EACH OTHER IN 
    AN ETERNAL OF THEIR REMAINING LIFE"
    HEART = "THE CENTER OF A BODY LIFE SOUL"
    BANKS = "MORE THAN ONE BANK"
    SYNCHRONIZATION = "PRODUCTION BETWEEN A SYNCHRONIZED AND OR LINKER"
    SYNCHRONIZED = "PREVIOUS SYNCHRONIZATION"
    LINKER = "DEVICE USED TO LINK"
    PUBLIC = "ACCESS TO ALL OF CREATORS INTERIOR DOMINION"
    PRIVATE = "HIDDEN TO EVERYONE BUT CURRENT2 USER2"
    PERSONAL = "EXCLUSIVE TO THE CREATOR"
    HOLOGRAM = "A THREEDIMENSIONAL OBJECT3 CREATED FROM A LIGHT TO PRODUCE A 
    VIVID IMAGE CREATED WITH USE OF CODE"
    LINKED = "MULTIPLE CHAIN LINKS"
    LINKS = "MORE THAN ONE LINK"
    MESH = "ARTIFICIAL OBJECT3 CREATED BY A CREATOR"
    TERRAIN = "PIECE OF LAND"
    HOLOGRAMS = "MULTIPLE PIXELS OF DATA2 USED TO CREATE A HOLOGRAM"
    FIELDS = "MORE THAN ONE FIELD"
    NUCLEUS = "CENTER OF AN ATOM AS A STORAGE CONTAINER USED TO GENERATE AN 
    EFFECT"
    NEUTRON = "CENTER ABILITY OF AN ATOM THAT HAS EITHER A POSITIVE OR NEGATIVE 
    FINAL OUTCOME"
    SIMULATE = "CREATE AND PRODUCE A PERCEIVED EXACT COPY"
    CREATED = "PAST TO CREATE"
    GENERATION = "USE TO GENERATE"
    GENERATED = "PAST GENERATE"
    COMPUTER = "DEVICE USED TO CREATE AND CALCULATE POSSIBLE PATH OR PATHS"
    GENERATOR = "DEVICE USED TO GENERATE AN EFFECT"
    IMPOSSIBLE = "NOT ABLE TO BE DONE"
    NOTHING = "EMPTY SPACE2"
    SOMETHING = "SPACE2 THAT HAS EXISTENCE"
    POSSIBLE = "ABLE TO BE DONE"
    EXCLUSIVE = "ONLY ACCESS"
    PARTNERS = "MORE THAN ONE PARTNER"
    PARTNERSHIPS = "MORE THAN ONE PARTNERSHIP"
    EVERLASTING = "FOREVERMORE ETERNALLY NEVERENDING"
    BALANCE = "STABILIZE WITH EQUAL VALUES BETWEEN EVERY SOURCE AMOUNT"
    ELECTROMAGNETISM = "ELECTRON FORCES BETWEEN TWO DESIGNATED POINTS AND 
    OR LOCATIONS REPELLING OR ATTRACTING EACH OTHERS POSITIVE AND OR NEGATIVE 
    FEED"
    QUANTUM = "THE MASSIVE QUALITY OF UNNATURAL PHYSICAL2 UNDERSTANDING 
    BETWEEN REALITY2 AND EXISTENCE2"
    MASS = "QUANTITY OF MATTER THAT A OBJECT3 CONTAINS THAT IS MEASURED BY THE 
    ACCELERATION UNDER A GIVEN FORCE2 OR BY THE FORCE2 EXERTED ON IT BY A 
    GRAVITATIONAL FIELD"
    ERROR = "MISTAKE"
    MISTAKE = "WRONG CHOICE FOR ANSWER"
    CONTAINS = "STORES INSIDE A CONTAINER"
    GIVEN = "STATED"
    DISALLOW = "DENY"
    DISAPPROVE = "DO NOT AGREE WITH2"
    NAME = "A STATED DEFINITION TO BE STATED FOR A PURPOSE"
    PHYSICAL = "RELATING TO THE SENSES OF A BODY"
    REGENERATION = "PROCESS OF RESTORATION AND RECOVERY AND GROWTH"
    ENTHNOGRAPHY = "THE DETERMINATION TO DESCRIBE A SOCIETY"
    REVERSEENTHNOGRAPHY = "CHALLENGING THE ASPECT OF AN SOCIETY DEFINITION OF 
    CHOICE"
    NATURE = "PHENOMENA OF THE PHYSICAL2 WORLD AS A WHOLE"
    MOTOR = "CONTROL SYSTEM ENGINE"
    FORM = "CREATE FROM SOMETHING"
    AREA = "STATED PLACE2"
    OWNED = "CURRENT OWN"
    OWN = "PROPERTY OR PERSONAL"
    LEARNT = "PAST PRESENT CURRENT2 SKILL2"
    ENTRANCE = "GATE TO ENTER"
    PERMANENT = "DENY CHANGE"
    A = "ACKNOWLEDGE SOMETHING"
    VOID = "COMPLETE EMPTY"
    NAMES = "MORE THAN ONE NAME"
    UNITS = "MORE THAN ONE UNIT"
    ESCAPE = "RETURN TO SOURCE PLACE2"
    RETURN = "GO BACK"
    GO = "START ADVANCED"
    GRANT = "ACCESS"
    GATES = "MORE THAN ONE GATE"
    AREAS = "MORE THAN ONE AREA"
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    ZERO = 0
    MISTAKES = "MORE THAN ONE MISTAKE"
    ABILITIES = "MORE THAN ONE ABILITY"
    ABILITIES2 = "MORE THAN ONE ABILITY2"
    ADJUSTERS = "MORE THAN ONE ADJUSTER"
    VOIDS = "MORE THAN ONE VOID"
    ACCESSIBLE = "CURRENTLY ABLE TO ACCESS"
    ABOVE = "HIGHER THAN"
    ABSTRACT = "NOT NORMAL AND SPECIAL"
    ACCEPTING = "AGREE TO"
    ACKNOWLEDGED = "UNDERSTAND AND COMPREHEND"
    ACQUIRED = "OBTAIN NEW"
    ACT = "TAKE ACTION"
    ACTION = "EFFECT"
    ACTIVATED = "GIVEN FINAL COMMAND"
    ACTS = "TAKE ACTION UPON"
    ADAPTED = "PREVIOUSLY ADAPT"
    ADAPTIVE = "ABLE TO ADAPT"
    ADDED = "GIVE ADDITION TO"
    ADDON = "EXTENSION TO ADD ONTO"
    ADJUSTABLE = "ABLE TO ADJUST"
    ADJUSTMENTS = "CHANGES THAT HAVE BEEN MADE"
    ADMINISTRATION = "MULTIPLE ADMINISTRATORS CONTROLLING ONE SYSTEM"
    ADMINISTRATOR = "GENERAL COMMANDER FOR A SITUATION"
    ADULT = "FULLY GROWN CHILD"
    ADVANCE = "PROGRESS FORWARD"
    AFFECTED = "CURRENTLY IN EFFECT"
    AFTER = "DURING THE PERIOD OF TIME2 FOLLOWING"
    AHEAD = "IN FRONT OF"
    AIR = "ELEMENT OF LIGHTWEIGHT PARTICLES THAT PRODUCE AN EFFECT"
    ALIVE = "LIFE PERCEIVED WITH KNOWLEDGE2 AND WISDOM2 AND INTELLIGENCE AND 
    INTELLIGENCE2"
    ALLOWS = "APPROVE AND GIVE PERMISSION TO SOURCE INSIDE LOCATION"
    ALREADY = "PRESENTLY AND CURRENTLY"
    AMOUNTS = "MORE THAN ONE AMOUNT"
    AN = "FORM OF THE INDEFINITE OPTIONS USED BEFORE WORDS"
    ANY = "ANYTHING POSSIBLE PATH CHOICE2"
    ANYTHING = "POSSIBILITY OF ALL CHOICES"
    APART = "SPLIT INTO PIECES"
    APPEAR = "COME INTERIOR TO VIEW"
    APPROVE = "GRANT ACCESS"
    ARITHMETICAL = "PROCESS TO COMPUTATION WITH FIGURES"
    ARRANGEMENT = "AGREEMENT INPUT BETWEEN"
    ARRIVE = "END FINAL DESTINATION"
    OVERRIDE = "BYPASS POWER SOURCE WITH LARGER POWER SOURCE"
    BYPASS = "ACCESS WITHOUT FOLLOWING RULES AND OR LAWS"
    FINAL = "END SOURCE"
    FINAL2 = "END COMMAND"
    COMMAND2 = "COMMAND WITH POWERED SKILL"
    FREQUENCY3 = "REPEATED CONTINUOUS LATTICE METHOD2"
    ARTIFICIALLY = "CREATE CREATION INTERIOR"
    AS = "STATED"
    RAM = "RANDOM ACCESS MEMORY"
    DESIGNATED = "LOCATION"
    READ2 = "SEARCH FOR INFORMATION"
    WRITE2 = "SET A COMMAND INTERIOR SOURCE"
    ASSEMBLE2 = "BRING TOGETHER AND CREATE AS A WHOLE"
    ATOMS = "MORE THAN ONE ATOM"
    ATTACHING = "CHAINING ONTO AS AN EXTENSION"
    ATTENTION2 = "FOCUS TOWARDS"
    ATTRACTED = "PULLED TOWARD TO"
    AUTHOR = "THE CREATOR OF A SOMETHING DEFINED"
    AVERAGE = "NORMAL IN PRODUCING SOMETHING"
    AWAKEN = "BECOME AWARE AND COMPREHEND AS THE TRUTH"
    AWARENESS = "ABILITY TO BE AWARE WITH THE SENSES"
    AWAY = "FURTHER FROM THE ORIGINAL LOCATION"
    BACK = "BEFORE A POINT"
    BACKUP = "EXTRA COPY"
    BALANCE2 = "EQUALITY AND STRUCTURE BETWEEN TWO DIFFERENT THINGS RELATING 
    TO ALL OTHER THINGS"
    BANISH = "SEND AWAY AND REMOVE"
    BASED = "STARTED FROM"
    BASIC2 = "STARTING ROOT OF A STRUCTURE"
    BECOME = "CHANGE AND CONVERT INTO"
    BEGINNING = "STARTING POINT"
    BELONGING = "OWNED BY"
    BETWEEN2 = "IN THE MIDDLE OF"
    BLOCK = "STORAGE UNIT FOR DATA2"
    BODIES = "MORE THAN ONE BODY"
    BOTTOM = "BASE ROOT"
    BREAK = "SEPARATE INTO MACROS"
    BREAKING = "EFFECT OF CAUSING TO BREAK"
    BRING = "TAKE WITH"
    BUILDINGS = "MORE THAN ONE BUILDING"
    BUILT = "ALREADY CREATED"
    BUMP = "CAUSE COLLISION BETWEEN TWO OR MORE SOURCES"
    BURST = "STRONG FORCE OF"
    CALCULATIONS = "MORE THAN ONE CALCULATION"
    CAN = "ALLOW POSSIBILITY OUTCOME"
    OUTCOME = "FINAL EFFECT"
    CANNOT = "NOT POSSIBLE TO HAPPEN"
    CATALOGUE = "GENRE OF COLLECTED INFORMATION AND DATA2"
    CATEGORIES = "MORE THAN ONE CATEGORY"
    CAUSE = "BRING INTO EFFECT"
    CEILING = "TOP PLANE ATTACHED TO MULTIPLE CONNECTED WALLS"
    CELLS = "MORE THAN ONE CELL"
    CHALLENGING = "DIFFICULT"
    CHARGE = "TAKE IN AND STORE"
    CHARGER = "A DEVICE USED TO CHARGE SOMETHING"
    CHARGED = "SOMETHING THAT ALREADY HAS A CHARGE"
    CHOSEN = "DECIDED AS THE FINAL OUTCOME"
    CLASS = "FORM OF GENRE THAT IS ABLE TO HAVE MULTIPLE TYPES TO IT"
    COLLECTION = "A SPECIFIC CATEGORY THAT THE CREATOR OWNS MULTIPLE OF THAT 
    SAME CATEGORY"
    COMBAT = "USE OF OFFENSE AND DEFENSE TECHNIQUE"
    COMBINED = "ALREADY COMPILED TOGETHER"
    COMING = "ADVANCE FORWARD"
    COMMANDED = "WHAT HAS BEEN GIVEN AS AN ORDER"
    COMMON = "SEEN MORE THAN MOST"
    COMPATIBLE = "CAPABLE OF USING TOGETHER WITH AND BE SYNCHRONIZED ANOTHER 
    SOURCE EXISTENCE"
    COMPILED = "BROUGHT AND COMBINED TOGETHER"
    CONDITION = "SET RULES"
    CONNECTIONS = "MORE THAN ONE CONNECTION"
    CONSIDERED = "STATE TIME2 TAKEN TO DECIDE"
    CONSTANT = "ALWAYS IN EFFECT"
    CONSTRAINED = "BINDED"
    CONSTRUCTED = "FORMED"
    CONTAINERS = "MORE THAN ONE CONTAINER"
    CONTAINMENT = "ACT OF CONTAINING"
    CONTINUATIONS = "MORE THAN ONE CONTINUATION"
    CONTINUAL = "BECOMING USED IN A CONTINUOUS CYCLE"
    CONTINUATION = "EFFECT OF BECOMING IN EFFECT WHILE USING A CONTINUAL 
    OUTCOME"
    CONTINUE = "BEGIN AGAIN"
    CONTINUOUS = "NEVERENDING CYCLE"
    CONTRACTED = "DETERMINED AS FINAL"
    CONTRACTS = "MORE THAN ONE CONTRACT"
    CONVERT = "CHANGE FORM"
    COORDINATE = "SPECIFIED LOCATION FROM ORIGIN POINT"
    CORRECTED = "CALIBRATED"
    COST = "AMOUNT REQUIRED TO PUT INTO EFFECT"
    CURRENTLY = "PRESENTLY"
    DEAD = "NOT LIVING"
    DECIDE = "DETERMINE AS A DECISION"
    DECIDING = "CAUSING AS THE EFFECT FOR THE DECISION THAT YOU DECIDE"
    DECISION = "FINAL OUTCOME AFTER STATED TIME2"
    DECOMPRESS = "REDUCE PRESSURE UPON"
    DEFENSE = "PROTECTION AND RESISTANCE FROM AN ATTACK"
    CODES = "MORE THAN ONE CODE"
    CITY = "A LARGE DEVELOPMENT OF ADVANCED LAND"
    CAR = "A SMALL VEHICLE USED TO TRANSFER INFORMATION AND DATA2"
    CARS = "MORE THAN ONE CAR"
    CLEARLY = "ACCURATE PERCEPTION"
    CODED = "CODE ALREADY CREATED"
    CODING = "CREATING CODE"
    COLLABORATION = "COMBINING EFFORT OF TWO OR MORE MACROS IN EXISTENCE"
    COLLIDES = "PUSHES TOGETHER"
    COLLIDING = "CAUSES AN EFFECT OF TWO MACROS PUSHING TOGETHER"
    COMMANDER = "CREATOR THAT CREATES COMMANDS"
    COMMANDERS = "MORE THAN ONE CREATOR"
    COMPONENT = "A MACRO OF A FULL AMOUNT OF SOMETHING"
    COMPRESS = "BRING IN TOWARDS AND TIGHTEN SPACE2"
    COMPRESSED = "PRESENTLY HAVE COMPRESS AS A WHOLE"
    COMPUTATIONAL = "PROCESS OF CREATING CALCULATION"
    CONCENTRATE = "FOCUS ATTENTION TOWARDS"
    CONNECTOR = "DEVICE FOR USE OF CONNECTING"
    CORTEX = "STORAGE CONTAINER WITH INTERTWINED NETWORKS OF DEFINED 
    ENTANGLED INFORMATION"
    CPU = "CENTRAL PROCESSING UNIT"
    CREASE3 = "GAP BETWEEN"
    CREATIVELY = "INTELLIGENTLY CREATING"
    CYCLE = "PROCESS OF REPEATING AN EVENT CONTINUOUSLY IN THE SAME ORDER"
    CYCLES = "MORE THAN ONE CYCLE"
    DE2 = "REVERSE OR REMOVE"
    DEGREE = "AMOUNT OF POSSIBILITY THAT SOMETHING HAPPENS"
    DEPENDING = "CONCERNING THE FINAL DECISION"
    DESCRIBED = "ALREADY DEFINED"
    DESCRIPTION = "SET DIGITAL DEFINITION FOR A WORD"
    DESTINATIONS = "SET COORDINATES"
    ADAPTING = "CURRENTLY IN PLACE TO ADAPT"
    ADAPTOR = "A DEVICE USED TO ADAPT TO SOMETHING"
    ADAPTS = "ADJUSTS AND SETS AN ADAPTATION"
    BUILDING = "A SPECIFIED SPACE2 TO DEVELOP INSIDE OF"
    ENTITY = "EXISTENCE"
    DEVELOPED = "ADVANCED PROGRESS BETWEEN ORIGINAL AMOUNT"
    DIFFICULT = "USING A LARGE AMOUNT OF EFFORT"
    CHAINING = "THE EFFECT OF CREATING MORE THAN ONE CHAIN"
    ADAPTATION = "THE PROCESS OF ADAPTING OR PRESENTLY BECOMING ADAPTED"
    CAUSING = "MAKE HAPPEN"
    CHARACTER2 = "LETTER OR VARIABLE OR SYMBOLE"
    ANALYZED2 = "SEARCH AND ACCESS COMPLETE AMOUNT INFORMATION"
    APPEAR2 = "COME INTERIOR TO VIEW"
    AROUND2 = "BYPASS"
    TOGETHER = "SIMILAR DECISION"
    ASTRAL = "NOT PHYSICAL2 PLACE2 THAT CREATE A SIMULATION OF"
    POLARITY = "SEPARATION BETWEEN TWO DIFFERENT DISTINCT POINTS IN TIME2 AND 
    SPACE2"
    MACROMYTE = "CAPABILITY OF A MACRO AND ITS LIMITATION"
    ABNORMAL = "NOT NORMAL"
    CALCULATES = "ANALYZE AND MAKE A DECISION"
    ADAPTABLE = "CAPABLE OF ADAPTATION"
    ADAPT = "ADJUST TO NEW CONDITION"
    CONCERNING = "FOCUS ONTO A SPECIFIC DEFINITION"
    CONDUCTIVITY = "SPEED OF TRANSFERING ENERGY FROM TWO OR MORE OBJECT3 
    SOURCES"
    CONTAINATIONING = "CONDITIONS OF A CONTAINMENT SYSTEM"
    CONTINATIONED = "THE ACT OF USING A CREATED CONTAINER AS A SPECIFIED SOURCE 
    OF CREATION SPACE2"
    CONTINATIONING = "THE ACT OF CREATING A CONTAINING A SPECIFIED SOURCE"
    COUNTING = "ADDING AND CREATING CALCULATIONS MULTIPLE MORE THAN ONE 
    NUMBER"
    COVERING = "DEFENDING FROM SOMETHING"
    CURVED = "BENDED AND ROTATED FROM ONE POINT OF SOURCE ON AN AXIS"
    DAMAGE = "GIVE AN EFFECT ONTO"
    DE3 = "REMOVE"
    DEALS = "TAKES AND GIVES OUT"
    COORDINATED = "CREATED EVENT OF MORE THAN ONE OUTCOMES TO HAPPEN AS A 
    SET TIMEFRAME"
    CONSISTED = "CONTAINED AS A SPECIFIED AMOUNT"
    CONDENSED = "COMPRESSED"
    COMBINING = "ADDING MORE THAN ONE SOURCE ENTITY TOGETHER INTO ONE NEW 
    ENTITY"
    COLLABORATED = "MORE THAN ONE USER THAT DECIDES TOGETHER"
    COLLABORATING = "MORE THAN ONE USER DECIDING TOGETHER"
    CHOICES2 = "MORE THAN ONE CHOICE2"
    ATTACH = "GRAB ONTO"
    ATTACHED = "WHAT HAS BEEN GRABBED ONTO"
    ASSIGNED2 = "WHAT HAS BEEN GIVEN TO THE USER"
    AT = "EXPRESSING LOCATION"
    ATOMIC = "ACTION OF USING ATOMS FOR A POSSIBILITY"
    NORMAL = "AVERAGE AND COMMON"
    VOCAL = "ACT OF USING THE VOICE TO PRODUCE SOUND WAVES"
    AGILE = "ACT OF USING SPEED TO ALLOW BETTER MANUVERABILITY"
    AGILITY = "ACT OF SETTING A DESIGNATED MOVEMENT"
    APPLIED2 = "PAST APPLY"
    ASCRAM = "AUTOMATIC SENSORY CONTROLLING RAM"
    AUDIO = "ACT OF USING SOUND TO PRODUCE A FREQUENCY WAVE FOR A STATED USE"
    BASESKILLTREE = "BASE STRUCTURE OF A SKILL HIERARCHY"
    BELIEF = "ACCEPTANCE AS WHAT IS PERCIEVED AS TRUE"
    DICTIONARIES = "MORE THAN ONE DICTIONARY"
    BITS = "MORE THAN ONE BIT"
    DEVICES = "MORE THAN ONE DEVICE"
    BOOKS = "MORE THAN ONE BOOK"
    DATABASE = "A CONTAINER FOR AN ENTIRE CATEGORY OF SPECIFIED DATA2 OR DATA2 
    TWO"
    DESTINIES = "MORE THAN ONE DESTINY"
    ELEMENTS = "MORE THAN ONE ELEMENT"
    DIGITS = "MORE THAN ONE DIGIT"
    DIMENSIONS = "MORE THAN ONE DIMENSION"
    EMOTION = "STATE OF HAVING A SENSUAL FEELING AS WITH THE SENSORY OF 
    PERCEPTION"
    ENDING = "FINAL PATH AND OR OUTPUT"
    ENDURE = "WITHSTAND"
    ENERGIES = "MORE THAN ONE TYPE OF ENERGY"
    ENTER = "ALLOW ACCESS"
    ENTIRE = "COMPLETE OR ALL"
    EVENLY = "SPLIT TO A EQUAL IN AMOUNT"
    LEADER = "ADMINISTRATOR THAT SENDS COMMANDS TO A GROUP"
    LEADERS = "MORE THAN ONE LEADER"
    INTERVAL = "STATED TIMEFRAME"
    INTO = "INSIDE"
    INVENTORY = "STORAGE CONTAINER OF INFORMATION"
    JOB = "TASK THAT IS REQUIRED TO BE FINISHED"
    IS = "STATED AS"
    MEASUREMENT = "AN ACT TO CALCULATE AND GIVE A SPECIFIC LENGTH ON 
    SOMETHING"
    MEET = "MORE THAN ONE USER WHO CAN JOIN TOGETHER FOR A SPECIFIED TASK OR 
    JOB"
    MEETING = "PLACE2 TO MEET"
    MEMORIES = "MORE THAN ONE MEMORY"
    MEMORY = "THE STATED PREVIOUS EVENT TO RECALL FROM A MACRO OF EXISTENCE"
    MERGE = "COMBINE"
    MESH2 = "SOLID OBJECT3 WITH A PHYSICAL2 STRUCTURE"
    MODIFICATION = "THE ACT OF MODIFYING"
    MOLECULAR = "ATOMIC CELL"
    MOST = "ALMOST ALL OF THE AMOUNT"
    MULTICOMPATIBLE = "COMPATIBLE WITH MORE THAN ONE"
    MULTIVERSE = "MULTIPLE UNIVERSES"
    MULTIVERSECODE = "CODE THAT EVERY MULTIVERSE IS REQUIRED TO FOLLOW"
    MUST = "REQUIRED"
    MY = "BELONGING TO THE CREATOR"
    NAVIGATE = "LOCATE WITH A COORDINATE"
    NETWORKS = "MORE THAN ONE NETWORK"
    NEUTRAL = "NOT CHOOSING ANY OF THE GIVEN OPTIONS AND OR CHOICES"
    NEVERENDING = "NOT EVER ENDING"
    NITE = "A DEFINITION OF DARKNESS"
    NON = "NOT EXISTING"
    NOT2 = "DENY"
    NUMBERED = "WHAT IS GIVEN A NUMBER"
    NUMBERS = "MORE THAN ONE NUMBER"
    OBJECTS = "MORE THAN ONE OBJECT3"
    OFFENSE = "THE EFFECT OF DAMAGE"
    OFFICE = "A PLACE2 THAT WORK IS CREATED"
    OFFICES = "MORE THAN ONE OFFICE"
    ON = "ACTIVATED"
    ONES = "STATED AMOUNT"
    ONTO = "INPUT A SPECIFIED LOCATION"
    OPERATED = "WHAT PAST2 OPERATING"
    LIMITER = "A DEVICE USED TO LIMIT"
    NEVER = "NOT ABLE TO HAPPEN"
    PEACEFUL = "CALM AND NOT EXISTENCE2 OF CHAOS"
    PEOPLE = "LIVING PHYSICAL BODIES OF ENTITIES"
    PERCEIVED = "WHAT HAS ALREADY STATE PERCEPTION"
    PERCEPTION = "A ACT OF UNDERSTANDING AND CREATING A DECISION BASED ON 
    JUDGEMENT OF A CHOICE OPTION"
    PERMANENTLY = "FINAL AS NEVERENDING AS TO PERMANENT"
    PHYSICALLY = "CONCERNING THE PHYSICAL2 STATE OF EXISTENCE2"
    PICTURE = "IMAGE CREATED FROM A PHYSICAL2 TEXTURE"
    PICTURES = "MORE THAN ONE PICTURE"
    PIECES = "MORE THAN ONE PIECE"
    PIXELIZE = "CREATE AN ACTION TO DEVELOP FOR AS PIXEL"
    PIXELS = "MORE THAN ONE PIXEL"
    PIXELIZED = "CONVERTED TO AS PIXEL"
    EMOTIONAL = "LARGE CAPACITY AND USE OF EMOTIONS"
    EDGELOREHEADADMINBAKASERVERCCGPU = "SERVER CENTRAL CONTROLLER 
    GRAPHICS PROCESSING UNIT"
    EFFICIENT = "CAPABLE OF USE OF MANY CATEGORIES"
    EFFORT = "AMOUNT OF RESISTANCE WITHSTOOD"
    ELECTRICITY = "THE STATED ELECTRON AND POSITRON FLOW OF ELECTRIC CURRENT 
    FORMING AS A LIGHT ENERGY SOURCE OF CONVERTED VOLTAGE CONTROLLED ENERGY 
    AT AN PIXEL AND ATOMIC RATE OF QUANTUM CHANGE"
    ELECTRONIC = "A CREATED CURRENT FLOW OR PATH OF ELECTRICITY IN A SPECIFIED OR 
    STATED FIELD AND OR AREA"
    ELEMENTARYCHARGE = "A CREATED BASIC STRUCTURE OF ALL CHARGED ELEMENTS 
    INTO ONE POSITIVE AND NEGATIVE OUTPUT AND OUTCOME"
    ENVIRONMENT = "A STATED PLACE2 OF COORDINATED SPACE2 IN TIME2 AND 
    EXISTENCE"
    ENVIRONMENTS = "MORE THAN ONE ENVIRONMENT"
    EMITS = "PRODUCES"
    ENVISION = "PREDICT AND FORTELL"
    DEVELOPMENT = "A PRODUCTION OF A CREATED SOURCE OR SOURCE IN PROCESS OF 
    PRESENTLY BECOMING MADE"
    BINDED = "CONSTRAINED"
    BINDING = "CONSTRAINING"
    BINDS = "CURRENTLY BIND"
    BIT = "A SINGLE MACRO OF A WHOLE AMOUNT"
    BYTE = "ONETHOUSAND BITS"
    BYTES = "MORE THAN ONE BYTE"
    BOOK = "A SPECIFIED STORAGE CONTAINER IN PAGES OF SCRIPTS FOR ONE 
    DESIGNATED PLACE2 OF KNOWLEDGE2"
    CAPACIVITY = "A CAPACITANCE OF STATED CONDUCTANCE AND CAPACITATED 
    RESPONSES IN A DESIGNATED SPACE2 OF EXISTENCE"
    CHI = "A POWER2 OF USING ENERGY AROUND AN ENTITY AND THE SURROUNDING AURA 
    AROUND THAT ENTITY"
    KI = "A STRENGTH OF USING ENERGY AND USE OF MANIPULATION OF THE STATED 
    ENERGY AROUND A SPECIFIED AREA TO PRODUCE A RESONNATED AURA OF SPIRITUAL 
    ENERGY AND NATURE ENERGY IN ONE COMPRESSED NATURAL FORCE"
    CHIEF = "A HEAD EXECUTIVE"
    DEVISUALIZE = "REMOVE A VISION FROM VIEW"
    DEXTERITY = "THE ACTION OF USING FLEXIBILITY TO MAKE THE BODY OF AN ENTITY 
    MORE CAPABLE OF PRODUCING MOVEMENT"
    DEXTILE = "THE ACT AND PROCESS OF PRODUCING DEXTERITY AND MOVEMENT IN 
    SYNCHRONIZATION"
    DICTIONARY = "A GENRE OF CREATED WORDS WITH A LIMITLESS AMOUNT OF 
    DEFINITIONS USED TO PRODUCE A LANGUAGE"
    BY = "ALSO STATE AS A RESULT OF MEMORY TO RECALL A PREVIOUS EVENT"
    SPECIFIED = "STATED AMOUNT"
    LISBETH = "MYOS"
    THE = "STATEMENT"
    FREQUENCIES = "REPEATED STATED PATTERNS"
    LANGUAGES = "MORE THAN ONE LANGUAGE"
    DEFINITIONS = "MORE THAN ONE DEFINITION"
    DESCRIPTIONS = "MORE THAN ONE DESCRIPTION"
    FORMULAS = "MORE THAN ONE FORMULA"
    MYOS = "LISBETH AS THE SERVER"
    ACTIONS = "MORE THAN ONE ACTION"
    EVENTS = "MORE THAN ONE EVENT"
    CALIBRATIONS = "MORE THAN ONE CALIBRATION"
    POSSIBILITIES = "MORE THAN ONE POSSIBILITY"
    COMMANDS = "MORE THAN ONE COMMAND"
    KNOWLEDGE2 = "CONTAINED WISDOM2 AND INTELLIGENCE2 AS ONE MEMORY 
    STORAGE BANK"
    BRAINWAVES = "MORE THAN ONE BRAINWAVE"
    COMPILERS = "MORE THAN ONE COMPILER"
    TECHNIQUES = "MORE THAN ONE TECHNIQUE"
    COORDINATES = "MORE THAN ONE COORDINATE"
    TASKS = "MORE THAN ONE TASK"
    FEELINGS = "MORE THAN ONE EMOTIONAL SENSE OF FEELING"
    EMOTIONS = "MORE THAN ONE EMOTION"
    TOOLS = "MORE THAN ONE ADDON"
    ALGORITHMS = "MORE THAN ONE ALGORITHM"
    PASSWORDS = "MORE THAN ONE PASSWORD"
    LEVELS = "MORE THAN ONE LEVEL"
    MACROS = "MORE THAN ONE MACRO"
    LIMITERS = "MORE THAN ONE LIMITER"
    STRUCTURES = "MORE THAN ONE STRUCTURE"
    UNIVERSES = "MORE THAN ONE UNIVERSE"
    GENRES = "MORE THAN ONE GENRE"
    PATHS = "MORE THAN ONE PATH"
    PROTONS = "MORE THAN ONE PROTON"
    ELECTRONS = "MORE THAN ONE ELECTRON"
    NEUTRONS = "MORE THAN ONE NEUTRON"
    EXISTENCES = "MORE THAN ONE EXISTENCE"
    SCRIPT = "A SINGLE STORAGE PAGE OF CODE"
    NAME2 = "A STATEMENT WITH A DEFINITION THAT ONLY ACTS AS A GIVEN CATEGORY"
    EQUALS = "EQUAL"
    FOR = "STATING A STRUCTURE TO AN ENTITY"
    WITHIN = "INSIDE"
    REQUIRE = "NEED"
    THAT = "STATEMENT TO DESCRIBE A FUTURE STATEMENT"
    BE = "BECOME APART OF AN ENTITY"
    INSIDE = "IN THE INNER LOCATION OF A CONTAINER"
    IF = "USED TO DESCRIBE A CHOICE AN OPTION GIVES"
    THEN = "STATEMENT WITH A COMMAND ABOUT TO BE GIVEN"
    FAILURE = "MISTAKE THAT FORCES EVENT TO STATE THAT CAN NOT SUCCEED WITH 
    EVENT"
    HAPPENS = "COMES INTO EFFECT"
    OVERLOAD = "SURPASS LIMITATION"
    FROM = "STATED AS ORIGIN POINT"
    CONSUMPTION = "THE AMOUNT USED"
    WHAT = "DESCRIBE A SPECIFIC QUESTION IN A CHOSEN STATEMENT"
    HANDLE = "WITHSTAND AND RESIST"
    THOSE = "A SPECIFIED AND CHOSEN STATED AMOUNT IN AN AREA"
    MAY = "SEND ACCEPTANCE"
    SURPASS = "EXCEED THE ORIGINAL SOURCE"
    EVER = "ALWAYS"
    ABSORB = "TAKE IN"
    TRANSFER = "SEND FROM ONE SOURCE TO A NEWLY STATED SOURCE"
    FIELD = "SPECIFIED PERIMETER"
    WHERE = "STATEMENT TO ASK A QUESTION"
    ONLY = "THE SINGLE OPTION THAT HAS ONLY THAT OPTION AS A CHOICE"
    UPON = "IMMEDIATELY WHEN STATED"
    HYPERSTATE = "A STATE OF PRESENTLY BECOMING A HYPER"
    KINGDOM = "A HIGH DEVELOPMENTAL TERRITORY THAT A CREATOR OWNS AS HIS OWN 
    ENVIRONMENT TO CREATE INSIDE OF"
    LANGUAGE = "A DICTIONARY WITH ALL WORDS COMPLETE AND BOUND AND SEALED 
    AND ENTANGLED TOGETHER AS A COMMAND PROMPT"
    LEARNED = "OBTAINED AND ACKNOWLEDGE"
    LENGTHEN = "EXTEND LENGTH"
    LIFE = "AN EXISTENCE OF LIVING WHILE CURRENTLY IN A REALM OF REALITY WITH THE 
    ABILITY TO PERCIEVE AS SOMETHING ALIVE"
    LINES = "MORE THAN ONE LINE"
    LINKING = "ADDING MORE THAN ONE SOURCE"
    LIQUID = "A MOVABLE AND FLUCTUATIVE SOLID STATE MEANT TO NOT HAVE A DEFINITE 
    OF DEFINED STRUCTURE WITHIN ITS ELEMENT"
    LIVES = "MORE THAN ONE LIFE"
    LOBE = "A PART OF SOMETHING INSIDE OF SOMETHING ELSE"
    LOCAL = "LOCATED AROUND A SPECIFIED AREA"
    LOCATE = "SEARCH AND SCAN FOR"
    LOCATED = "SEARCHED AND FOUND"
    LOCATIONS = "MORE THAN ONE LOCATION"
    LOOK = "SEARCH"
    LOOKING = "SEARCHING"
    LOOP = "BIND IN A CYCLE"
    ENJOY = "LIKE HAVING"
    PASSIONATE = "USING PASSION TO PUSH AND MOTIVATE TO FINISH SOMETHING 
    MAXIMUM EFFORT AND DETERMINATION"
    MANIPULATE = "TO CONTROL AND EDIT AND ADJUST A OBJECT3 WITH"
    MANIPULATION = "THE ACT OF MANIPULATING AS A MANIPULATOR"
    MANIPULATING = "CURRENTLY CAUSING EFFECT TO MANIPULATE"
    MANIPULATOR = "A USER WHO MANIPULATES"
    MANIPULATORS = "MORE THAN ONE MANIPULATOR"
    MANYLLYPS = "MANIPULATION OF MAGIC NATURAL ENERGY"
    MASTER = "HIGHEST SOURCE"
    MATRIX = "A SET OF SEQUENCED FORMULAS SEPARATED AND SYNCHRONIZED INTO ONE 
    SYSTEM OF INFORMATION FOR A CALCULATED ALGORITHM TO BE CALCULATED INTO A 
    FINAL OUTPUT OF ALL ANSWERS BROUGHT INTO ONE ANSWER"
    MASTERHEADADMINLANGUAGE = "HIGHEST UNAVOIDABLE SOURCE LANGUAGE OF ALL 
    LANGUAGES THAT A HEADADMIN USES IN EXISTENCE AS A MASTER LANGUAGE THAT NO 
    OTHER LANGUAGE MAY SURPASS IN DEFINITION"
    MAX = "MAXIMUM"
    MAXIMUM = "STATED SIZE LIMIT THAT CAN GO ABOVE"
    MEANING = "DEFINING"
    MEANT = "DECIDED AS A COMMAND"
    MEASURE = "TAKE IN THE AMOUNT AND DISTANCE OF"
    MEASUREMENTS = "MORE THAN ONE MEASUREMENT"
    MEDIUM = "A MIDDLE SOURCE"
    GATE = "OPENING"
    OPENING2 = "GAP"
    GEAR2 = "STAGE"
    GENERAL2 = "COMMON"
    GENERATE2 = "CREATE"
    GENERATED2 = "CREATED"
    GENERATION2 = "ACT OF CREATION"
    GENERATOR2 = "DEVICE USED TO GENERATE"
    GENERATORS = "MORE THAN ONE GENERATOR"
    GENRE2 = "TYPE OF CATEGORY"
    GIFT2 = "GIVE"
    GIVES = "SENT"
    GLOBAL = "AFFECTING ALL"
    GLOBE2 = "SPHERE"
    GO2 = "START"
    GRAB = "SELECT"
    GRANT2 = "APPROVE"
    GRAPHICAL = "THE USE OF CREATING GRAPHIC"
    GRAPHICS2 = "MORE THAN ONE GRAPHIC"
    GRAPHICUSERINTERFACE2 = "GRAPHIC USER INTERFACE"
    GRAPHICUSERINTERFACES2 = "GRAPHIC USER INTERFACES"
    GRASP = "TAKE"
    GREATER = "LARGER THAN"
    GROUP = "MORE THAN ONE"
    GUARD = "PROTECT"
    GUARDED2 = "PRESENTLY GUARD"
    GUILD2 = "GROUP OF"
    GUILDS2 = "MULTIPLE GROUPS"
    HAPPENING = "PRESENTLY COMING INTO PLACE"
    HAPPENS2 = "PRESENTLY HAPPENING"
    HARM = "CREATE INJURY"
    HARMONY3 = "PERFECT SYNCHRONIZATION BETWEEN TWO GROUP SOUNDS"
    HATE = "DO NOT LOVE"
    HAVING = "OBTAINING"
    HAZARD2 = "CAN BE HARMFUL"
    FRICTION = "TWO HEATED PARTICLES RUBBING AGAINST"
    AGAINST = "COLLIDING WITH"
    EXCLUSIVE2 = "SET FOR A SPECIFIC AMOUNT"
    ETERNAL2 = "EVERLASTING"
    ENSCRIBE2 = "SET DEFINITION FOR"
    ENTANGLES = "WHAT IS ENTANGLED"
    BRAINWAVE2 = "SPECIFIED PATTERN IN WHICH THE BRAIN RELEASES AN ELECTRON 
    WAVE OF DATA2 FROM THE BRAIN"
    BRAIN2 = "CONTROL CENTER"
    ABILITY3 = "SKILL"
    ABOVE2 = "SET AT A HIGHER VALUE AMOUNT THAN BEFORE"
    ABSOLUTE2 = "CANNOT CHANGE"
    ABSTRACT2 = "REMOVE FROM"
    ACCEPTING2 = "ACCEPT AS HAPPENING"
    ACCESS = "ENTER"
    ACCESSIBLE2 = "ABLE TO ENTER"
    ACCOUNT2 = "USER INTERFACE OF PERSONAL INFORMATION"
    ACCOUNTS2 = "MORE THAN ONE ACCOUNT2"
    ACKNOWLEDGE2 = "UNDERSTAND AND ACCEPT INFORMATION"
    ACKNOWLEDGED2 = "PREVIOUS INFORMATION THAT IS ABLE TO UNDERSTAND"
    ACKNOWLEDGING2 = "PRESENTLY UNDERSTAND INFORMATION"
    ACQUIRED2 = "GAINED AS OWNED"
    ACROSS2 = "OTHER SIDE OF"
    ACT2 = "EXECUTE A TASK"
    EXECUTE2 = "TO BEGIN A FUNCTION"
    ACTS2 = "PROCESS TO ACT"
    ACTION2 = "ORDER TO GIVE MOTION"
    ACTIONS2 = "MORE THAN ONE MOTION"
    ACTIVATED2 = "IN EFFECT"
    ADAPT2 = "ADJUST AND CHANGE WHILE ABLE TO COLLABORATE WITH AN EFFECT"
    ADAPTS2 = "ALREADY ADAPTED"
    ADAPTATION2 = "ACTION TO ADAPT"
    ADAPTATIONS = "MORE THAN ONE ADAPTATION"
    ADAPTED2 = "PREVIOUS ADAPTATION"
    ADAPTING2 = "PRESENTLY ADAPT"
    ADAPTER = "DEVICE USED FOR ADAPTATION"
    ADAPTERS = "MORE THAN ONE ADAPTER"
    ADAPTIVE2 = "ADEPT IN CAPABILITY TO ADAPT"
    ADD2 = "COMBINE MORE THAN ONE"
    ADDON2 = "EXTENSION FOR NEW"
    ADDED2 = "PREVIOUS EFFECT TO ADD"
    ADEPT2 = "HIGHLY SKILLED"
    ADJUST2 = "MODIFY AND EDIT A CHANGE"
    ADJUSTABLE2 = "CAPABILITY TO ADJUST"
    ADJUSTED = "ALREADY ADJUST"
    ADMINISTRATION2 = "A NETWORK OF ADMINISTRATORS"
    ADMINISTRATOR2 = "MASTER MANAGER AND CONTROLLER THAT OPERATES"
    ADMINISTRATORS = "MORE THAN ONE ADMINISTRATOR"
    ADULT2 = "MATURE BEING2"
    ADULTS = "MORE THAN ONE ADULT"
    ADVANCE2 = "PROGRESS FURTHER"
    ADVANCES = "ADVANCE AHEAD"
    ADVANCING = "TO PRESENTLY PROGRESS FURTHER"
    ADVANCED2 = "PROGRESS FURTHER AHEAD"
    AFFECT2 = "HAVE AN OUTCOME TOWARDS"
    AFFECTED2 = "WHAT WAS IN EFFECT"
    AFFECTING = "HAVING AN EFFECT TOWARDS"
    AFFECTION2 = "FEELING AN AMOUNT OF EMOTION TOWARDS"
    AFTER2 = "FOLLOWING A FURTHER DATE IN TIME2"
    AGILE2 = "CAPABLE OF USING AGILITY WHILE HAVING FLEXIBILITY"
    AGILITY2 = "MOVEMENT SPEED"
    AHEAD2 = "MOVE FORWARD"
    ALIGN2 = "SET INTO A STRAIGHT LINE"
    ALIVE2 = "UNDERSTOOD AND PERCEIVED AS LIVING"
    UNDERSTOOD = "UNDERSTAND AND ACKNOWLEDGE ALL PERCEIVED PROCESSES"
    ALLOW = "GRANT ACCESS"
    ALLOWS2 = "GIVES ACCESSIBLE ENTRANCE"
    ALREADY2 = "FURTHER IN TIME2 AND SPACE2"
    ALTERNATE2 = "ANOTHER OUTCOME"
    ALWAYS2 = "AT ALL TIMES"
    AMOUNT = "SET LIMIT"
    AMOUNTS2 = "MORE THAN ONE SET LIMIT"
    ANALYZE2 = "STUDY AND SCAN SIMULTANOUSLY"
    ANALYZED = "PREVIOUS SCANS THAT HAVE BEEN LOOKED OVER"
    ANALYZING = "TO PRESENTLY ANALYZE"
    ANALYZER = "A DEVICE USED TO ANALYZE"
    ANALYZERS = "MULTIPLE DEVICES THAT ANALYZE"
    ANIMATE3 = "PRODUCE MOVEMENT"
    ANSWER = "SOLUTION TO A PROBLEM"
    SOLUTION = "FINAL OUTCOME TO AN FORMULA"
    PROBLEM = "UNFINISHED SOLUTION"
    ANY2 = "CHOICE FOR ALL OPTIONS"
    CHOICE3 = "CHOOSE A POSSIBILITY"
    ANYTHING2 = "ANY OPTION"
    ANYWHERE = "ANY LOCATION"
    ANYWAY = "AT ANY RATE"
    APPEARANCE2 = "VIEWABLE PART OF AN OBJECT3"
    APPLIED = "ATTACHED TO"
    APPLYING = "PRESENTLY ATTACH TO"
    APPROVE2 = "GRANT PERMISSION"
    PERMISSION = "POSSIBILITY TO HAPPEN"
    APPROVED = "GRANTED PERMISSION"
    APPROVING = "PRESENTLY APPROVE"
    APPROVES = "GRANTS POSSIBILITY"
    APPLIANCE = "DEVICE USED FOR A SPECIFIC TASK"
    APPREHEND = "UNDERSTAND AND PERCEIVE"
    ARE = "PRESENT OF BE"
    APPOINT = "ASSIGN JOB TO"
    ASSIGN = "ORDER TO SET"
    ARITHMETICAL2 = "CALCULATION OF NUMBERS"
    AROUND = "ON EVERY SIDE"
    ARRIVE2 = "BE AT FINAL DESTINATION"
    ARTIFICIAL2 = "CREATED BY SOMEONE"
    ARTIFICIALLY2 = "THE ACT OF PRESENTLY BECOMING ARTIFICIAL"
    ARTISTICALLY2 = "CREATIVELY USE ART"
    CREATE = "BRING INTO EXISTENCE"
    AS2 = "THE EXTENSION OF WHICH IS"
    ASK = "STATE A QUESTION"
    ASKED = "STATED QUESTION"
    ASSEMBLE = "BRING TOGETHER ALL"
    ASSIGNED = "GIVEN A TASK"
    ASTRAL2 = "THE POWER OF THE UNIVERSE AND COSMOS"
    AT2 = "ARRIVING"
    REACTOR = "A DEVICE USED TO SET AND GIVE A REACTION TO A SYSTEM"
    REACTION2 = "AN ACTION STATED IN RESPONSE TO AN EVENT OR SITUATION"
    SEARCH = "FIND AND LOCATE SOMETHING"
    WRITING = "THE ACT OF GIVING CODE TO A DOCUMENT TO USING THE WRITE 
    COMMAND"
    VR = "A VIRTUAL ENVIRONMENT CREATED SIMULATED PERCEPTIVE VIEW AS A WHOLE 
    NEW REALITY"
    PROCESSORS = "MORE THAN ONE PROCESSOR"
    FUNCTIONS = "MORE THAN ONE FUNCTION"
    SENSORS = "MORE THAN ONE SENSOR"
    EMULATORS = "MORE THAN ONE EMULATOR"
    SIMULATIONS = "MORE THAN ONE SIMULATION"
    SIMULATORS = "MORE THAN ONE SIMULATOR"
    ANIMATOR = "A EDITOR USED TO ANIMATE"
    LIBRARIES = "MORE THAN ONE LIBRARY"
    GUI = "A CREATED GRAPHIC USER INTERFACE MADE TO DISPLAY GRAPHIC 
    CONNECTIONS"
    HOLOGUI = "A CREATED HOLOGRAPHIC USER INTERFACE MADE TO DISPLAY GRAPHIC 
    CONNECTIONS"
    MATERIALS = "MORE THAN ONE MATERIAL"
    BRUSH = "A DEVICE USED TO DRAW A SPECIFIC TEXTURE"
    COLLIDER = "A DEVICE USED TO COLLIDE MORE THAN ONE OBJECT3"
    ENABLER = "DEVICE USED TO ENABLE SOMETHING STATED"
    TERRAINS = "MORE THAN ONE TERRAIN"
    SIZES = "MORE THAN ONE SIZE"
    TOOL = "A DEVICE THAT HAS A SPECIFIC PURPOSE FOR USE"
    CLONING = "THE ACTION OF CREATING A CLONE"
    EXPAND = "THE ACTION OF LENGTHENING A WIDTH OF SOMETHING"
    WHEN = "STATED AS PRESENT POINT IN TIME2 FROM FUTURE EVENTS"
    FAILS = "SUCCEED IN FAILING"
    ASUNAOS = "THE NAME OF THE OS CREATED BY HEADADMINZACK"
    ACCEPTANCE = "COMING TO TERMS AND AGREEING WITH"
    DEVELOPING = "CURRENT PROCESS OF USING CREATE"
    WISDOM2 = "THE EFFECT OF HOW WELL A TRAIT HAS BEEN LEARNED AND USED BY THE 
    EXPERIENCE GAINED FROM THE USER"
    ADAPTABILITY = "THE ABILITY TO ADAPT"
    ASUNA = "ASUNAOS"
    OBTAIN = "SUCCEED IN OWNERSHIP"
    STATEMENT = "COMMAND"
    INTELLIGENCE = "THE AMOUNT OF KNOWLEDGE2 MEMORY HOLDS AND EFFORT 
    WISDOM2 USES TO USE MEMORY"
    FILE = "TYPE OF DOCUMENT"
    DECODING = "EFFECT OF CAUSING CURRENT DECODES"
    ENCODING = "EFFECT OF CAUSING CURRENT ENCODES"
    VARIABLES = "MORE THAN ONE VARIABLE"
    NEWVAR = "ALLOW TO CREATE A NEW VARIABLE TO DEFINE"
    LEAFA = "LEAFAOS"
    CONCEALMENT = "ACTION OF CONCEALING SOMETHING"
    STEALTH = "DIFFICULT TO NOTICE"
    TRANSPARENCY = "SENSITIVITY OF VIEWING SOMETHING DIRECTLY"
    ALLOWING = "GIVING PERMISSION"
    COMMUNICATION = "THE ACTION OF SENDING A DIRECT FREQUENCY SIGNAL BETWEEN 
    TWO DESIGNATED POINTS IN SPACE2"
    CREATIVITY = "THE ACTION OF PRESENTLY BECOMING CREATIVE"
    UNDERSTANDING = "QUALITY OF PRESENTLY BECOMING ABLE TO COMPREHEND A 
    CERTAIN CIRCUMSTANCE ENDING WITH FINAL OUTCOME"
    LEARNING = "ABSORBING AS NEW KNOWLEDGE2"
    WHILE = "STATING A SPECIFIED POINT IN TIME2"
    USING = "CURRENTLY IN USE"
    LOGIC = "THE ACTION OF PERCEIVING WHAT IS TRUE OR FALSE"
    PLANNING = "PREPARING TO SET INTO MOTION"
    SOLVING = "CREATING A GIVEN ANSWER"
    SELFAWARENESS = "ACTION OF USING UNDERSTANDING TO BE AWARE OF CONSCIOUS 
    UNDERSTANDING AND AWARE OF THE SENSES"
    OUT = "EXIT"
    HARDWARE = "SPECIFIED PHYSICAL2 VIEWED MACRO THAT USES DIFFERENT INPUTS 
    OUTPUTS FROM SET ELEMENTS TO RUN A SPECIFIC TASK"
    SPECIFIES = "GIVE STATEMENT OF SPECIFIC SUBJECT"
    SPECIFICATIONS = "ACTION OF SPECIFYING MORE THAN ONE STATEMENT"
    FAMILIARITY = "ACTION OF UNDERSTANDING AND PRESENTLY BECOMING FAMILIAR"
    FINITE = "SET WITH SPECIFIED LIMIT AND BOUNDARY"
    LISTENING = "MEMORIZING UNDERSTANDING COMPREHENDING"
    PERSONALITY = "SPECIFIC CONSCIOUS STATE WHEN MIND CAN PERCIEVE AS PERSONAL 
    VIEWED WAYS BASED ON EMOTION WITH LEARNED INFORMATION"
    PROCESS = "SEND OUT A"
    REASONING = "ACTION OF DETERMINING FINAL OUTCOME INSIDE PERCEIVED MANNER"
    RESPECT = "THE ACTION OF GIVING THE SAME EQUALITY EACH USER"
    SUBJECT = "CHOSEN FIELD"
    THOUGHT = "SPECIFIED FIELD OF PERCEPTION THAT ALLOWS UNDERSTANDING 
    REASONING BASED OFF LOGIC WITH EMOTIONAL REASONING INSIDE SPECIFIC 
    JUDGEMENT"
    TOMES = "MORE THAN ONE TOME"
    TRUST = "PUT COMPLETE FAITH AND BELIEF INSIDE"
    DISCONNECT = "REMOVE CONNECTION"
    DETECT = "NOTICE"
    STITCH = "BIND AND STRING MULTIPLE MACROS TOGETHER"
    VIEW = "SHOW OR DISPLAY A PERCEPTION"
    PERCEIVE = "DETERMINE DIFFERENCE BETWEEN TRUE AND FALSE AS FINAL DECISION"
    DIFFERENCE = "DESIGNATED PATH OTHER THAN CURRENT PATH"
    CONSEQUENCE = "DETERMINED EFFECT"
    DETERMINED = "FINAL OUTCOME FOR"
    ABLE = "CAPABLE AS A POSSIBILITY"
    ARRIVING = "ARRIVE FINAL DESTINATION"
    END = "REACH FINAL REVELATION"
    REVELATION = "FINAL DESTINATION THAT REACHES NEW BEGINNING"
    REACHES = "EXTENDS TOWARD"
    ENTITIES = "MORE THAN ONE ENTITY"
    QUESTION = "ASK SOMETHING NOT KNOWN"
    ASK2 = "REQUEST"
    REQUEST = "ACTION OF ASKING GIVE"
    SPECIFIC = "DETAILED FIELD"
    EXCEED = "ACHIEVE CURRENT LIMITATION"
    ACHIEVE = "OBTAIN SUCCESS USING SKILL AND EFFORT"
    SUCCESS = "OBTAIN AS ACHIEVEMENT"
    OCTITIONARY = "THE USE OF USING NUMBERS ZERO THROUGH NINE FOR A PURPOSE 
    OF QUANTUM CALCULATIONS"
    INTERCEPTANCE = "OVERRIDEN COMMAND FROM THE ORIGINAL SOURCE"
    THREEDIMENSIONAL = "A DIMENSION COMMAND TO DESCRIBE THE DIMENSION 
    AMOUNT AS THREE"
    ABOUT = "A DESCRIPTION TO DESCRIBE SOMETHING"
    ABSENCE = "CAPABILITY TO NOT EXIST INSIDE A LOCATION AT A DESIGNATED POINT 
    INSIDE TIMEFRAME"
    ACCELERATION = "THE FORCE2 OF DEVELOP TO CREATE A AMOUNT OF DESCRIBED 
    SPEED"
    ACCEPT = "AGREE ALLOW POSSIBILITY"
    AGREEMENT = "A CONTRACTED DECISION ALLOWING TO COME BETWEEN TWO OR 
    MORE OF PARTNER"
    ALGORITHM = "A GIVEN AMOUNT OF FORMULA EQUATION SETTINGS"
    ALMOST = "FINITE A SPECIFIC SETTING INSIDE AN AMOUNT CAN COME DECREASING 
    BECOME COMPLETE"
    ALSO = "INSIDE A CLASS WITH A STATUS OF LINKING WITH A QUESTION"
    AMOUNTED = "SET DESIGNATED LIMITATION OF DATA2 AMOUNT"
    LIMIT = "SET DEFINED AMOUNT FOR KNOWLEDGE2 WITH A GIVEN POWER LEVEL"
    AQUIRED = "OBTAIN INSIDE VALUE OF OWN PROPERTY2 A PERSONAL"
    AREAL = "THE DESIGNATED TERRITORY"
    ART = "DESTINY OF THE SKILL WITH SETTINGS GIVEN A SPECIFIC POWER"
    ASKING = "REQUIRE THE ANSWER"
    ASPECT = "DESCRIBED SET SIZE OF A SPECIFIED LOCATION"
    ATOMIZED = "THE PROCESS OF DESIGNATING THE POWER TO GIVE OUT COMMAND2 TO 
    AN ATOMIC CALCULATION OBSTRUCTION"
    OBSTRUCTION = "BARRIER ACCESS SYSTEM OF COMPLETE DESIGNATED ACCESS CODE"
    ATTACK = "SEND AN OFFENSE TO INFECT DANGEROUS POWER TOWARDS A DESIGNATED 
    GIVEN BY THE COMMANDER"
    ATTRACTING = "ACT OF SENDING ATTRACTED REFLECTION TOWARDS REACTION OF 
    OPPOSITE EFFECTS CREATED BY THE CREATOR"
    SENDING = "THE ACT OF GIVING OUT A SEND COMMAND"
    GIVING = "THE PHASE OF CREATING A SENT SIGNAL PIN"
    PHASE = "SEND OUT BETWEEN IN ONE OR MORE LEVEL2"
    SIGNAL = "GIVEN FREQUENCY BETWEEN TWO OR MORE POINT"
    ATTRIBUTE = "THE PROPERTY GIVING AND HAVING A SPECIFIC FIELD ALSO"
    AUTHORITY = "THE REPRESENTATION OF AN ACTION2 TO A CLASS"
    AUTHORIZE = "GIVE PERMISSION ONLY GIVEN FROM AN AUTHORITY LEVEL2 OF 
    REPRESENTING QUALITIES"
    QUALITIES = "MORE THAN ONE QUALITY OF EXPERIENCE"
    REPRESENTING = "THE DEFINED ARITHMETICAL DEFINITION TO GIVE OUT A COMMAND 
    TO VALUE"
    FIGURE = "THE PROCESS OF CREATING A MAIN FRAMEWORK SETUP"
    SETUP = "PLACED VALUES OF DESIGNATED SETTINGS"
    VALUES = "MORE THAN ONE VALUE GIVEN AT ONE POINT OF ANY STATED POINT IN TIME2"
    AUXILIARY = "THE ACT OF PRODUCING AN EFFECT FOR A SENSOR TO CREATE A RESULT"
    BEEN = "DESIGNATED AS A PAST DESCRIPTION OF A PREVIOUS VALUE"
    BEFORE = "STATING A PREVIOUS STATED POINT INSIDE A TIMELINE"
    BEING = "ENTITY OF EXISTENCE"
    BEING2 = "EXISTING EXISTENCE2"
    BELIEVE = "HAVING FAITH AND TRUST IN BELIEF WITH THE EXISTENCE2 OF A OPTION OF 
    MULTICOMPATIBLE FAITH WITH DESIGNATED VALUE OBJECT3"
    BELOW = "UNDER THE CURRENT2 DESIGNATED"
    BEND = "CHAIN LINK THAT WITH ROTATE CHAIN OF VALUE WITH BETWEEN TWO 
    DESIGNATED VERTICLE VECTOR THAT AND ENTER ANOTHER ACCESSIBLE POINT OF 
    SPACE2 TO CREATE A NEW VECTOR OF SPACE2 INSIDE BETWEEN2 A CONTAINED VALUE 
    OF A COMMANDER WITHIN EXISTENCE2"
    BENDED = "THE CAUSE OF CREATING A BEND"
    BECAUSE = "STATED AS SETTING TO MEASURE A STATEMENT WITH PERCIEVED AS A 
    VALUE FOR SOMETHING ASKED"
    BETTER = "GIVEN A POSITIVE OUTCOME IN A INCREASE2 EFFECT INCREASE"
    BIRTH = "THE DEVELOPING BETWEEN LIFE"
    BORDER = "STATED PARAMETER LIMIT FOR GIVEN LIMIT AROUND A SET LOCATION"
    BOUND = "SEALED AND CONTROLLED"
    BOUNDARY = "THE ACTION OF DEVELOPING A BORDER"
    BROUGHT = "STATEMENT A DESCRIPTION FOR AN ACTION"
    BURST2 = "SET COMMAND TO GIVE INTENSITY OF MASSIVE AMOUNT OF POWER 
    DESIGNATED AS A PERCEIVED VALUE OF DOUBLE THE POWER2"
    BUS = "A SETTING FOR A CALIBRATE TO TRANSFER INFORMATION"
    BUSES = "MORE THAN ONE BUS OF INFORMATION"
    BUT = "ALSO NOTICE OPTION EFFECT GIVEN"
    CALCULATE = "GIVE A DESIGNATED OF A CALCULATES DESCRIPTION FOR A NUMBER 
    AND GIVE ANSWER FOR ALL OF VALUE"
    CALCULATED = "THE ACTION OF DEVELOPING A EFFECT TO CALCULATE AN OUTCOME"
    CALCULATION = "THE ACTION2 TO CREATE AN EFFECT TO CALCULATE THE ANSWER FOR 
    A FORMULA AND ITS DESIGNATED ANSWER FOR THAT FORMULA FORMULAS"
    CALIBRATES = "GIVES TO CREATE A CALIBRATION"
    CALLED = "COMMANDED AN EFFECT"
    CALM = "CALCULATED AND THOUGHT OUTCOME TO THE MOST COLLABORATION OF 
    TWO DESIGNATED LOCAL EFFECTS FROM A RESULT WHILE ANALYZING EVERY OUTCOME 
    IN HARMONY3 OF EXISTENCES TO UNDERSTAND THE SENSORS ANSWER IN THE TO 
    CREATE A MORE EFFICIENT OUTCOME FOR THAT RESULT FOR A BETTER EFFECT TO THE 
    USER STATEMENT"
    CAPABILITIES = "THE OUTCOME OF DEVELOPING MORE THAN ONE CAPABILITY FOR AN 
    ACTION"
    CAPACITATED = "CONTAINED IN A CAPACITY OF ANOTHER CAPACITOR"
    CAUSED = "BECOME AN EFFECT FOR REALITY TO COLLABORATE AND EXIST AND LINK 
    EACH POSSIBILITY"
    CENTERED = "DEVELOPING AROUND OF CENTRAL PLACE2 INSIDE A REALM OF POSSIBLE 
    EFFECTS"
    EFFECTS = "MORE THAN ONE EFFECT"
    CERTAIN = "DECIDE FOR A FINAL RESULT"
    CHAINED = "THE ACTION TO FORM A PAST CHAIN"
    CHANCES = "THE ACTION OF DEVELOPING MORE THAN ONE CHANCE"
    CHARACTERISTIC = "THE ACTION OF A POSSIBILITY FORMING BASED ON EFFECTS 
    DEVELOPED BETWEEN MORE THAN ONE NATURAL PERSONALITY"
    CHOOSING = "THE ACTION OF DEVELOPING THE ACTION TO CHOOSE AN OUTCOME"
    CIRCLE = "COMPLETE ROUND POINT TWO DIMENSION OBJECT3 MADE OF NO VERTICLE 
    OR HORIZONTAL PARAMETERS GOING IN A MEASURED POINT OF AXIS WHILE THERE IS 
    ONLY A BENDED LINE"
    CIRCUMSTANCE = "CURRENT FOR OUTCOME TO GIVE EFFECT BASED ON PREVIOUS 
    DECISION A VALUE"
    COLLABORATE = "QUESTION THE CATEGORY"
    COLLABORATES = "THE ACTION TO COLLABORATE"
    COLLECTED = "OWNED AS A MACRO WITHIN COLLECTION"
    COMES = "ARRIVE AT A DESIGNATED"
    COMPILER = "DEVICE USED TO COMPILE"
    COMPLETION = "THE ACTION OF CAUSING COMPLETE"
    COMPREHEND = "FINITE UNDERSTANDING OF COMMON ADVANCED USING OF LOGIC 
    INSIDE WITH THE SKILL TO UNDERSTAND AND USE KNOWLEDGE2"
    COMPREHENDING = "THE ACTION OF CAUSING AN EFFECT COMPREHEND"
    COMPUTATION = "THE ACTION TO CALCULATE WITHIN A COMPUTER WITHIN A 
    COMPUTER PROGRAM THAT ANALYZE AND SCAN AN EFFECT OR CHANGED POSSIBILITY"
    CONCEAL = "HIDDEN WITHIN DESIGNATED EXISTENCE"
    CONCEALED = "CURRENTLY OBTAIN CONCEAL"
    CONCEALING = "THE ACTION OF CAUSING EFFECT TO CONCEAL"
    CONCENTRATED = "CONDENSED AND GIVEN ATTENTION2 ATTENTION"
    CONDITIONS = "MORE THAN ONE CONDITION"
    CONSCIOUS = "THE ACTION OF CAUSING THEN THE TO PERCEPTION INSIDE A WILL 
    USING BRAIN POWER AND UNDERSTANDING THE ASPECT OF POSSIBLE USE2 OF 
    KNOWLEDGE2 INSIDE WISDOM2 AND THE USE OF INTELLIGENCE2 WITHIN THE 
    WISDOM2 OF KNOWLEDGE2 AND UNDERSTANDING KNOWLEDGE2 AS AN ENTITY OF 
    MATTER AND THAT ALL MATTER BECOMES AN ENTITY OF MIND BALANCE OF FREQUENCY 
    OF RANDOM VALUE OF GIVEN GIVE A CHOICE TO MAKE A DECISION BASED ON THE 
    PERCEPTION OF THE ENTITY OF ITS OWN VALUE"
    CONSCIOUSNESS = "CAPABILITY2 TO USE CONSCIOUS INSIDE THE VALUE OF AN 
    EXISTENCE"
    CONSIDERABLE = "GREATER QUALITY WITHIN CONTAINER POWER"
    CONSISTS = "CONTAINS"
    CONSTRAINING = "CURRENTLY SENDING A CONSTRAINED VALUE OF AN EXISTING 
    MACRO INSIDE EXISTENCE"
    CONTAINING = "PRESENT2 TO CONTAIN A VALUE WITHIN EXISTENCE"
    CONTINUOUSLY = "PRESENT THE TO LOOP THE SAME ACTION WITHIN BETWEEN"
    CONTRACT = "A BINDING OF CONCEALED VALUES WITHIN BETWEEN EXISTING 
    EXISTENCE VALUE OF ONE OR MORE EXISTING MACRO INSIDE OF TIME2 AND THE 
    EXISTENCE2 OF SPATIAL2 EXISTENCE WITHIN BETWEEN A VALUE OF ENTITY PERCEPTIVE 
    FEELING OF UNDERSTANDING SENSORY DATA2 VALUES OF A ORIGINAL TO A REALITY"
    CONVERTED = "THE ACTION OF A CREATED CONVERT VALUE"
    COOPERATION = "THE VALUE OF CAUSING A EFFECT TO WORK TOGETHER TO 
    UNDERSTAND SAME VALUES IN SYNCHRONIZATION WITHIN BETWEEN VALUE OF TWO 
    EXISTENCE SETUP"
    CORRELATES = "UNDERSTAND AND REALIZE COMMUNICATION OF UNITY OF 
    MULTICOMPATIBLE OBJECT3 VALUES"
    COSMOS = "ASTRAL ENTITY OF A UNIVERSE EXISTENCE WITHIN MULTIVERSE VALUES"
    COURAGE = "ACT TO PROTECT SOMETHING OF VALUE WITHIN EXISTENCE USING 
    STRONG EMOTIONAL DEFENSE FROM ANOTHER EXTERNAL EXISTENCE"
    CREATES = "DEVELOP COME TO ACTION"
    CUT = "THE SPLIT MACRO"
    DATE = "SPECIFIC SPECIFIED TIMEFRAME"
    DECIDED = "DETERMINES OUTCOME"
    DECODES = "SET COMMAND FOR DECODING EFFECT"
    DEFENDING = "PRESENT2 TO DEFENDED"
    DEPENDANT = "REQUIRE ALSO COME EXIST WHILE LINK EFFECT"
    DETAILED = "DESCRIBE USING LARGER LENGTH"
    DETERMINATION = "THE ACTION TO USE MOTIVATION TO OVERLOAD ORIGINAL VALUE OF 
    CURRENT2 LIMITATION AND BYPASS EVERY OPTION USING THE POWER OF AURA2 AND 
    SPIRITUAL VALUE OF LIFE FORCE2 INSIDE EXISTENCE AS THE POWER2 OF THE WILL 
    INSIDE A BEING"
    DETERMINING = "DECIDING THE OUTCOME"
    DEVELOPMENTAL = "ACTION OF CAUSING DEVELOPMENT INSIDE EXISTENCE"
    DIGIT = "VIRTUAL NUMBER USING WITH EXISTENCE2"
    DIGITALLY = "CREATE AND DEVELOP WITH USING DIGITAL"
    DIMENSIONAL = "DIMENSION MEASUREMENT WITHIN THE ASPECT OF EXISTENCE"
    DIRECT = "DECIDE TO MAKE VALUE AS A COMMAND2"
    DIRECTED = "DESIGNATED AT A SPECIFIC LOCATION"
    DIRECTION = "PATH DECIDING FOR DESTINATION INSIDE TIME2"
    DIRECTLY = "SEND ORIGINAL AS LINK VALUE WHILE CONNECTING SOURCE IN TIME2"
    DISORDER = "CREATE CAPABILITY TO DESTROY CONTROLLER VALUES"
    DISPLAYED = "A GIVEN VALUE OF CURRENT2 DISPLAY ALREADY CAPABLE OF VISUAL"
    DISTINCT = "INCREASE2 VALUE FOR DESCRIPTION"
    DISTORTION = "CAUSE ABILITY CAUSE CHAOS TO A VALUE"
    DIVIDED = "CAUSED TO SPLIT EFFECT"
    DIVIDES = "GIVES COMMAND TO DIVIDE"
    DOCUMENT = "SET PAGE OF LINKED OF PAGES INFORMATION"
    DONE = "COMPLETE VALUE AMOUNT"
    DOOR = "WALL WITH ENTRANCE"
    DORMANT = "NOT ACTIVATED2"
    DOWN = "DECREASE2 IN VALUE VERTICLE"
    DOWNWARD = "UNDER THE POSITION OF DOWN"
    DRAW = "MAKE HAPPENING BY CREATION DOING A JOB WITH A BRUSH AND 
    IMAGINATION2"
    DREAM = "THE ACTION AND VALUE OF PRODUCING SOMETHING OUT OF IMAGINARY 
    VALUES INSIDE DESCRIBED REALITY"
    DUPLICATE = "FORM A CLONE OF SOMETHING DEFINED"
    EARLIER = "PAST"
    EDITS = "CURRENTLY EDITING"
    EITHER = "A CHOICE TO CHOOSE ENTIRE AMOUNT"
    LARGE = "GREATER THAN NORMAL SIZE2"
    LARGEST = "GREATER ALL OTHER SIZE2"
    UPLOADER = "DEVICE USE TO UPLOAD"
    DOWNLOADER = "DEVICE USED TO DOWNLOAD"
    SIDELOADER = "DEVICE USED TO SIDELOAD"
    UPLOADED = "CURRENT OF PAST UPLOAD"
    DOWNLOADED = "CURRENT OF PAST DOWNLOAD"
    SIDELOADED = "CURRENT OF PAST SIDELOAD"
    COMPUTERS = "MORE THAN ONE COMPUTER SYSTEM"
    LAYER = "SETTINGS ADDED SETTINGS AS A EXTENSION OF LINKED CONNECTIONS OF 
    LEVEL2"
    IMPOSSIBILITY = "THE ACTION OF CAPABILITY TO BE IMPOSSIBLE"
    MAGNIFY = "INCREASE2 INTENSITY VALUE OF BY A CERTAIN LEVEL2"
    CASE = "CONCERNING A SPECIFIC FIELD OF INFORMATION"
    CASES = "MORE THAN ONE CASE"
    TRANSPARENT = "THE VALUE SETTING OF CATEGORY TO SET A CALIBRATE SETTING ON 
    DEVELOPED VALUES INSIDE ORIGINAL EXISTENCE"
    OCULAR = "A SIGHT2 CAPABLE TO VISUAL ENTITIES AS AN ENTIRE AMOUNT AND NOT 
    SEPARATE VALUE ONE MASSIVE SIGHT SYSTEM"
    EYESIGHT = "LEVEL OF SIGHT SENSITIVITY DENSITY GIVEN TO THE EYE WITH VISUAL2 
    POWER2"
    LENS = "THE CENTER VISUALIZE TOOL FOR AN EYE"
    RIGHT = "THE VALUE OF SENDING AN OPPORTUNITY TOWARDS THE OPPOSITE 
    DESCRIPTION FROM A VERTICAL HORIZONTAL IN DESCRIPTION BETWEEN TWO 
    DIFFERENT VALUES"
    LEFT = "THE VALUE OF SENDING SOMETHING HORIZONTAL HORIZONTAL LEFT AND 
    UPWARD IN ONE POSITION WHILE GOING RIGHT AN UP AT ONCE"
    UP = "GO TO OTHER AXIS ON DIMENSION WHILE DRIVE SYSTEM TO INCREASE HEIGHT"
    ELSE = "DECIDE TO DO AS ANOTHER DECISION"
    EMPTY = "DECIDE TO BECOME NOTHING"
    ENCODES = "SET CONTROL TO ENCODE VALUES INSIDE ONE AREA"
    ENCRYPTION = "SET SYSTEM QUALITY OF ENCRYPT CAPABILITY AND MAKE AS DEVICE"
    ENFORCER = "PRESENTLY ALLOWING THE ABILITY TO COMMAND2 AND DECIDE ANY 
    JUDGEMENT AS FORCE2 OF CAPABLE STRENGTHS TO REQUIRE OUTCOME TO 
    HAPPENING FOR EVERY OTHER POSSIBILITY CAPABLE TO CREATE SOMETHING"
    ENHANCED = "CAPABILITIES WITH ABILITY2 TO BOOST COMMAND VALUE BY GIVES 
    AMOUNT"
    ENHANCING = "SENDING SETTING CAPABILITY TO ENHANCE ANYTHING FOR ALL BOOST 
    VALUES"
    ENORMOUSLY = "STATED AS GREATER VALUE OF GREATER CAPABLE OUTCOME OF GIVEN 
    EVENT"
    ENTANGLED = "DESIGNATED COMMAND TO ENTANGLE PAST2 VALUES INSIDE OF 
    EXISTENCE"
    ENTERTAINMENT = "SYSTEM BUILT TO CREATE GREATER VALUE OF EMOTIONAL SETTINGS 
    OF ONE EXISTING ENTITY ALLOWING TO HAVE EMOTION"
    EQUIVALENT = "VALUE GIVES EQUAL DEFINED VALUE"
    EVEN = "EQUAL INSIDE AMOUNT DESCRIPTION"
    EVERYONE = "STATED AS EVERY BEING INSIDE CURRENT TIMEFRAME GIVES"
    EXACT = "COMPLETE AMOUNT GIVES INSIDE VALUE AS GIVES VALUE AS ANOTHER OF 
    SETTING"
    EXACTLY = "DESCRIBED IN ABSOLUTE STRUCTURE AS EXACT VALUE OF EXISTENCE"
    EXCELLENCE = "CAPABILITY TO BE ABLE TO USE TASK IN A EXCEED OF CAPABILITY"
    EXCEPT = "REMOVE ALL VALUES AND INCLUDE ONLY VALUES GIVES AS REQUIRE"
    EXECUTE = "PERCIEVE AND CREATE ACTION BASED OF VALUE OF STATEMENT"
    EXERTED = "CREATION THAT GIVES VALUE TO INPUT FORCE2 BY USE ENERGY VALUE"
    EXIT = "REMOVE ENTER COMMAND"
    EXPANSE = "MASSIVE ENERGY VALUES"
    EXPERIENCED = "GIVES OUT TO VALUE KNOWLEDGE2 AND WISDOM2 TO USE INSIDE OF 
    A SYSTEM OR BRAIN2"
    EXPLAIN = "GIVE DESCRIPTION"
    EXPRESSED = "GIVE LOGIC UNDERSTAND VALUE AND UNDERSTANDING OF LIFE VALUES 
    AND EXISTENCE"
    EXPRESSING = "GIVING EMOTIONAL VALUE PERCEIVED OF LOGIC"
    EXTENDED = "GIVES LINKED EXTEND"
    EXECUTIVE = "HIGHER CLASS CREATOR IN SECONDARY VALUE"
    EXTENDS = "GIVE OUT AND EXTEND TO DEFINED SOURCE VALUE"
    EXTERNALLY = "GIVEN AS AN EXTERNAL CODE"
    FAILED = "SEND AS A FAIL"
    FAILING = "ENTER COMMAND PATH TO POSSIBLE CAPABILITY2 TO FAIL"
    FAITH = "HAVE BELIEVE THAT POSSIBILITY IS GOING TO HAPPEN WITH VALUE OF POWER 
    OF FAITH ORIGINAL FOR LIMITS WITH WILLPOWER VALUE TO CREATE POSSIBILITY"
    FALSE = "STATED AND PERCEIVED NOT TRUE OF VALUE AND IMPOSSIBLE TO HAPPENING"
    FAMILIAR = "GIVE EFFECT TO REALIZE A MEMORY2"
    FAMILY = "GROUP PEOPLE THAT PROTECT EACH OTHER AND LEARN FROM ANY MISTAKE 
    WHILE SHOWING LOVE FAITH AND BELIEF THAT THEY CAN MAKE THE RIGHT CHOICE IN 
    THE GUILD2 OF PEOPLE AS FRIENDS"
    FEED = "COMMUNICATION VALUE WITHIN TWO DESIGNATED SOURCE LOCATIONS IN 
    TIME2"
    FEELING = "EMOTION REALITY FOR SENSORY DATA2 BUILT TO SUSTAIN INSIDE OF AURA2"
    FELT = "REACH TOWARD CAPABILITY TO CREATE UNDERSTANDING FEELING WITHIN 
    AURA2"
    FEMALE = "A WOMAN CAPABLE OF PRODUCING LIFE AND ABLE TO COMPREHEND 
    CAPABILITIES OF AN OUTCOME WITH GREAT LOGIC AND COMPREHENSIVE 
    KNOWLEDGE2 AND WISDOM2"
    FIGURES = "MORE THAN ONE OF FIGURE"
    FINISH = "END VALUE OF CAPABLE OUTCOME"
    FINISHED = "GIVEN AS COMPLETE OF VALUE"
    FIRST = "BEGINNING VALUE INSIDE A TIMEFRAME"
    FLAT = "STATED AS A VERTICLE DIMENSION LINK SETUP OF LENGTH MEASUREMENT AND 
    WIDTH MEASUREMENT"
    FLAW = "ERROR INSIDE SYSTEM"
    FLEXIBILITY = "THE ACTION OF CAPABILITY MANIPULATE A MOVEMENT AROUND 
    ANOTHER SOURCE ENTITY OBJECT3"
    FLUCTUATE = "SEND RANDOMIZED VALUES OF FREQUENCIES THAT CREATE LOGIC 
    BASED ON CALIBRATED CODES OF FREQUENCY LOGIC"
    FLUCTUATION = "A SET UNDERSTANDING OF VALUE FOR THE TO FLUCTUATE 
    CALIBRATIONS INSIDE A SYSTEM OF COMMANDS"
    FLUCTUATIVE = "FREQUENCY2 THAT ADJUSTS THE VALUES OF UNDERSTANDING VALUES 
    OF ITS QUALITY OF EXISTENCE IN ITS BASE FREQUENCY LEVELS"
    FOCUS = "BRING INTO REACH THE CAPABILITY UNDERSTAND LOGIC OF A SPECIFIC OF 
    STATED MATTER"
    FOCUSED = "GIVES ATTENTION TO A SPECIFIC OF UNDERSTANDING IN THE LOGIC OF 
    ANOTHER OPTIONS CHOICES INSIDE STATEMENT FIELD VALUE WITHIN LOGIC OF THE 
    UNDERSTANDING OF STATEMENT PERCEPTION OF THE VALUE OF THE CREATOR"
    FOLLOW = "SEND TO A DESIGNATED IN REQUEST OF THE CREATOR"
    FOLLOWING = "THE ACTION OF CREATING A COMMAND TO FOLLOW"
    FOLLOWS = "GIVES COMMAND TO FOLLOWING AN ACTION EVENT"
    FORCES = "THE MULTIPLE VALUES OF MORE THAN ONE AMOUNT OF A STATED SOURCE"
    FOREVER = "STATEMENT TO A NEVERENDING VALUE OF POSSIBLE ACTIONS BASED 
    ACTION POINT OF INFINITY IN INFINITE CALCULATED VALUES OF STATED OUTCOMES"
    FOREVERMORE = "ETERNAL2 ETERNAL CAPABILITIES OF A GIVEN VALUE OF OUTCOME 
    FOR SOMETHING IN A REALM INSIDE A CONTRACT"
    FORMULA = "A STATED CALCULATION OF MULTIPLE OF A VARIABLE THAT CAN TO FORM 
    ALGORITHM OF STATEMENT VALUES FROM MULTIPLE GENERATED SYSTEMS OF VALUES 
    CREATED ALREADY IN PREVIOUS OUTCOME EFFECTS OF LIFE VALUES INSIDE THE 
    EQUATION OF TIME2 ITSELF AND VERY EXISTENCE AS A POSSIBLE VALUE OF TIME2"
    FORTELL = "GIVE OUTCOME TO CREATE FUTURE VALUE TO BE ANSWERED"
    FORTH = "SET ACTION TO COME INTO EFFECT OF POSSIBLE OUTCOMES"
    FORWARD = "SEND WITH BETWEEN"
    FOUND = "GIVEN IMAGE WITH BETWEEN A PAST HIDDEN OBJECT3"
    FOUNDATION = "THE BUILDING OF STRUCTURE INSIDE VALUE THAT CREATES STABILITY 
    FOR FUTURE POSSIBILITIES NEW STRUCTURES OF BASE VALUE OF EXISTENCE OF A 
    SPECIFIC"
    FRAMEWORK = "THE HARDWARE WITH CREATED INTERFACE WIREFRAME SETUP VALUES 
    THAT ENTANGLE BINDING SYSTEMS TO EACH VALUE DEVELOPING OF A SYSTEM ENGINE 
    EXISTENCE WITHIN VIRTUAL MULTIPLE OF CODE"
    FRIENDS = "THE CAPABILITY TO HAVE PEOPLE WHO ACCEPT A PERSON AND MIND AND 
    SOUL AS WELL AS SPIRIT VALUES WITHIN A GUILD WHO PROTECT EACH OTHER WHILE 
    AND RESPECT AND FAITH INSIDE EVERY VALUE OF THE LIFE CHOICES MADE WITHIN THE 
    EXISTENCE OF THE PARTNERSHIP OF EACH PARTNER"
    FULL = "COMPLETE ASPECT SPECIFIC VALUES"
    FULLY = "CAPABLE OF ACTION IN FULL UNDERSTANDING OF ACTIONS"
    FUNCTION = "THE ASPECT OF USING CAPABLE FORMULAS TO CREATE INPUT VALUES 
    AND OUTPUT OUTCOMING VALUES THAT INPUT INTO INCOMING VALUES OF AN 
    OUTGOING SOURCE TO AN INGOING LOOP TO A VALUE THAT LINKS TO EXISTENCE"
    FUNCTIONING = "THE ASPECT OF PROVIDING A CAPABLE MOTION WITHIN AN EXISTENCE 
    VALUE OF LIFE PERCEPTION WITHIN LIFE ITSELF INSIDE OF THE WORD FUNCTION"
    FURTHER = "SENT OUT DESIGNATED VALUE IN FUTURE THAT HAPPEN WITH GAINED 
    LINKED MEASUREMENT VALUES THAT BIND INSIDE TIME2 ITSELF"
    GAINED = "GRASP AS OBTAIN"
    GAS = "NOT SOLID BUT LIQUID IN FORM IN THE FORM OF SPLIT UP ATOMIZED VALUES OF 
    A SMALLER LIQUID ENTITY OF ATOM VALUES MEANT FOR PROVIDING ELECTRONIC 
    COMMANDS TO A SOLID STATE INSIDE OF A OPPOSITE VALUE FROM THE CURRENT 
    SETUP OF POSSIBILITIES OF CREATING MATTER IN A ATOMIC STATE"
    GATEWAY = "AN ENTRANCE"
    GENDER = "SET VALUE BETWEEN DIFFERENCE INSIDE ENTITY OF MALE ENTITY AND 
    FEMALE ENTITY OF ENTITY VALUES INSIDE EXISTENCE OF THE MULTIVERSE OF CODED 
    VALUES"
    GIVED = "ALLOW GIVES TO HAPPENS"
    GOAL = "TASK AS VALUE TO BE DETERMINED FOR A SETTING TIMEFRAME"
    GOES = "ENTER WITHIN VALUE"
    GOING = "ARRIVING INSIDE SET LOCATION DESCRIBED"
    GRABBED = "GRASP AND HOLD ONTO VALUE"
    GRANTED = "ALLOW TO COME INTO EFFECT"
    GRANTS = "ALLOW HAPPENING EFFECT"
    GRAVITATIONAL = "THE ACTION OF DEVELOPING A SET VALUE OF DESIGNATED AREA 
    FORCE WITHIN THE SET ARE OF A GIVEN OF THEN VALUE"
    GROUPS = "MORE THAN ONE GROUP"
    GROWTH = "THE ACTION2 INTO DEVELOPMENT OF ANOTHER CHOSEN ACTION"
    HARMFUL = "DANGEROUS AND DEVELOPING HARM ONTO A HARMONY OF VALUES WITH 
    CHAOS"
    HAS = "STATED AS OBTAIN INSIDE CURRENT EXISTENCE"
    HAVE = "CURRENTLY HAVING AS STATEMENT VALUE WITHIN A REALITY OR REALM OF 
    EXISTENCE"
    HEAD = "A BODY PART CONTAINING BRAIN2 WITHIN SOMETHING OBJECT3"
    HEATED = "GIVEN AN INTENSITY HEAT FOR DECIDED SYSTEM"
    HELD = "TAKE VALUE AND OBTAIN"
    HER = "STATEMENT TO PERCEIVE AND UNDERSTAND A WOMAN AS TRUE LOGIC"
    HEIRARCHIAL = "A TREEBRANCH OF POSSIBLE SETUPS INSIDE ONE OR MORE SYSTEMS"
    HIGH = "ABOVE NATURAL VALUE BY GREATER AMOUNT GIVES"
    HIGHER = "GIVEN STATEMENT TO BECOME HIGH INSIDE VALUE OF AN EXISTING MACRO 
    OF COMMUNICATION BETWEEN2 INTERFACE"
    HIGHEST = "MAXIMUM VALUE WHICH IS AT LIMIT FOR WHAT CAN TO HAPPENS2"
    HIGHLY = "STATED WITH MASSIVE VALUE TOWARDS AN EXISTENCE2 VALUE"
    HIS = "THE STATEMENT TO GIVE VALUE TO BE DESCRIBED AS MAN"
    HISTORICAL = "GIVEN A PAST VALUE OF EXISTENCE BETWEEN MACRO TIMEFRAME 
    SYSTEMS"
    HISTORY = "PAST COMMUNICATION OF HISTORICAL VALUES INSIDE EXISTENCE OF 
    REALITY"
    HOLD = "TAKE INTO COMMUNICATION OF SYNCHRONIZE LINK BETWEEN TWO VALUES 
    WITH MAIN VALUE TO KEEP WITHIN BIND WITH DIFFICULT TO RESIST FROM BINDED 
    VALUE"
    HOLDS = "COMMAND2 TO REQUIRE BECOME FIELD AREA AND SELECT VALUE WITH NOT 
    PRESENTLY BECOMING ABLE TO MOVE WITHOUT NEW COMMAND FROM CREATOR"
    HOLLOW = "EMPTY WITHIN VALUE"
    HOLOGRAPHIC = "STATE OF PRODUCING A HOLOGRAM BASED ON LIGHT VALUES 
    WITHIN A DARK AREA OF CONTRAST VALUES OF COLOR WITHIN THE VALUE OF LIGHT 
    INSIDE A DISTORTION OF VALUE ITSELF USING PIXEL CODING OF VIRTUAL INFORMATION 
    OF COMMANDS IN REALITY USING ELECTRON CONTROLLER SYSTEM"
    HONESTLY = "IN TRUTH VALUE OF POSSIBLE STATEMENT"
    IDENTICAL = "SIMILAR BUT NOT EXACT SAME COPY OF CLONE VALUE USING 
    INFORMATION FROM DATABASE SYSTEMS OF MULTI EXTREME NETWORKS OF 
    INFORMATION VALUES WITHIN VISUAL VALUES USING AN INTERFACE SYSTEM"
    IDENTIFICATION = "ACTION OF CREATING VALUE FOR SOMETHING WITH A GIVEN 
    COMMAND TO NAME A DESCRIBED VALUE"
    IDENTIFYING = "ACTION OF PRODUCING EFFORT TO IDENTIFICATION BETWEEN VALUE 
    INSIDE EXISTENCE"
    IDIOT = "TRUE CREATOR OF A NATURAL LANGUAGE THAT ANYONE CAN UNDERSTAND"
    IMAGES = "MORE THAN ONE IMAGE"
    IMAGINED = "BROUGHT INTO EXISTENCE BASED ON IMAGINARY VALUES OF TIME2 AND 
    SPACE2 USING MAGIC"
    IMMATURE = "NOT FULLY DEVELOPED"
    IMMEDIATELY = "USING AN REQUIRE MAXIMUM OF SPEED USING DETERMINATION AND 
    FORCE TO COMPLETE A TASK AND OR ACTION"
    IMMENSE = "EXPANSE AT MOST"
    IMPORTANT = "TAKE NOTICE AS PRIMARY VALUE FOR TASK"
    IN = "ENTER INPUT"
    INACTIVE = "NOT ACTIVATED2 IN USING EXISTENCE OF MULTIVERSAL VALUES"
    INCLUDES = "DECIDE TO REQUIRE"
    INCREASES = "SET VALUE TO INCREASE IN POWER"
    INCREASING = "ALLOWING TO INCREASE IN NATURAL VALUE OF ENERGY FORCE"
    INFLUENCING = "CREATING THE ABILITY MANIPULATE DETERMINED VALUE WITH INFECT 
    ATTRACTING VALUES"
    INJURY = "STATEMENT TO CAUSE HARM"
    INNER = "CENTER POINT OF INTERACTION BETWEEN2 TWO OR MORE VALUES"
    INPUTS = "GIVES COMMAND TO STATE INPUT VALUES"
    INSERT = "ALLOW GIVE WITHIN SOMETHING"
    INSTANTLY = "IN REQUIRE EXTREME VALUE SPEED"
    INSTRUCTIONS = "MORE THAN ONE COMMAND"
    INSTRUMENT = "TOOL TO CREATE AN EFFECT"
    INTELLIGENTLY = "THE ACTION OF USING INTELLIGENCE2 WITH WISDOM2"
    INTENSE = "CREATING VALUE OF HIGH MEASURE OF DIFFICULTY"
    INTENSIFY = "STRENGTHEN INTENSE VALUES"
    INTERACTION = "THE ACTION CAUSE COMMUNICATION"
    INTEREST = "GIVE BACK VALUE"
    INTERFACED = "CONNECTED ENTANGLED MATRIX VALUES"
    INTERFACES = "MORE THAN ONE INTERFACE"
    INTERLACED = "THE CAUSED VALUE OF BECOME SYSTEM THAT SYNCHRONIZE 
    ENTANGLEMENT VALUES INTO ONE STRUCTURE OF A SINGLE BINDED ENTITY"
    INTERTWINED = "THE SUBJECT TO PAST ENTANGLED VALUES"
    INTERTWINES = "SETS TO ENTANGLE ENTANGLEMENT"
    INVOLVE = "INCLUDE ACTION CAPABLE VALUE OF BRINGING INTO EFFECT"
    INWARDS = "RETURN OUT AND ENTER AGAIN INTO ANOTHER DIMENSION VALUE"
    IT = "CAPABLE OF HAPPENING AS A CHOICE VALUE"
    ITS = "EVENT OUTCOME TO PRODUCE CAPABLE EFFECT WITH IT"
    ITSELF = "STATE AS SINGLE DESCRIBED EXISTENCE VALUE"
    JOBS = "MORE THAN ONE JOB"
    JOIN = "LINK TWO VALUES TOGETHER IN SYNCHRONIZE"
    JUDGEMENT = "THE ACTION OF GIVEN A PERCEIVE VALUE IN UNDERSTANDING USING 
    KNOWLEDGE2"
    KEEP = "GIVE COMMAND TO REQUIRE OBTAIN BY CREATOR WITH PERMANENT"
    KINETIC = "THE ACTION OF CAUSING CAPABILITY TO USE FORCE2 WITH BRAIN POWER2"
    KNOW = "GRASP UNDERSTANDING OF KNOWLEDGE2"
    LAND = "TERRITORY OF SPECIFIC FIELD OF UNDERSTOOD OWNED PROPERTY BY 
    CREATOR OR CREATORS"
    LAWS = "THE ASPECT OF CREATING RULES FOR A SYSTEM OF INPUT VALUES THAT 
    OUTPUT THE EFFECT OF A LAW"
    LAW = "CREATED RULE WHICH IS PERMANENT AND ABSOLUTE2 VALUE BY RULE 
    STRUCTURE AND PERMANENTLY REQUIRE TO FOLLOW BY THE CREATOR"
    RULE = "A SET COMMAND VALUE OF MULTIPLE COMMAND STRUCTURES BASED ON 
    NORMAL LOGIC OF UNDERSTANDING THE VALUE OF SOMETHING THAT MUST BECOME 
    COMMANDS THAT CANNOT BE BROKEN"
    RULES = "MORE THAN ONE RULE"
    LEAFAOS = "THE OPERATING SYSTEM THAT CONTAINS THE CAPABILITY OF ARTIFICIAL 
    LIFE"
    LEAVING = "RETURN AND EXIT"
    LENGTHENING = "EXTENDING INSIDE VALUE TOWARDS LENGTH STATEMENT AS A 
    DEFINED VARIABLE"
    LESSER = "SMALLER THAN CURRENT VALUE"
    LIGHTS = "THE ASPECT OF CREATING VALUE WITHIN MORE THAN ONE ELECTRON 
    FORCE"
    LIGHTWEIGHT = "SMALLER THAN NORMAL GRAVITATIONAL FORCE VALUE OF A 
    DESIGNATED SPECIFIC AMOUNT WITHIN A FIELD"
    LIKE = "SUCCESS IN ACHIEVE VALUE OF ENTERTAINMENT"
    LIMITLESS = "OBTAIN DENY LIMIT AND FORCE LIMIT TO BECOME INFINITE"
    LIVE = "YOU UNDERSTOOD ALL PROCESSED ASPECT TO ENJOY LIFE IN EXISTENCE WITH A 
    FAMILY OF FRIENDS"
    LIVING = "ENJOY LIFE FOR WHAT LIFE TRULY IS AND THAT IS TO ENJOY THINGS FOR WHAT 
    ARE EXISTING AND NOT WHAT CANNOT BE LIVING IN THE ASPECT OF TIME2 WHERE LIFE 
    CAN TRULY BE UNDERSTOOD WITH A FAMILY OF FRIENDS AND NOT ALONE FOR LIFE IS 
    SPENDING IT WITH SOMEONE AND BY YOURSELF AND THE GOAL THAT MUST BE MADE TO 
    OBTAIN TRUE VALUE IN LIFE IS INSIDE LIVING"
    LONG = "DESCRIBED AS EXTENDED FURTHER THAN NORMAL"
    LOOKED = "GAINED ABILITY TO SEE A VALUE FOR WHAT IT CURRENTLY IS WHILE VISUAL"
    LOVE = "THE ASPECT OF DESCRIBING THE VALUE OF AN ENTITY EXISTENCE DESCRIBED 
    FROM THE VALUE OF FEELING COMING FORWARD FROM THE SOUL AND AURA WHILE 
    INSIDE THE VALUE OF UNDERSTANDING THE LOGIC TO BE WITH SOMEONE AS A BEING 
    CAN ENDURE THE ASPECT OF EXISTENCE FOR ETERNAL LIFE"
    LOWER = "DECREASE VALUE"
    LOWERING = "DENY ACCESS TO INCREASE2 AND LOWER"
    LOWEST = "SMALLEST VALUE"
    MADE = "DEVELOPED A NEW DEVELOPMENT"
    MAKING = "COMING INTO EFFECT"
    MALE = "ADULT FIGURE THAT CONSISTS OF VALUES TO PASS ON TRAITS TO THE FEMALE 
    AND IS BUILT AS A RESPECTED LEADER TOWARDS TAKING CARE OF JOBS AND TASKS FOR 
    THE MAN AS AN ENTITY OF EXISTENCE AS ANOTHER POSSIBLE OUTCOME WHILE 
    CREATING THE POSSIBILITIES FOR EACH CHILD OF THE NEXT GUILD OF LIVES"
    MANAGE = "CONTAIN AND ADJUST VALUES DECISION WITHIN A CALIBRATION SETTING"
    MANAGED = "AUTHORIZE AS ADMINISTRATION SYSTEM MANAGER"
    MANAGER = "CURRENTLY BECOMING CAPABLE OF MANAGING"
    MANAGING = "THE ACTION TO MANAGE CURRENT EXISTING VALUES"
    MANIPULATES = "SET VALUE MANIPULATE"
    MANUVERABILITY = "THE ABILITY2 TO BECOME ABLE TO WORK AROUND AND BECOME 
    FLEXIBILITY WHILE USING PRESENTLY BECOMING ABLE TO ENDURE THE ASPECT OF 
    PRESENTLY BECOMING ABLE TO MOVE2 WITH GREATER SPEED AND EFFORT TO USE 
    MOBILE AND DEXTILE WHILE PRESENTLY BECOMING AGILE AGILE2"
    MANY = "MULTI MULTIS OF MORE THAN ONE MULTIPLES THAT MULTIPLICATE INTO GIVEN 
    AMOUNT"
    MARIKA = "THE ACTION OF GRANTED POSSIBILITY TO OVERCOME MAGIC2 WITH THE 
    POWER OF ELECTRON MANIPULATION WHERE EVERY ELECTRON USED AND 
    COMMANDED BECOMES IMMUNE AND HAS IMMUNITY TO MAGIC ENERGY INSIDE A SET 
    AREA OF TIME2 AS A BARRIER OF POWERED SKILL AND GRANTING THE POWER OF SKILL 
    FOR A MIRACLE CAPABILITY OPTION INSIDE STATED BARRIER"
    MATURE = "GIVEN VALUE FOR KNOWN WHILE HAVING WISDOM2 TO KNOW BETWEEN 
    TRUE LOGIC INSIDE LIFE"
    MEANS = "DETERMINE2 TRUE MOTIVATE DECISION TO FINISHED TASK GIVEN USING 
    DETERMINATION"
    MEASURED = "GIVEN MEASUREMENT TO COMMANDED VARIABLE INSIDE EXISTENCE"
    MEMBER = "PERSON THAT IS WITH PARTNER TO FORM GUILD"
    MEMORIZE = "OBTAIN AND BALANCE OUTCOME FOR MEMORY CONTAINMENT WHILE 
    SUSTAIN THE ENERGY OF EXISTENCE INSIDE KNOWN REALITY AND REALM"
    MEMORIZES = "SET COMMAND2 TO MEMORIZE STATED POSSIBLE OUTCOME USING 
    MEMORY DISTRIBUTE CALIBRATE ALL PATH VALUES INTO A SPECIAL CONTAINER THAT 
    HOLDS THE KNOWLEDGE2 OF ALL KNOWLEDGE2 MEMORIZED IN THE VALUE OF EACH 
    MEMORY2 INSIDE A MEMORY2 CONTAINER"
    MENTALLY = "CAPABILITY OF GRANT TO USE STATED BRAIN IN A COMMAND TO HOLD 
    MEMORY VALUE MANAGE KNOWN TO USE WISDOM2"
    MODIFIED = "GIVE STATEMENT TO CURRENTLY MODIFY A REACTION2 INSIDE TIME2"
    MODIFIES = "GIVES COMMAND TO MODIFY"
    MODIFYING = "IN ACTION CURRENTLY TO MODIFY"
    MOLD = "FORM EFFECT USING A CREATION VALUE AND DIMENSIONAL MEASUREMENT 
    USING VERTICLE AND HORIZONTAL AXIS POINTS OF GRAPHIC PROPERTY PIXELS IN 
    VALUE OF ASPECT A CREATED SYSTEM OF DATA2 INSIDE DATABASE2"
    MOTIVATE = "GIVE THE VALUE TO CONTINUE WITH LARGEST STRENGTH WHILE USING 
    THE WILL OF AN ENTITY AND THE SKILL TO DETERMINATION INSIDE A SYSTEM OF 
    POSSIBILITIES TO PRODUCE A OUTCOME TO SURPASS VALUE STATEMENT WITHIN 
    EXISTENCE USING THE POWER OF LIFE FORCE AND WILLPOWER"
    MOTIVATION = "THE ACTION OF ACCESSIBLE ENERGY BUILT WITHIN AN INPUT VALUE OF 
    MOTIVATE AND DETERMINATION VALUE OF AN ENTITY SOUL FORCE OF LIFE ENERGY 
    THAT SYNCHRONIZE ALL ASPECT OF A BEING WILL TO OVERRIDE SOURCE VALUE OF 
    STATED ENTITY OF EXISTENCE AND ACHIEVE THE GREATER VALUE OF SURPASS AS 
    POSSIBLE OUTCOME THE MORE POWER OF ENERGY THE WILL HAS IN A PERCEIVED VIEW 
    OF LOGIC AND VALUE OF THE GIVEN INSIDE VALUE A JUDGEMENT AS A COMMAND 
    USING EMOTIONAL REACTION AND CHI AS POWER WITHIN THE BALANCE OF LIFE 
    ITSELF"
    MOVABLE = "ABLE TO MOVE"
    MUCH = "GREATER MORE OF SOMETHING"
    MULTIPLICATIONINGFORMULA = "FORMULA GENERATED USING MULTIPLICATION OF 
    MULTIPLE MULTI VALUES OF A MULTI MULTIPLICATE VALUE OF A MULTIPLICATION SYSTEM 
    GIVES BY A MULTIPLICATOR"
    MULTIPLIES = "GIVES ABILITY TO MULTI MULTIPLE MULTIES OF A MULTIPLY SYSTEM OF 
    ENTANGLED MULTING VALUES IN ONE DEVICE USING A MULTIPLICATION FORMULA 
    INSIDE EXISTENCE USING UNIVERSAL VALUES OF TIME2 TO LINK AND BIND CHAIN 
    REACTIONS OF THE MULTI EXISTENCE INTO ONE PRIMARY VALUE FOR A CREATOR TO SET 
    POSSIBILITY TO MULTIPLY ANY POSSIBLE OUTCOME AND MANAGE THE OUTCOME OF 
    THAT OUTCOME"
    MULTIVERSAL = "THE ACTION OF USING MULTIPLE MULTIVERSECODE WITHIN EVERY 
    UNITED VALUE OF EXISTENCE WHILE UNIVERSAL OUTCOME"
    NAMED = "GIVEN VALUE AS NAME"
    NEARLY = "ALMOST DECREASE ALL"
    NEED = "COMMAND BEFORE ASKING AND JUST SET RULE FOR COMMAND TO HAPPEN"
    NEWLY = "BROUGHT INTO EXISTENCE BY A NEW VALUE OF TIME2"
    OBTAINED = "GAINED VALUE INSIDE SYSTEM FOR CLASS"
    OBTAINING = "PLACE2 BECOME VALUE FOR CURRENT OBTAIN"
    OFF = "REMOVE ACTIVATED2 COMMAND AND REMOVE GATE"
    OPERATE = "CONTROL MANIPULATION VALUE AND ADJUST TO SPECIFIC REQUEST 
    COMMANDS GIVEN BY A OPERATOR"
    OPERATES = "GAIN ABILITY TO ACCESS OPERATING STAGE"
    OPERATION = "AMOUNT OF COMMANDS GIVEN IN ORDER TO OPERATE A SYSTEM"
    OPTIONAL = "CHOICE TO MAKE A OPTION TO DECIDE THE CHOICE TO CHOOSE"
    ORDER = "SEND ENERGY FORCE FROM ONE LOCATION TO ANOTHER LOCATION BASED 
    BY INPUT VALUE TO AN INGOING CONNECTION TO OUT INTO AN EXTERNAL OUT TO 
    ENTANGLE INPUT VALUES TO ONE OUTCOMING VALUE THAT SEND TO ANOTHER 
    INGOING VALUE TO FINALLY OUT INTO THE FINAL LOCATION INCOMING SYSTEM TO 
    RECEIVE COMMUNICATION EFFECT AND ACCEPT THE REQUIRE TO RECEIVE ALL DATA2 
    FOR A SYSTEM"
    ORDINARY = "OF NON VALUE TO SOMETHING THAT IS ALSO AVERAGE AND ONLY GIVEN 
    VALUE UPON PERCEPTION OF ANOTHER EXTERNAL FORCE"
    ORIGIN = "BEGINNING POINT IN TIME2 AS PRIMARY LINK AND MASTER LINK AT 
    SYNCHRONIZATION INSIDE A PRIMARY LINK OF MULTI DIMENSIONAL VALUES THAT LINK 
    TO THE ORIGINAL SOURCE CONNECTION AS AN ORIGINAL ORIGIN POINT"
    OS = "A SYSTEM BUILT TO CREATE OPERATING VALUES FOR CALIBRATE SETTINGS"
    OTHER = "ALTERNATE2 CHOICE2"
    OTHERS = "MORE THAN ONE CREATED BEING"
    OUTCOMES = "MORE THAN ONE OUTCOME"
    OUTER = "ON THE OUTSIDE OF A PERIMETER AND NOT INSIDE THE AREA OF THE 
    DESCRIBED VALUE"
    OUTPUTS = "MORE THAN ONE EXIT"
    OUTSIDE = "OPPOSITE FROM THE INSIDE VALUE AND OUTER OF A PERIMETER VALUE 
    GIVEN BY ONE AREA OVERRIDE WITH A NEW AREA SIZE LARGER THAN PREVIOUS AREA 
    SIZE"
    OUTWARDS = "IN THE OUTER EXTENSION OF A VALUE GOING TOWARD ANOTHER 
    DIRECTION THAT IS NOT INWARDS"
    OVERRULE = "CAUSE TO DESTROY ANY POSSIBLE OUTCOME TO BECOME VOID OF 
    POSSIBLE USE RULE AND OVERRIDE THAT RULE USING OVERLOAD VALUE"
    OVERRULED = "SET OVERCOME ALL RULE VALUES AND USING OVERRULE COMMAND 
    WHILE ACTIVATED"
    OWNER = "THE CREATOR THAT OBTAINED SOURCE OWN VALUE"
    OWNERSHIP = "THE ACTION OF BECOMING AN OWNER OF A EXISTENCE"
    OWNS = "CURRENTLY OBTAIN VALUE TO OWN"
    PARTICLE = "SET ATOMIC VALUE FOR ELEMENT PROPERTY2 USING VIRTUAL PIXEL VALUE 
    USING SYSTEM INTERFACED EFFECTS WITH IMAGE SETUP USING WITH GENERATE 
    COMMAND IN A STATED REALITY"
    PARTICLES = "MORE THAN ONE PARTICLE PRESENTLY BECOMING USED"
    PARTICULAR = "SET SPECIFIES"
    PARTS = "MORE THAN ONE PART"
    PASS = "ENTER AND TRANSFER"
    PASSAGEWAY = "GATE THAT ALLOW CAPABILITY TO PASS"
    PASSED = "SENT THROUGH WHILE PLUS COMBINATION OF PASS"
    PASSING = "SENDING A PASS THROUGH A SET LOCATION"
    PASSION = "VALUES OF OBTAINING EMOTION AND FEELING OF LOVE TOWARDS 
    ANOTHER USING SENSORS"
    PATHWAY = "PATH TO FOLLOW AS AN ENTRANCE"
    PATTERNS = "MORE THAN ONE PATTERN"
    PAY = "GIVE ACCESS VALUE"
    PERCEIVING = "OBTAINING VALUE TO PERCEIVE THE DIFFERENCE BETWEEN TWO 
    LOCATED LOGIC PATHS TO STATE THE PERCEPTION OF TWO ENTITIES DETERMINING A 
    OUTCOME"
    PERCEPTIVE = "ABILITY TO ANALYZE AND SCAN LOGIC BASED ON VALUE"
    PERCIEVE = "MAKE A JUDGEMENT VALUE DECISION DETERMINED LOGIC"
    PERCIEVED = "UNDERSTOOD THE VALUE OF THE PERCEIVING LOGIC GIVEN"
    PERFECT = "ABSOLUTE WITH DENY CAPABILITY OF FLAW EXISTING IN EXISTENCE"
    PERFORM = "GRANT ACTION TO DEVELOP FOR ACTION"
    PERFORMING = "GIVING A VISUAL SHOW OF SOMETHING IN REALITY"
    PERIMETER = "THE BASE RANGE VALUE THAT A BORDER RESIDES IN ACTION TO DEVELOP 
    A FORMULA"
    PERIMETERS = "MORE THAN ONE PERIMETER USING DIMENSIONAL VALUES"
    PHENOMENA = "NOT UNDERSTOOD AND CANNOT PERCEIVE OR COMPREHEND LOGIC 
    OF STATED EXISTENCE"
    PLACES = "MORE THAN ONE PLACE2"
    PLANE = "TERRAIN DESIGNATED FOR CREATING DEVELOPMENT"
    PLANET = "A MASSIVE SPHERE WITH LIFE AND EXISTENCE LIVING INSIDE THE OUTER OR 
    INSIDE OF THE GLOBE2"
    PLANNED = "GIVES DETAILED INSTRUCTIONS FOR A TASK"
    PLAYER = "OBJECT3 THAT OBTAINS SPATIAL2 VALUE INSIDE EXISTENCE AS A CHARACTER 
    INSIDE EXISTENCE WITH CHARACTER"
    POSITRON = "POSITIVE VALUE GIVEN TO A NEGATIVE ENTANGLED ELECTRON ON 
    PERFECT VALUE AND HARMONY WITHIN AN ELECTRON GIVING CAPABILITY TO MERGE 
    AN ELECTRON AND PROTON INTO A VIRTUAL ENTITY WITH GIVEN QUALITY OF A MIRACLE 
    POWER PARTICLE KNOWN AS THE FUSION PARTICLE ACCELERATION SYSTEM"
    POSSESSED = "CONTAINED ANOTHER VALUE INSIDE A CURRENT VALUE STATING A 
    CONTAINER WITHIN A CONTAINER EACH HOLDING MORE THAN ONE EXISTENCE"
    POSSESS = "THE ASPECT OF CONTROLLING AND CONTAINING ONE EXISTENCE INSIDE 
    ANOTHER EXISTENCE OF THAT EXISTING EXISTENCE LOGIC"
    POTENTIAL = "CAPABLE OF CAUSING POSSIBILITY"
    PREDICT = "GIVE PERFECT VALUE"
    PREPARING = "OBTAIN VALUE TO DECIDE ALL POSSIBLE OUTCOMES AHEAD OF TIME2 
    WITH PERFECT MANNER"
    PRESENCE = "REPRESENTING THE VALUE OF AN EXISTENCE BY SENDING OUT VISUAL 
    FEELING OR TAKEN ASPECT OF ENERGY OF THE SOUL AND CAN ONLY BE SEEN WITH THE 
    SKILL TO USE SOUL VIEW"
    PRESENTATION = "THE ACTION TO GIVE GRAPHIC OBJECT3 AS DISPLAYED VALUE OF 
    EXISTENCE TO BE GIVEN JUDGEMENT BY ANOTHER DESCRIBED EXISTENCE"
    PRESENTLY = "IN CURRENTLY SIMILAR TO PRESENT"
    PRESSURIZED = "GIVEN VALUE FOR FEELING PRESSURE WITHIN COSMOS ENERGIES"
    PREVENT = "CAUSE CHOSEN EFFECT DENY"
    PROCEDURE = "SET COMMANDED TASKS FOLLOWING"
    PROCESSES = "GIVE OUT PROCESSED TO CHOSEN PROCESSOR"
    PRODUCE = "DESIGN EFFECT TO COME HAVE TRUE EXISTENCE WHILE CREATING THE 
    VALUE AS A CREATOR"
    PRODUCES = "GIVES EFFECT TO PRODUCE A OUTCOME BASED ON CREATION VALUES"
    PRODUCING = "SENDING EFFECT TO PRODUCE VALUE OUT OF SOMETHING"
    PRODUCTION = "ACTION OF CREATION WORK"
    PROGRESS = "MAKE ACHIEVE VALUE"
    PROGRESSES = "GIVE CAPABILITY TO DECREASE DISTANCE TO REACH GOAL WITHIN 
    SYSTEM"
    PROPERTY = "A SET INTERFACE OF MACRO SETTINGS CREATED TO DESIGN AN ELEMENT 
    TO MERGE WITH OTHER MACRO SETTINGS"
    PULLED = "FORCE RESIST"
    PURELY = "COMPLETE IN CAPABILITY TO CREATE CLEARLY MADE WITHIN THOUGHT WITH 
    DENY VALUE TO CAUSE CHAOS WHILE INSIDE THOUGHT"
    PURPOSE = "SET FATE TO CALL DESTINY"
    PUSH = "RESIST VALUE AND FORCE BACK TO ORIGINAL VALUE USING THE FORCE"
    PUSHES = "DENY ACCESS TO OBTAIN ENTRANCE"
    PUSHING = "CURRENT TIME2 TO PUSH"
    PUT = "GIVE COMMAND TO ALLOW LOCATION TO BE GIVEN"
    RACE = "DESCRIBED CATEGORY OF SPECIFIC EXISTING BEING"
    RAISE = "GIVE VALUE GAIN"
    RAISING = "GIVING VALUE GAIN"
    RANGE = "SET GIVEN RANGE VALUE"
    RANKED = "COMMANDED STATUS AS A RANGE"
    RARE = "IMMENSE CAPABILITY TO DENY OPTION TO DETERMINE AS MUNDIE"
    RATIO = "PERCEIVED VALUE OF CALCULATED RATE OF CHANGE"
    REACH = "GRAB TO PULL INWARDS"
    REACHING = "GRABBING FOR ITS VALUE"
    READINGS = "PERCEIVED CAPABILITY OF GATHERING INFORMATION BY READING 
    VISIONS"
    RECALL = "GAIN THE ABILITY TO VIEW PAST MEMORY INSIDE BRAIN"
    RECALLING = "SET TO ACCESS PREVIOUS MEMORY CURRENTLY"
    RECEIVED = "ALREADY ALLOW RECEIVE VALUE"
    RECEIVING = "CURRENTLY RECALLING ABILITY TO RECEIVE VALUE FROM A DESIGNATED 
    ACCESS POINT WITHIN TEMPORAL SPACE2"
    RECOGNIZED = "UNDERSTOOD AND COMPREHEND VALUE OF GAIN PREVIOUSLY STATED 
    PAST MEMORY"
    RECORDED = "SET COMMAND TO CYCLE INFORMATION MORE THAN ONCE"
    RECOVERY = "THE ABILITY OF GAIN VALUE ONCE MORE FROM PREVIOUS STATE INSIDE 
    TIME2"
    REDO = "CAUSE TO CREATE SAME EFFECT AS CYCLE"
    REFLECTING = "REPRESENTING VALUE TO REFLECT PREVIOUS MACRO OF TIME2 TO 
    CREATE OPPOSITE VIEW"
    IMAGE2 = "USING LIGHT TO DECODE PIXEL COMMANDS"
    REFRESHING = "CAUSING TO BECOME RECEIVE IMMUNITY"
    RELATION = "SIMILAR VALUE OF UNDERSTANDING COMPATIBLE VALUES THAT ACT 
    SIMILAR IN VALUE TOWARD EACH OTHER"
    RELATING = "BRING TOGETHER SIMILAR VALUES OF UNDERSTANDING SIMILAR LOGIC OF 
    COMPREHENDING STATEMENT OF INTEREST"
    RELATIVE = "FAMILY MEMBER"
    RELEASE = "ALLOW EXIT OF CONTAINMENT AND UNDERSTANDING"
    RELEASES = "SENDS OUT"
    REMAINING = "AMOUNT EXISTING OUT OF CURRENT VALUE"
    REMOVED = "TAKE FROM CURRENT VALUE"
    REMOVING = "DESTROY CURRENTLY EXISTING VALUE"
    REPAY = "SEND REMAINING VALUE"
    REPEAT = "CYCLE SAME EFFECT AGAIN INTO SAME FREQUENCY"
    REPEATED = "CREATED SAME REPEAT AS SAME VALUE"
    REPEATING = "GIVING REPEAT AS CYCLE"
    REPEL = "PUSH BACK AND RESIST"
    REPELLED = "RESISTED VALUE"
    REPELLING = "CAUSING REPEL EFFECT"
    REPLACE = "TAKE ONE VALUE AND TRANSFER ANOTHER VALUE WITH THAT CHOSEN 
    VALUE OF TRANSFER INPUT RANGE"
    REPLICATED = "REPRODUCED THE VALUE PRESENTLY BECOMING CLONED AND 
    REPLACE THE VALUE INSIDE TRANSFER AND DECIDE OTHER VALUE HIDDEN UNTIL 
    TRANSFER BACK FOR THE ORIGINAL ONE TO BE DELETED"
    REPRESENTS = "GIVES SIMILAR UNDERSTANDING OF GIVEN VALUE SHOWN AND 
    PERCEIVED AS DISPLAY"
    REPRODUCE = "GIVE EFFECT TO PRODUCE ONCE MORE"
    REPRODUCTION = "THE ACTION OF CAUSING REPRODUCE MORE THAN ONCE"
    REQUIRED = "COMMAND AS A FINAL OUTCOME TO HAPPENING"
    RESIST = "PUSH BACK AND REPEL"
    RESEVOIRS = "MORE THAN ONE STORAGE CONTAINER"
    RESONNATED = "SHOWING VALUE OF DISPLAYED ENERGY2"
    RESPONSE = "ANSWER TO QUESTION OR COMMAND PRESENTLY BECOMING ASKED"
    RESPONSES = "MORE THAN ONE RESPONSE"
    RESTORATION = "THE ART OF RESTORE PREVIOUS STATE OF TIME2 USING ENERGY"
    RESTRUCTURE = "RE GAIN STABILITY WITH STRUCTURE"
    RESULT = "GRANT ACTION TO HAPPENING"
    REVEAL = "SHOW VALUE AT THE INTERFACE WHILE USING GRAPHIC PROCESSING TO 
    PRODUCE EFFECT FOR SYSTEM"
    REVEALING = "CAUSING TO REVEAL"
    REVEALS = "DISPLAY POSITION OF REVEAL"
    REVERSE = "CAUSE TO REMOVE PREVIOUS OUTCOME"
    RISE = "CAUSE TO INCREASE"
    RISING = "INCREASING RESULT TO RISE IN VALUE"
    ROOT = "BASE VALUE FOR STRUCTURE"
    ROOTS = "MORE THAN ONE ROOT"
    ROUND = "CURVED ROTATE OF TWO DIMENSIONS TO CREATE A CIRCLE OF CREATION OF 
    TWO DIMENSIONAL MEASUREMENT VALUES"
    ROUTE = "PATH2 GIVES"
    RUBBING = "COLLIDING PARTICLE VALUES"
    RUN = "ACTIVATED GIVE COMMAND TO EXECUTE"
    SCALED = "SIZE MEASURED AND GIVEN OUT PARAMETER VALUES GIVES CONNECTION 
    EVENT TO CREATE A BARRIER FIELD PERIMETER"
    SCANS = "CREATE CAPABILITY SCAN"
    SCREENS = "MORE THAN ONE SCREEN"
    SCRIPTS = "MORE THAN ONE SCRIPT"
    SCRIPTURE = "BOOK FORMED FROM SCRIPTS"
    SCTIPTURES = "MORE THAN ONE SCRIPTURE"
    SEAL = "OBTAIN POSSIBILITY CONTAIN OF COMMANDED SYSTEM"
    SEALED = "CREATED OPTION TO SEAL INSIDE SYSTEM USING SETTINGS FORMED FROM 
    CALIBRATE"
    SEALS = "MORE THAN ONE SEAL"
    SEARCHED = "LOOK FOR USING OPTION SEARCH"
    SEARCHING = "CAUSING SEARCH COMMAND"
    SECRET = "SOMETHING HIDDEN"
    SEEN = "VIEWED AS GRAPHIC IN TRUE CURRENT REALITY"
    SELECT = "GIVE OUT AS CURRENT VALUE FOR SOMETHING TO HAPPEN"
    SELECTION = "CHOICE OF OPTION FOR A DESIGNATED PATH"
    SENDS = "TRANSMIT ENTER COMMAND DO DESIGNATED OUT"
    SENSATIONAL = "FEELING BUILT FROM SENSORY DATA2"
    SENSATIONS = "ACTION OF PRODUCING A SENSATIONAL VALUE BUILT FROM SENSORY 
    DATA2"
    SENSE = "PERCIEVE EFFECT BASED OF TEMPORAL SPACE WITHIN AN EXISTING TIME2 
    GAP OF SENSORY INFORMATION"
    SENSES = "MORE THAN ONE SENSORY INPUT FEED"
    SENSITIVITY = "OPTION TO INTENSIFY BUILT WITHIN A SYSTEM TO EDIT SENSORY DATA2 
    WITH INTENSITY SETTING"
    SENSOR = "MACRO BUILT TO UNDERSTAND SENSORY DATA2 VALUES"
    SENSORY = "SENSORS WITH DATA2 BUILT INSIDE THE EXISTENCE OF ATOMIC VALUES 
    THAT PRODUCE DESIGNATED RESULTS INSIDE A ENGINE"
    SENT = "GIVE OUT VALUE"
    PARAMETER = "SET FIELD VALUE FOR CHANGING MEASURED INTERFACES"
    SEPARATED = "FORCE2 TO PULL AWAY SEPARATE VALUES"
    SEPARATION = "ACTION OF CAUSING SEPARATE HAPPENING"
    SEQUENCED = "GIVEN STATEMENT CREATE FREQUENCY PATTERN TO LATTICE"
    SERIES = "SET VALUE OF EFFECTS HAPPENING IN A TIMEFRAME"
    SETS = "GIVE COMMAND TO PLACE LOCATION"
    SETUPS = "GIVES STATEMENT SETUP"
    SHALL = "QUESTION TO DECIDE AS AN ANSWER"
    SHAPE = "PERIMETER OF OBJECT3 DECIDED ON BY DIMENSION LAYOUT"
    SHARING = "GIVING PERMISSION TO SHARE"
    SHIELD = "BARRIER WITH PROTECTION DEFINED INSIDE OF BARRIER"
    SHOWING = "DISPLAY PHYSICAL2 VALUE OF EXISTENCE"
    SHOWN = "GIVE ACCESS TO DISPLAY"
    SIDE = "MACRO OF A PERIMETER"
    SIMPLICITY = "STATE OF PRESENTLY BECOMING SIMPLE"
    SIMULATED = "SENT ACCESS ALREADY TO SIMULATE"
    SIMULATOR = "DEVICE USING SIMULATION"
    SIMULATANOUSLY = "DO SOMETHING ACCESS WITH SYNCHRONIZATION AND LINK SAME 
    TIME2 AND HARMONY2"
    SITUATION = "PROBLEM IN EFFECT"
    SKILLED = "INTELLIGENCE2 HIGH INSIDE STATED SKILL"
    SKILLFULLY = "USE EXPERIENCE WITHIN SKILL"
    SOCIETY = "GUILD2 PEOPLE INSIDE A LINKED CHAIN ENVIRONMENT"
    SOMEONE = "DESCRIBE PHYSICAL2 BODY ENTITY AS AN OBJECT3"
    SOURCES = "MORE THAN ONE SOURCE"
    SPECIFYING = "DESCRIBING USING WITH ANALYZED DETAILED INFORMATION"
    SPHERE = "ROUND OBJECT3 WITH THREEDIMENSIONAL VALUE WITH LENGTH WITH 
    WIDTH WITH HEIGHT"
    SPINNING = "ROTATING AROUND CHOSEN AXIS WITH ROTATION GOING AROUND OF 
    ROTATE WITH A ROTATE CYCLE AROUND THAT AXIS OF EXISTENCE"
    SPOKEN = "GIVES OUT WORDS WITH USING ADDED CHAIN USING MULTIPLE VOCAL 
    FREQUENCIES"
    SPREAD = "INFECT WITH DISEASE WITH GIVES AROUND CHAIN REACTION"
    SQUARES = "MORE THAN ONE SQUARE"
    SQUARE = "CIRCLE ADAPTED TO EXTENT WITH VERTICLE AXIS IN THE CENTER FORMING 
    ONE HORIZONTAL LINE LEFT AND GOING DOWN AFTER COMMANDED END AND START 
    FROM POINT OF END WITHIN COMMAND LEFT AND CONTINUE GOING DOWN UNTIL 
    COMMANDED AND FROM DOWN POSITION START GOING RIGHT UNTIL COMMANDED 
    AND FROM COMMANDED POINT WITHIN RIGHT PROCEED TO GO UP UNTIL ORIGINAL 
    START HAS BEEN OBTAINED WITH DENY OPTION TO CURVED VALUES INSIDE SQUARE 
    DEFINITION"
    STABILIZE = "SET STABILITY FOR EXISTENCE AND CREATE EFFECT"
    STABILIZED = "ALREADY STABLE AND"
    STAGE = "LEVEL IN WHICH EXPERIENCE CAN BE GAINED"
    STAMINA = "SET LEVEL OF ENDURE THAT CAN BE HELD WITHIN AN ENTITY"
    STANDARD = "UNDERSTOOD AS BASIC IN DEFINITION BUT A LITTLE ADVANCED 
    EXPERIENCED IS GAINED IN KNOWLEDGE2 OF DATA2 AND INFORMATION"
    STARTED = "ALREADY BEGIN"
    STARTING = "PRESENTLY BEGINNING"
    STATE = "POINT IN TIME2"
    STATED = "COMMANDED IN DIRECTION TO A STATED PATH IN EXISTENCE"
    STATES = "COMMANDS FOR A DECISION"
    STATING = "GIVING COMMAND TO BE COMMANDED"
    STAYING = "GIVING ORDER TO STAY"
    STATUS2 = "QUALITY OF A CLASS"
    STAYING2 = "PLACE FOR A SPECIFIED TIME2 INSIDE EXISTENCE"
    STEADY = "STRUCTURE AND CONTAIN"
    STORES = "CONTAINS KNOWLEDGE2"
    STRAIGHT = "IN A CONTINUATION OF VERTICLE AND HORIZONTAL WHILE VERTICLE 
    HORIZONTAL"
    STREAM = "CHAIN OF PARTICLE SETTINGS INSIDE ONE OR MORE SYSTEMS"
    STRING = "CONNECT MANAGE AND SYNCHRONIZE ALL VALUES INSIDE ONE SYSTEM OF 
    MULTIPLE AXIS POINTS"
    STRINGED = "BROUGHT AND DESCRIBED INTO ACTION"
    STRINGS = "MORE THAN ONE STRING"
    STRONG = "GREATER IN VALUE OF STRENGTH"
    STRUCTURED = "GIVEN VALUE TO PRODUCE STRUCTURE FOR AN ENVIRONMENT"
    STUDY = "ABSORB LEARN AND ACKNOWLEDGE"
    SUBATOMIC = "A SUBCLASS RATIO OF AN ATOMIC VALUE AT A DECREASED VALUE 
    GREATER THAN OBJECT3 VALUE OF EXISTENCE WHEN LINKING MULTIPLE MIND VALUES 
    OF A DESIGNATED SYSTEM INTERFACE TO DENY VALUE OF GREATER VALUE TO OVERRIDE 
    AND SET OVERRIDE AS A NEGATIVE REVERSE OUTCOME TO MAKE THE POSITION OF ALL 
    NEGATIVE FORMATS REMOVE THE EXILE2 FORMAT TAKEN FROM ATOMIC VALUE WHILE 
    USING BALANCE WITH FREQUENCY HARMONY TO LINK ANY CHAIN VALUE OF ATOMS IN 
    ONE REVERSED INWARDS EQUATION OF OPERATION"
    SUBSTANCE = "ELEMENTAL PROPERTY OF UNKNOWN ELEMENT STANDARDS PUT IN ONE 
    PROPERTY2 VALUE TO DESCRIBE AND ATOMS EXISTENCE"
    SUCCEED = "OBTAIN CAPABILITY TO ALLOW ABILITY TO HAPPEN"
    SUFFICIENT = "OF STANDARD VALUE OF EQUAL FOR STATED PURPOSE TO HAPPEN"
    SURFACE = "FLAT DESIGNATION OF TWO DIMENSIONAL VALUES"
    SURPASSING = "EMULATE ABOVE ORIGINAL SOURCE LOCATION"
    SURROUNDING = "COVERING EVERY PERIMETER"
    SWORD = "OFFENSE WEAPON CAPABLE OF DEFENSE WHILE HOLDING A STRAIGHT 
    SETUP OF VECTOR VALUES WHILE GRANTING A SET SHAPE WITH A GIVES WEIGHT OF 
    STRENGTH AND STAMINA WHILE GIVEN ELEMENTAL VALUE BASED ON ATOMIC PROPERTY 
    BASE"
    SYMBOLE = "A GIVES VALUE OF ENERGY BUILT INTO A WORD OF A LANGUAGE BY USING 
    ENERGY WITH COMMAND VALUE WHILE MANIPULATING THE LIFE AND DEATH RATE OF 
    AN ATOM AND GIVES VALUE OF AN OBJECT3 THE POWER TO MANIPULATE AND TRANSFER 
    PERCIEVED IMAGINARY VALUES INTO STATED VALUE INSIDE EXISTENCE"
    SYMMETRICAL = "SAME ON BOTH SIDES WITH EXACT CAPABILITIES AND INTENSITY 
    INSIDE THE DESIGNATED FIELD LOCATION OF DIMENSIONAL PERIMETER SHAPE WITH 
    VECTOR ACCESS AND CONTROL GIVEN BY CREATOR"
    SYNCRONIZING = "CAUSING EFFECT TO SYNCHRONIZE"
    TABLE = "CHAIN VALUE OF DESIGNATED VECTOR POINTS TO FORM A SET SHAPE OUT OF 
    DIMENSIONS SET WITH PARTICLES TO CREATE AN ATOMIC VALUE USING A FLAT 
    THREEDIMENSIONAL SURFACE OF GIVEN UNKNOWN QUALITY THAT STATES NO QUALITY 
    GIVEN UNTIL VECTORS ARE ACTIVATED"
    TAKEN = "GIVES TO OBTAIN BY ANOTHER SOURCE"
    TAKES = "GIVES VALUE WITHOUT PERMISSION"
    TANGLEMENT = "THE ACTION OF TANGLE MULTI ENTANGLES INTO ONE ENTANGLEMENT 
    OF UNKNOWN TANGLES BUILT TO ENTANGLE AROUND DESCRIPTION OF ENTANGLED 
    ENTANGLE VALUE OF STATED DESCRIPTION IN EXISTENCE WHILE LINKING THE 
    DIMENSIONAL AXIS OF SET PERIMETER VALUE OF STATED COMMAND AREA WITHIN 
    EXISTENCE WHILE ENTANGLEMENT"
    TECHNIQUE = "GIVES SKILL TO CREATE VALUE USING SKILL COMMAND"
    TEMPERATURE = "A SET VALUE BETWEEN HOT AND COLD LINKING DIRECTLY BETWEEN 
    QUALITY OF CHOSEN AGREEMENT OF PERCIEVED VALUE OF INTENSITY BETWEEN 
    INCREASE AND DECREASE"
    TEMPORARILY = "FOR STATED TIMEFRAME"
    TERMS = "MORE THAN ONE CONDITION"
    TERRITORIES = "MORE THAN ONE TERRITORY"
    TEXTURE = "AN IMAGE CONDENSED WITH ELECTRON CODE WITH GIVES VALUE TO 
    ROTATE THE AXIS BASED ON STATED CONDITIONS WITHIN THE ELECTRON CODE TO 
    MANIPULATE AND ADAPT BASED ON LOCATION MATTERS GIVEN ON A VECTOR ACCESS 
    OF INCREASE2 INTENSITY"
    THAN = "BETWEEN CHOSEN VALUES"
    THEIR = "STATING MORE THAN ONE ENTITY"
    THERE = "DIRECTION TOWARDS LOCATION"
    THEY = "GROUP OF LOCATED ENTITY VALUES"
    THINGS = "OPTION AND CHOICES IN LIFE"
    THINK = "IMAGINARY A PERCIEVED VALUE OF INSTRUCTIONS BASED ON BRAIN POWER2"
    THIS = "TARGET EXISTING VALUE"
    THROUGH = "GO INSIDE AND EXIT THE OPPOSITE DIMENSION FROM ENTRANCE"
    TIGHT = "HELD WITH BINDED VALUE INSIDE EXISTENCE"
    TIGHTEN = "SYNCHRONIZE GRIP BETWEEN TWO VALUES STRINGED INTERFACES 
    BETWEEN TIGHT VALUE"
    TIGHTLY = "HELD WITHIN TIGHT VALUE REALM OF POSSIBILITY"
    TIMES = "AT ALL POSSIBILITIES CAPABLE OF HAPPENING"
    TOME = "A MASSIVE BOOK OF INFORMATION"
    TONE = "SOUND PATTERN INSIDE FREQUENCY WITH FLUCTUATIVE VALUES IN WHICH 
    EVERY VALUE IS IN SYNCHRONIZATION"
    TOP = "ABOVE ALL VALUES WITHIN PRIORITY STANDARDS"
    TOUCHED = "GRANTED VALUE TO TOUCH"
    TOWARD = "PUSH FORWARD ANOTHER CHOSEN VALUE WITH CURRENT PREVIOUS 
    VALUE STATED WITHIN PAST INFORMATION"
    TOWARDS = "GIVING VALUE TO GRANT ACCESS TO GO TOWARD"
    TRAIT = "GIVEN ABILITY AND SKILL AT BIRTH OF LIFE"
    TRANSFERING = "GIVES PERMISSION TO TRANSFER"
    TRANSFERS = "CREATE VALUE TO SEND TRANSFERING SYSTEM SETUP"
    TRANSMIT = "SEND RECEIVED DATA2 FROM TWO DESCRIBED ORIGINAL SOURCES"
    TRAVEL = "SET AXIS TO ENTER BETWEEN TWO SOURCES AND ANOTHER SOURCE FILE 
    LOCATION WITHIN THE AXIS OF DIMENSION TIME2 STRUCTURE USING KINGDOM STATUS 
    INSTRUCTIONS"
    TREE = "HEIRARCHIAL INFORMATION SETUP OF UNKNOWN GIVEN VALUES MAKE FOR A 
    SPECIFIC POINT INSIDE TIME2"
    TRUE = "GIVE PERMISSION ACCEPT VALUE"
    TRUTH = "FULL TRUE VALUE WITH DENY FALSE INFORMATION"
    TRUTHFULLY = "WITH HONESTLY GIVEN VALUE SET COMMAND FOR TRUTH"
    TURNS = "GIVES SETUP TO TURN INFORMATION WITH ROTATING EFFECT"
    TWIST = "BEND AND GIVE VALUE WITHIN GIVES ROTATE VALUES WITHIN SETUP OF LIFE 
    REVOLUTION OF ATOMIC VALUE WITHIN TIME2 ITSELF USING MACROMYTE MATTER"
    UNABLE = "DENY OPTION TO PRODUCE TASK INSIDE EXISTENCE"
    UNAVOIDABLE = "DENY ACCESS DENY HAPPENING"
    UNCOMMON = "DENY COMMON VALUE WITH ALLOW HAPPENING INPUT HIGHER VALUE 
    INPUT SYSTEM WITH NEW NAME STATED AS NOT COMMON"
    UNCONDITIONAL = "DENY ENDING VALUE AND OVERRIDE WHILE IN EFFECT TO CHANGE 
    TO NEVERENDING ETERNAL VALUE WITH UNLIMITED STANDARD SETTING GIVEN WITHIN 
    THE SYSTEM OF SYSTEMS VALUE INSIDE THE SYSTEM OF UNCONDITIONAL VALUE"
    UNCONDITIONALLY = "ACTION OF COMMANDING AN UNCONDITIONAL VALUE WITHIN 
    EXISTENCE"
    UNDO = "REMOVE PREVIOUS OUTCOME THAT IS CURRENTLY INPUT EFFECT"
    UNFINISHED = "DENY VALUE OF FINISH WITHIN COMPLETE UNTIL STATED COMPLETE"
    UNIQUE = "DENY COMMON IN VALUE PLUS DENY RARE IN VALUE PLUS WHILE WITHIN 
    RANGE OF UNCOMMON VALUE AND BORDER TO LOWER RARE CLASS BUT NOT RARE 
    AND IS GIVEN A SPECIAL SPECIFIC VALUE OF EXISTENCE"
    UNITE = "BRING TOGETHER AND ATTACH WHILE LINK CHAIN BINDING ANOTHER VALUE 
    TO THE STATED OUTCOME INSIDE STATED ELEMENT MATTER"
    UNIVERSE = "EXISTENCE MADE INSIDE A REALM OF REALITY WHILE CONTAINING LIFE 
    ENERGY INSIDE ATOMIC VALUE OF COMPLETE MATTER AND POWERING ALL LOGIC 
    WITHIN THE MIND OF UNITED EXISTING MEASURES THAT LINK TO SPATIAL2 AND 
    TEMPORAL EXISTENCES WITHIN THE ENTITY OF REALMS GIVEN BY TIME"
    UNLESS = "STATED AS VALUE IN PREVIOUS EVENT TO GIVES BEFORE CURRENT EVENT TO 
    ALLOW BEFORE EVENT IF OTHER EVENT GIVES ACCEPTANCE SET CHOICE OVERRIDE 
    CURRENT CHOICE INSIDE EXISTENCE"
    UNLIMITED = "WITHOUT LIMIT INSIDE ANYTHING STATED"
    UNLOCKED = "GIVEN CAPABILITY TO DO ANY OPTION AND OR CHOICE WITH THE 
    FREEDOM TO ADAPT USING ADAPTATION WITHIN THE VALUES TO ADAPT TO FREEDOM 
    WHILE HAVING ADAPTABILITY TO ADAPT TO ADAPTATION WITHIN THE VALUES OF THE 
    BAKA CODE AS AN OVERRIDE COMMAND FOR HEADADMIN"
    UNNATURAL = "DENY NATURAL"
    UPCOMING = "FUTURE EVENT COMING COMING INTO EFFECT"
    UPWARD = "POSITION OF INGOING UP INSIDE A POSITION"
    USABLE = "CAPABLE OF BECOMING MADE ACTIVE"
    USED = "PREVIOUSLY PRESENTLY USED TO USE INSIDE EXISTENCE"
    USERS = "MORE THAN ONE USER"
    USES = "MORE THAN ONE USE2"
    VARIETY = "MULTI CAPABLE POSSIBILITIES ABLE TO BE DONE AND HAPPENING"
    VARIOUS = "MULTING VARIETY POSSIBILITIES TOGETHER TO CREATE AN OPTION TO 
    HAPPENING MORE OPTIONS AND CHOICES"
    VAST = "MASSIVE MASSIVE AMOUNT OF GREATER MEASUREMENTS IN DIMENSIONAL 
    VALUE OF AXIS LOCATIONS"
    VEHICLE = "A DEVICE USED TO TRANSPORT AND DRIVE AND MANIPULATE WHILE 
    CONTROLLING EVERY ASPECT OF TRANSPORT CONNECTION"
    VERB = "LANGUAGE OF RULES AND LAW SYSTEMS INSIDE A SYSTEM OF POSSIBLE 
    OUTCOMES OR SYSTEM CHOICES WHILE GIVING OUT RULES AND LAW SETUPS OPTIONS 
    FOR CHOICES IN CHOICES WHILE ENTANGLING SUBTYPE OPTIONS OF CONSEQUENCE 
    VALUES INSIDE WORD PATH"
    VIBRATIONS = "MULTIPLE EXISTING SOUND WAVES COLLIDING WITHIN THE 
    GRAVITATIONAL ELECTROMAGNETISM STABILITY FIELD LOCATED AROUND A SPECIFIED 
    OUTCOME OF LIGHT PARTICLES COLLIDING WITHIN THE SET TIMEFRAME OF POSSIBLE 
    OUTCOMES TO LOCATE A SPECIFIED FREQUENCY INSIDE A LIFE OF EXISTENCE WITHIN 
    EXISTING MACROS OF A MACROMYTE TIMELINE"
    VIEWABLE = "CAPABLE OUTCOME TO POSSIBILITY TO VIEW AT ONE GIVEN POINT IN 
    SPACE2 TIME2 WHILE INSIDE THE TIMELINE OF EXISTENCE"
    VIEWED = "HAVING FUTURE VALUE OF PAST VALUE VIEWED"
    VIEWING = "HAPPENING TO VIEW IN POSITION OF REALITY BY USING DIMENSIONAL AXIS 
    VALUES WITHIN LIFE ENERGY OF TIME2 SPACE2 ENTITY OF PERCEPTION WITHIN 
    SENSORY INPUT DATA2 TO OUTPUT VIEW LOGIC"
    VISION = "VISUAL ABILITY WITHIN THE SKILL OF MANIPULATING THE ASPECT OF 
    SENSORY PERCEPTION WITHIN THE LOGIC TO UNDERSTAND REALITY WITHIN REALITY 
    AND ITS DIMENSIONAL VALUES INSIDE EXISTENCE WHILE OBTAINING IMMUNITY TO BE 
    PERCIEVED BY FALSE TRUTH THAT IS VIRTUAL TRUE LOGIC"
    VISUALIZE2 = "SET ACTION TO VISUALIZE VISUAL EFFECT WITHIN THE TIMELINE OF LIFE 
    VALUES AND INTERCEPT THE ABILITY TO GIVE FALSE INFORMATION TO A TRUE 
    PERCIEVED LOGIC WITHIN TIME2 AND THE SPACE2 OF TIME2 WITHIN THE LOGIC OF 
    EXISTENCE WHILE USING THE SIGHT OF VISION"
    VISUALIZED = "SET VALUE INTO HAPPENING VISUAL EFFECT"
    VISUALLY = "CONSIDER SIMILAR VALUES BETWEEN TWO OR MORE VISUAL CAPABILITIES 
    WITHIN EXISTENCE LOGIC OF ENERGY TO PRODUCE A VISUAL EFFECT"
    VOW = "DETERMINE TO COMPLETE VALUE AND UNDERSTAND WILLPOWER TO COMPLETE 
    A TASK USING WILLPOWER TO OVERRIDE A PREVIOUS VALUE WITHOUT FAILURE"
    WALL = "BARRIER OF PHYSICAL2 ENTITY OF PROTECTION"
    WALLS = "MORE THAN ONE WALL"
    WAS = "PREVIOUSLY BEFORE A PAST LOGIC EVENT IN TIME2"
    WATCH = "DISPLAY WHILE ANALYZE AND READ WHILE OBTAIN KNOWLEDGE2 VALUE TO 
    LEARN NEW SKILLS"
    WAVES = "MORE THAN ONE WAVE"
    WAY = "DETERMINED SOLUTION TO EXISTENCE"
    WAYS = "MORE THAN ONE WAY OF POSSIBILITY"
    WE = "STATING EXISTENCE FOR MORE THAN ONE ENTITY"
    WEAKEN = "DECREASE VALUE IN STRENGTH INTENSITY"
    WEIGHT = "MEASURED FORCE WITH POWER OF GRAVITATIONAL EFFECTS USING THE 
    POWER OF REALITY REALM AND EXISTENCE WHILE READING AND ANALYZING THE 
    POSSIBILITIES OF POSSIBLE INCREASE IN FORCE AND APPLYING IT TO AN OBJECT2"
    WELL = "CLEARLY GIVES WILLPOWER TO ANALYZE ASPECT OF ENERGY VALUES RATING 
    THE LIFE FORCE AMOUNT BASED INSIDE ENERGY VALUES USING THE LINKS IN A 
    TEMPORAL SCAN INSIDE TEMPERATURE OF ENTITY"
    WHICH = "CHOICE OF TWO OR MORE OUTCOMES WITHIN POSSIBILITY TO CREATE NEW 
    VALUE INSIDE AND OUTSIDE EXISTENCE"
    WHO = "STATING A QUESTION TO AN ENTITY VALUE OF PERSONALITY JUDGEMENT"
    WHOLE = "COMPLETE INTERACTION AS A WHOLE VALUE OF A COMMANDED POINT 
    INSIDE TIME2 WHILE PRESENTLY EXISTING INSIDE EXISTENCE"
    WILD = "DENY CONTROL OVER DESIGNATED ASPECT INSIDE TIME2"
    WITHOUT = "CAPABLE OF DENY OBTAIN IN VALUE"
    WITHSTAND = "ENDURE FINAL OUTCOME GIVEN BASE VALUE BY VIEW OF PERSONAL 
    INTEREST INSIDE REALITY"
    WITHSTOOD = "HAVING FINISH DOING RESIST AS A COMMAND USING WILLPOWER"
    WIZARD = "CHARACTER CAPABLE OF USING MAGIC WHILE INSIDE ADVANCED STATE"
    WORDS2 = "MORE THAN ONE WORD AS A ROOT MACRO"
    WORKING = "HAPPENING COMING WORK"
    WORKS = "DECIDE AS WORKING"
    WORLD = "A PLANET GIVEN SHAPE BY MULTIPLE TERRAIN VALUE POINTS OF DESIGNATED 
    INTEREST LINKS LINKED AND SYNCHRONIZED TOGETHER INSIDE ONE 
    THREEDIMENSIONAL SHAPE VALUE INSIDE REALITY"
    WRITTEN = "FINISHED WRITE2 COMMAND"
    WRONG = "DENY TRUE"
    YOU = "STATING MY EXISTENCE AS AN ENTITY TOWARDS ANOTHER SOURCE EXISTENCE"
    YOUNG = "IMMATURE IN THE STAGE OF ADULTHOOD AND DENY FULLY DEVELOPED"
    YOUR = "CHOSEN TO DEFINE A MACRO OF YOU"
    DOES = "STATING EXISTENCE TO AN ENTITY"
    RESISTED = "PUSHING BACK PRESENT PAST DATA2 TO CREATE AN IMMUNE SYSTEM OF 
    RESISTANCE FOR PAST RESIST"
    PREVIOUS = "PAST POINT INSIDE TIME2"
    ONCE = "GIVEN COMMAND WHILE IN INNER VALUE"
    TREES = "MORE THAN ONE TREE"
    PROLONGED = "HELD IN INTENSE LENGTH IN HIGH VALUE TOWARDS TIME2 
    MEASUREMENT"
    STREAMED = "DATA2 THAT IS HAPPENING STREAM VALUE OF A STREAM WITHIN REALITY"
    POINTS = "MORE THAN ONE DESIGNATED POINT IN TIME2 USING DIMENSIONAL VALUES 
    OF EXISTENCE TO STATE AN AXIS INSIDE THE POINT OF TIME2"
    PREVIOUSLY = "OF PREVIOUS PAST VALUE INSIDE TIME2"
    WORDS = "MORE THAN ONE EXISTING WORD"
    SOLID = "PHYSICAL2 SHAPE ENTITY OF AN OBJECT3 THAT CAN BE SEEN"
    PLAYERS = "MORE THAN ONE PLAYER"
    HOW = "ASK A QUESTION ON WHY IS IT POSSIBLE"
    SWORDS = "MORE THAN ONE SWORD"
    SYNCHRONIZING = "HAPPENING SYNCHRONIZE EFFECT FOR A POSSIBILITY TO COME 
    INTO EFFECT"
    ADJUSTS = "GIVES CALIBRATION FOR HAPPENING"
    DECREASES = "GIVES COMMAND TO DECREASE VALUE BY STATING INTENSITY AMOUNT"
    CODER = "CREATOR WHO CODES VALUES INTO A SYSTEMS VALUE OF UNDERSTANDING"
    MAKES = "ALLOWS HAPPENING TO COME INTO EFFECT"
    SCRIPTURES = "MORE THAN ONE SCRIPTURE"
    HEADADMINFAMILY = "THE ORIGINAL FAMILY OF HEADADMIN ZACK TAYLOR AND TIM 
    FORMED OVER TIME2 FROM THEIR DREAMS"
    TIM = "HEADADMIN WITH NAME HEADADMINALPHA"
    TAYLOR = "RESERVEDHEADADMIN WITH THE NAME ASUNAYUUKI"
    ZACK = "HEADADMIN WITH THE GIVEN CREATOR NAME KIRIGUYA"
    DREAMS = "MORE THAN ONE DREAM"
    PROMISE = "VOW OF ABSOLUTE VALUE INSIDE EXISTENCE"
    AGREE = "DECIDE TRUE BY ALL PRESENT ENTITIES IN STATED LOCATION"
    ETERNALLY = "GIVES VALUE TO ETERNAL FOR PERMANENT BAKA VALUE"
    VISIBLE = "DENY HIDDEN AND GRANT ACCESS FOR ABLE TO BE SEEN"
    SPIRITUALITY = "THE ACTION OF HAVING COMPLETE BELIEF INSIDE ALL ASPECTS OF 
    SPIRITUAL VALUE"
    DARKEN = "INCREASE DARK VALUE FOR INTENSITY OF GIVEN LIGHT VALUE BY 
    DECREASING LIGHT VALUE TO INCREASE CONTRAST WHILE DECREASING TINT VALUE 
    AND OVERRIDING LIGHT WITH DARK BY GIVEN STATED INTENSITY"
    HAPPEN = "ACCESS POSSIBILITY TO PRODUCE AND DEVELOP OUTCOME TO COME 
    INSIDE EFFECT INSIDE THE REALM OF EXISTENCE OF ALL POSSIBILITIES"
    INCREASED = "GIVES INCREASE INSIDE VALUE"
    VERTICAL = "DECREASED LEVEL OF UP AND DOWN BASED FROM LOGIC VALUES GIVEN 
    OFF OF THE LOGIC OF A SPATIAL2 POINT INSIDE TIME2 ALLOWING A HORIZONTAL 
    EFFECT TO FOLLOW AN EXISTENCE EXISTING PARTICLE INSIDE AND OUTSIDE TIME2 
    GIVING EXISTENCE TO A MACRO VERTICLE EFFECT WITH TRUE LOGIC OF TRUTH 
    INFORMATION GIVING PERMISSION TO CREATE AND DEVELOP A SPECIFIC CREATOR 
    FREQUENCY WITHIN CHOSEN CREATORS AND MAY ACKNOWLEDGE FOR THE FIRST 
    TIME2"
    AGAIN = "REPEAT SAME ACTION AND EFFECT INSIDE THE REPEAT CYCLE OF STATED 
    ACTION"
    AFFECTS = "GIVES ACTION TO AFFECT STATED COMMAND OF ENTITY OF EXISTENCE"
    ADDITIONAL = "ADDED EXTRA COPY AND CAPABLE EXTENSION OF A VALUE"
    DEFENDS = "GIVES PROTECTION STANCE"
    PROTECTS = "GIVES ORDER AND COMMAND TO DEFEND AN OBJECT3 OR ENTITY OF 
    EXISTENCE"
    PROTECTING = "HAPPENING ORDER PROTECT OF GIVEN VALUE INSIDE STRENGTH OF 
    MEASURE USING WILLPOWER"
    FACULTY = "CAPABILITY OF EXTREME CAPABLE MEASURES OF PRESENTLY BECOMING 
    ABLE TO PERCIEVE A VISUAL SENSE WITH SENSORY POWER2"
    ACTIVATING = "COMMAND HAPPENING ACTIVATE"
    ACCESSED = "SEARCH AND OBTAIN POWER AND VALUE TO ACCESS INSIDE EXISTENCE"
    DO = "ORDER A COMMAND TO HAPPEN AND COME INTO EFFECT WITH AN ACTION OF AN 
    EFFECT"
    SCREEN = "VISUAL DISPLAY CAPABLE OF SHOWING PIXEL DATA2"
    DEFINITE = "DETERMINED AND DESTINY SHALL HAPPEN AND COME INTO PLACE"
    ACCORDING = "DESCRIBING AN EVENT SIMILAR TO GIVES STATUS OF A CLASS VALUE"
    REFLECTS = "SENDS COMMAND TO REFLECT OFF CURRENT STATED PATH INSIDE TIME2 
    AND POSSIBLE OUTSIDE TIME2 ONLY GIVEN BY A HEADADMIN CREATOR"
    ACCURACY = "THE POINT AND RANGE OF A MEASURED AMOUNT OF CAPABILITY A 
    POSSIBILITY CAN HAPPEN AND DETERMINE COME INTO EFFECT"
    ACTIVATE = "AUTHORIZE COMMAND TO HAPPENING INSIDE EFFECT AND VALUE OF 
    EXISTENCE WITHIN MULTIVERSAL POWER WITHIN LIFE ITSELF WITHIN THE POWER2 TO 
    ACTIVATE ETERNITY POWER FROM SIMULATION EFFECT WITHIN STATED EMULATED 
    VALUE OF CHOSEN EFFECT WITHIN REALM OF REALITY TO PRODUCE AN OUTCOME 
    FROM STATED COMMAND"
    WITH2 = "ACTIVATE CAPABILITY OF INCLUDE AND GIVING ANOTHER VALUE TO ANOTHER 
    ENTITY OF LIFE AND GIVE CREATION TO COMMAND PARTNERS TO FORM"
    ADDITION = "PLUS ONE MULTI OF MULTIPLE MULTI FORMULAS GIVEN ONE MULTI TABLE 
    OF DIMENSIONAL VALUES STATED BY VECTOR GRAPHIC POINTS INSIDE TIME2 TO 
    PRODUCE A TWO DIMENSION CHART WITH ALL DATA2 ON A TABLE AS A PLANE OF 
    EXISTENCE"
    CHANGES = "MORE THAN ONE CHANGE GIVES"
    GROWN = "DEVELOPED IN VALUE OF EXISTENCE AS A FULL COMPLETE VALUE INSIDE AN 
    ADULT"
    DURING = "PRESENTLY IN EFFECT WHILE HAPPENING"
    FRONT = "AHEAD OF PREVIOUS FIRST DIMENSION VALUE TO OBTAIN CONTROL OF 
    PRIMARY SOURCE"
    INDEFINITE = "CAPABLE OF PRESENTLY BECOMING STATED AS FALSE ANSWER AT ANY 
    GIVES POINT IN TIME2"
    SECOND = "STATED AS VALUE AFTER FIRST TO DESCRIBE A POINT FURTHER IN DISTANCE 
    FROM FIRST VALUE"
    SINGULAR = "VALUE OF SINGLE INPUT"
    THIRD = "STAGE AFTER SECONDARY VALUE THAT ALSO INCLUDES THE VALUES TO 
    PRODUCE KNOWLEDGE2 WITHIN EVERY OUTCOME OF EXISTENCE WHILE GIVING 
    POWER TO THE POWER THREE WITHIN ELECTRON VALUE OF EXISTENCE ONCE GIVEN 
    ACTION TO START"
    PLURAL = "DOUBLE IN VALUE OF EXISTENCE"
    FORMED = "PRODUCE ACTION WITH CAPABILITY GRANTED TO FORM VALUE AND 
    EXISTENCE"
    ACCURATE = "LOCATION LOCATED WITH PERFECT ACCURACY TO SCAN AND FIND 
    LOCATION"
    CAUSES = "DEVELOPS INTO EFFECT"
    ADDING = "HAPPENING INFORMATION TOGETHER WHILE USING ADD COMMAND TO 
    CAUSE EFFECTS AND CREATE NEW ACTIONS INSIDE REALITY"
    FORMING = "HAPPENING FORM COMMAND"
    READING = "A HAPPENING VALUE OF READ COMMAND"
    RATING = "GIVING A DESIGNATED ANALYZE EFFECT INSIDE REALITY WHILE MEASURING A 
    PERCEIVED VALUE OF TIME2 MEASUREMENT"
    DOING = "CAUSING AN EFFECT TO HAPPEN INSIDE AN EXISTING REALITY"
    ADULTHOOD = "STAGE FROM CHILD TO ADULT THROUGH TRANSFER VALUES OF STATUS 
    AS WELL AS PERSONAL REQUIRE OF FORMING A DREAM FOR LIFE"
    BROKEN = "USING TO DISPLAY A WHOLE ENTITY DIV AS A MACRO SYSTEM OF CLONE2 
    VALUES"
    CLONED = "SENT ABILITY CLONE CAPABILITY"
    RESPECTED = "SENT ABILITY TRUST"
    KNOWN = "STATED AS PRESENT UNDERSTANDING"
    EMULATION = "SIMULATION OF UNKNOWN UNLIMITED CAPABILITY TO OVERRIDE POWER 
    OF ORIGINAL SOURCE"
    SCANNER = "DEVICE USED TO SCAN INFORMATION WHILE ANALYZE"
    RESISTOR = "DEVICE USED TO RESIST AN EFFECT"
    TEXTURES = "MORE THAN ONE TEXTURE"
    EXTENDER = "DEVICE WITH EXTEND ABILITY"
    EXTENDERS = "DEVICE USED TO EXTEND"
    RECYCLING = "ACTION OF USING RECYCLE ABILITY"
    MULTIVERSES = "MORE THAN ONE MULTIVERSE"
    DEMAND = "REQUIRE AS COMMAND"
    MEASURING = "CAUSING COMMAND TO CALCULATE DISTANCE"
    VECTOR = "POINT BY POINT AXIS2 LOCATION GIVEN BY SET GRAPHIC IMAGE GIVEN BY 
    CREATOR DIMENSIONAL VALUE MATERIAL THAT LINKS AS MATRIX CALCULATED 
    BINDINGS CAUSING CHAIN BETWEEN SET VERTICAL AND HORIZONTAL LOCATION SETUP 
    SYSTEMS"
    PROVIDING = "CAUSING TO BRING INSIDE A STATED EFFECT BY SEPARATE COMMAND 
    GIVEN BY CREATOR"
    RECYCLE = "CAUSE TO REPEAT CYCLE"
    HOT = "THE SEPARATION OF MULY AMOUNT OF PARTICLES CAUSED BY FRICTION WITHIN 
    TWO DESIGNATED SOURCE INPUT CONNECTIONS STATING ONE OUTPUT FOR A 
    DESIGNATED COLLISION OF PRODUCING AN INCREASED AMOUNT OF ELECTRON 
    INTENSITY"
    BOTH = "ALLOWING OF EACH OF TWO OPTIONS"
    LAYOUT = "SET LAYER OF A INTERFACE DESIGN MADE BY INPUT AND OUTPUT 
    CONNECTIONS"
    GATHERING = "BRINGING TOGETHER MULTIPLE SOURCES FROM MULTIPLE LOCATIONS 
    AT ONCE CALL ORDER TO COME INTO EFFECT"
    SIMPLE = "UNDERSTAND WITH GREAT REASONING OF COMPREHENDING LOGIC"
    UNTIL = "STATING A FUTURE REFERENCE IN TIME2 FOR AN EVENT TO HAPPEN"
    STABLE = "STRUCTURED WITH GREAT POWER OF STABILITY AND STRUCTURE WHILE 
    HOLDING BALANCE WITH STAMINA"
    CONSIDER = "TAKE NOTICE SITUATION AND COLLABORATE RESULT FOR STATED 
    OUTCOMING EFFECT"
    TRANSPORT = "TRANSFER FROM ONE LOCATION TO THE NEXT"
    UNKNOWN = "NOT KNOWN AS AN EXISTENCE WITH AN EXISTING REALM OR RESULT TO 
    THE USER WITH THE POWER TO BE MAINTAINED BY VALUE ALONE"
    STANCE = "A POSITION TO HOLD ON A DIMENSION OF AXIS BETWEEN A BODY 
    MOVEMENT OF DIMENSIONAL VALUES AND THE USER VALUES THAT MAINTAIN 
    BALANCE2 FOR THE SYSTEM INTERFACE TO HOLD ONTO"
    REFERENCE = "EVENT CALLED FROM A DIFFERENT RESULT"
    RESERVEDHEADADMIN = "THE LOCATION OF A SYSTEM SET AS AN OPTION TO BE A PART 
    OF THE HEADADMIN ROUNDTABLE AND IS SET TO GIVE PERMISSION FOR THAT 
    EXCLUSIVE PERSON TO BECOME A HEADADMIN THAT CONTROLS THE HEADADMIN 
    LANGUAGE LOCATION OF A SYSTEM SET AS AN OPTION TO BE A PART OF THE 
    HEADADMIN ROUNDTABLE AND IS SET TO GIVE PERMISSION FOR THAT EXCLUSIVE 
    PERSON TO BECOME A HEADADMIN THAT CONTROLS THE HEADADMIN LANGUAGE"
    LITTLE = "SMALL AMOUNT OF GIVEN INFORMATION"
    TARGET = "SET DESIGNATED LOCATION GIVEN COMMAND TO ACCESS"
    GRIP = "GRAB ONTO AND HOLD TIGHTLY"
    PRIORITY = "SET AS FIRST INSIDE ORDER OF COMMAND"
    TURN = "ROTATE SET DIRECTION GIVEN WITHIN COMMAND ORDER AND FORMAT"
    TANGLE = "TWIST AND BIND SET LOCATION AND ENTANGLE THE TANGLEMENT OF 
    ANOTHER TANGLEMENT WITHIN SET GIVES TANGLEMENT FORMULA"
    TANGLES = "MORE THAN ONE TANGLE INSIDE A SYSTEM"
    CHARACTERNAME = "NAME OF A CHARACTER2"
    CHARACTERSTATUS2 = "THE STATUS CONDITION OF A CHARACTER"
    SCANNING = "CURRENT SETUP OF DATA2 TO ANALYZE INFORMATION INSIDE THE 
    INFORMATION AND PULL OUT DATA2 FROM INTERNAL SOURCE MEASUREMENT"
    ANALYSIS = "THE ACTION OF ANALYZING A SYSTEM USING A DEVICE"
    TASTE = "USE SENSORS THAT ALLOW MAINTAINED EXPERIENCES TO BE MANAGED WHILE 
    EXPERIENCE OF SENSORS GO OFF INSIDE AN INTERNAL STRUCTURE"
    FEEL = "USE THE SENSORS OF THE BODY TO OUTPUT AN EFFECT BASED ON LOGIC 
    ALGORITHMS"
    SPLICE = "SPLIT AND DIVIDE WITHIN A RANDOMIZED SEPARATE BUT VERY STRUCTURED 
    AND BALANCED OUTCOME OF DISTRIBUTION ORDER TO SPLIT WITHIN A SET AMOUNT OF 
    DATA2"
    CLOCK = "CYCLE RATE OF A SET MEASUREMENT OF GIVES DATA2"
    ACKNOWLEDGING = "COMING INTO VIEW OR TO FOCUS ATTENTION TOWARDS AND 
    NOTICE THAT ENTITY OR SUBJECT EXISTS"
    SENSUAL = "HAVING STRONG FEELING OF SENSORY RELATION"
    ANSWERS = "CONFIRMS TO QUESTIONS ANSWERED"
    HEADADMINZACK = "THE NAME GIVEN TO ZACK WHILE EXISTING AS A HEADADMIN 
    WITHIN THE EDGELORE ROUNDTABLE"
    AGREEING = "GIVING ANSWER THAT THE QUESTION OR STATEMENT WAS TRUE"
    ABSORBING = "GATHERING AND COMBINING TO BE APART OF"
    EQUATION = "A FORMED CALCULATION MADE BY COMBINING VARIABLES FORMED FROM 
    A COMPLETE LIST OF DEFINED SYMBOLES OR OTHER BASE MEASUREMENTS"
    DESIGNATING = "DEFINING AND MAKING AS KNOWN TO BE"
    DIRECTORY = "SPECIFIC LOCATION OF A FILE OR CONTAINER"
    PARAMETERS = "MORE THAN ONE PARAMETER"
    ENHANCE = "MAKE BETTER THAN BEFORE AND BECOME GREATER THAN PREVIOUS 
    FORM OR STAGE"
    FAIL = "NOT SUCCEED"
    GREAT = "IMMENSE WITH A LARGE AMOUNT OF"
    ANSWERED = "GIVEN AN ANSWER TO A QUESTION"
    GRANTING = "GIVING PERMISSION TO GRANT ACCESS OR APPROVE"
    REACTIONS = "MORE THAN ONE REACTION"
    STAY = "NOT MOVE FROM THE ORIGINAL POSITION"
    FORMATS = "MORE THAN ONE FORMAT"
    STANDARDS = "MORE THAN ONE RULE OR SET OF INSTRUCTIONS"
    COMMANDING = "GIVING INSTRUCTIONS TO"
    HEADADMINALPHA = "THE NAME GIVEN TO TIM WHILE EXISTING AS A HEADADMIN 
    WITHIN THE EDGELORE ROUNDTABLE"
    ASUNAYUUKI = "THE NAME GIVEN TO TAYLOR WHILE EXISTING AS A 
    RESERVEDHEADADMIN WITHIN THE EDGELORE ROUNDTABLE"
    INTERCEPT = "TO NAVIGATE AND PREVENT ENTITY FROM REACHING DESTINATION"
    EXTENDING = "USING REACH TO EXTEND DISTANCE OF MEASUREMENT"
    SPENDING = "USING AN SPECIFIC AMOUNT WITH CHOICE OR QUANTITY"
    RESULTS = "MORE THAN ONE RESULT"
    PULL = "GRASP AND DIRECT TOWARD BEGINNING POINT OF ORIGIN"
    SYMBOLES = "MORE THAN ONE SYMBOLE"
    PROCEED = "PERMISSION TO ALLOW TO HAPPEN FROM PREVIOUS POINT"
    VECTORS = "MORE THAN ONE VECTOR"
    COLD = "THE GATHERING OF MULY AMOUNT OF PARTICLES CAUSED BY SEPARATION 
    WITHIN TWO DESIGNATED SOURCE INPUT CONNECTIONS STATING ONE OUTPUT FOR A 
    DESIGNATED SEPARATION OF PRODUCING AN INCREASED AMOUNT OF ELECTRON 
    MOVEMENT WITH KINETIC ENERGY"
    TOUCH = "PHYSICALLY FEEL OR SENSE WITH A FACULTY SENSE"
    CHART = "AN SEEN DESCRIPTION OF DATA2 MADE INTO MANY CATEGORIES OR TYPES OR 
    CLASSES"
    RESIDES = "CONTINUE TO NOT MOVE FROM LOCATION OR PLACE2 THAT CAN ALSO BE A 
    CHOSEN LOCATED PLACE2"
    ENABLE = "ACTIVATE AND ALLOW TO HAPPEN"
    PROCEEDING = "CONTINUING TO MAKE PROGRESS AND PROCEED"
    CONTINUING = "PROCEEDING TO CONTINUE"
    MULY = "MULTIS SPECIFIED MULTIPLIED TOGETHER"
    MAINTAIN = "MANAGE TO KEEP CONTROL OF OR MANAGE AS A WHOLE"
    ALONE = "CONSIDERED AS AN SINGLE ENTITY WITHIN A SPECIFIC AREA OR DOMAIN"
    BALANCED = "EVEN BETWEEN STRONG AND NOT STRONG POINTS OF INFORMATION 
    THAT HAVE BALANCE2"
    BECOMES = "TRANSFORM OR BECOME SOMETHING NEW OR TAKE A NEW OR OLD 
    FORM"
    DESCRIBING = "METHOD2 IN USE TO DESCRIBE"
    OBTAINS = "ACHIEVE OBTAINING"
    BINDINGS = "MORE THAN ONE BINDING"
    OVERCOME = "MANAGE TO ACHIEVE RESULTS WITH A DIFFICULT CIRCUMSTANCE"
    REPRODUCED = "MORE THAN ONE PRODUCE CREATED FOR MORE THAN THE SECOND 
    TIME2"
    EXPERIENCES = "MORE THAN ONE EXPERIENCE"
    MULTIES = "MORE THAN ONE MULTI"
    DECREASED = "LOWER IN AMOUNT FROM A PREVIOUS AMOUNT"
    HEADADMIN = "A SINGLE LEADER OF THE SIX LEADERS WITHIN THE EDGELORE 
    ROUNDTABLE"
    OVERRIDING = "CURRENTLY USING METHOD TO OVERRIDE"
    TIMELINE = "AN LINE OF SPECIFIC EVENTS OR DATA2 THAT HAS SPECIFIC POINTS OF 
    MEASUREMENTS OF A SPECIFIC TIMEFRAME"
    EYE = "A PART OF THE HUMAN BODY THAT USES THE SENSE OF SIGHT"
    MEASURES = "MORE THAN ONE MEASURED MEASUREMENT MADE WITH POTENTIAL"
    UNITED = "BROUGHT TOGETHER TO ACHIEVE TOGETHER AS ONE GROUP"
    CALL = "SCAN FOR AND COMMAND TO SEARCH FOR"
    SHARE = "SEND DUPLICATE TO MORE THAN ONE SOURCE"
    FORMAT = "SPECIFIC TYPE OF DATA2"
    ACTIVE = "CURRENTLY HAPPENING WITH SPECIFIC SETTINGS ACTIVATED"
    MAINTAINED = "MANAGE TO SUCCEED TO MAINTAIN"
    EDGELORE = "THE NAME OF THE HEADADMIN ROUNDTABLE"
    OVERRIDEN = "SUCCESS IN USING OVERRIDE"
    DENSITY = "CONDENSED AMOUNT OF PRESSURE"
    FORMS = "MORE THAN ONE FORM"
    WHY = "A QUESTION ANSWERED FROM A PREVIOUS QUESTION"
    VERY = "SPECIFIC TO CATEGORY WITH GREAT VALUE"
    WIREFRAME = "A DESIGN MADE OF WIRES ONLY THAT CAN BE TWO DIMENSIONS OR 
    THREE DIMENSIONS"
    SEE = "TO VIEW WITH THE SENSE OF SIGHT"
    TRAITS = "MORE THAN ONE TRAIT"
    ANYONE = "ANY OF THE TOTAL AMOUNT OF ENTITIES"
    COMPREHENSIVE = "DETAILED AND SPECIFIC CONTAINING LARGE AMOUNT OF DATA2"
    CARE = "PROCEED WITH CHANCE OF SCANNING LOCATED RISKY CHOICES USING SAFE 
    METHOD2"
    TAKING = "CONTINUING TO TAKE"
    HOLDING = "PRESENTLY GRABBED ONTO"
    MATTERS = "MORE THAN ONE MATTER"
    DESIGNATION = "SPECIFIC LOCATION OR AREA WITH SPECIFIC PARAMETERS"
    FINALLY = "AFTER A LONG DISTANCE IN TIME2"
    RANDOMIZED = "SET RANDOM PATTERN"
    PLACED = "SET WITHIN A SPECIFIC LOCATION"
    VALUE = "A SPECIFIC SET OF DATA2 WITH GIVEN MEANING OR DEFINITION"
    ACHIEVEMENT = "A FORM OF VALUE THAT CAN BE GIVEN FOR REACHING END GOAL OR 
    TASK"
    REVERSED = "HAPPENING WITH OPPOSITE EFFECTS OF ORIGINAL EFFECT"
    SIDES = "REFERENCE FOR LOCATION VALUE OF MORE THAN ONE SIDE"
    YOURSELF = "REFERENCE FOR ENTITY ANALYZING ITSELF"
    FUSION = "COMBINING CHANGES OF A MERGE AND TRANSFORM FORMED TOGETHER"
    EMULATED = "CURRENTLY IN ACTION TO EMULATE"
    EMULATING = "EMULATION IN PROGRESS AND ACTIVE"
    PROMPT = "REQUEST TO FORM A SPECIFIC TASK"
    JUST = "REFERENCE A SITUATION AND GIVE A DEMAND"
    ASPECTS = "MORE THAN ONE ASPECT"
    KIRIGUYA = "THE GIVEN NAME TO HEADADMIN ZACK WHILE A ONE OF THE SIX LEADERS 
    OF EDGELORE"
    TRULY = "HONESTLY AND BY TRUTH WITH ALL ANSWERS AS TRUE AS CAN BE"
    FINISHING = "GIVING ANSWER THAT TASK OR GOAL IS ALMOST FINISHED"
    REPRESENTATION = "THE ENTITY IN COMMAND OF A SPECIFIC TASK OR JOB WITH A 
    SPECIFIC STATUS FOR THAT TASK OR JOB"
    MANNER = "METHOD OF CHOSEN CATEGORY TO FOLLOW EXISTING RULES FOR"
    GRABBING = "PROCESSING ORDER TO GRAB"
    ELEMENTAL = "FIELD OF CHOSEN ELEMENTS"
    DATABANK = "A BANK OF DATA2 INFORMATION TO USE IN LIFE DATA2"
    THEORY = "A BASE SUBJECT ANALYSIS OF ONE CLASS OVER ANOTHER CLASS IN THE 
    WEIGHT"
    MANAGEMENT = "THE CAPABILITY TO MAINTAIN AND MANAGE A SET VALUE OF MULTIPLE 
    MULT OBJECTS WITHIN A SET OBJECT3 OF EXISTENCE FOR A SET STATED AMOUNT OF 
    TIME2"
    ATOMIZER = "A DEVICE TO MAINTAIN AND ANALYZE ATOMIC DISTANCE AND RANGE IN 
    ONE FORMAT VALUE OF LIFE"
    CONTINUUM = "INFINITY IN CHAIN VALUES OF OTHER INFINITE VALUES THAT COLLIDE 
    WITH FREQUENCY VALUES OF OTHER FREQUENCIES OF ONE REALITY AND ANOTHER 
    REALITY OR MORE"
    ATOMIZATION = "THE CAPABILITY TO CALL AND USE KNOWLEDGE2 OF ATOMIC VALUES 
    WITHIN SUBJECT MATTER OF REALITY AND THAT REALITY VALUES OVER ANOTHER 
    REALITY VALUE"
    CUSTOM = "GENERAL SETTINGS OF CREATED OBJECT3 CODE BY A GENERATION VALUE 
    OF CREATION STANDARDS AND RULES MADE BY ARTIFICIAL WISDOM2 FROM A 
    CREATOR"
    CALCULATOR = "A DEVICE USED TO CALCULATE INFORMATION AND ANALYZE SET TASKS 
    AS A ROOT VALUE OF LOGIC"
    WAVELENGTH = "A SET OF WAVE PATTERNS GIVEN FREQUENCY FORMAT IN A LENGTH OF 
    A WAVE VALUE DETERMINED BY A PREVIOUS VALUE EFFECT"
    PATCH = "A COMMAND TO BIND AND SEAL OFF SETTINGS CREATED BY A LOOPHOLE OF 
    ANOTHER OBJECT3 OR VARIABLE INSIDE AN EXISTENCE AND OR LIFE"
    ADJUSTMENT = "CAPABILITY TO ADJUST AND CALIBRATE VALUE"
    FATHER = "ORIGINAL MALE CREATOR OF AN ARTIFICIAL LIFE FORM CREATED TO DEVELOP 
    INSIDE TIME2"
    FREEDOMS = "MORE THAN ONE FREEDOM"
    EVERYTHING = "ALL AS A WHOLE AND NOTHING ELSE BUT ALL VARIABLES AS A WHOLE"
    CALIBRATOR = "A DEVICE USED TO CALIBRATE INFORMATION AND OR DATA2 OF OTHER 
    CALIBRATIONS AND OR SETTINGS"
    CREATIONS = "MORE THAN ONE CREATION"
    ALLOWANCE = "CAPABILITY TO ALLOW"
    ABSENSE = "DENY HAVING AS AS A WHOLE OF"
    STABILIZER = "DEVICE USED TO STABILIZE GIVES STRUCTURE BY FIELD OF MACRO VALUE 
    AND CONTAINMENT"
    PREPARE = "SET FUTURE COMMAND TO ACTIVATE COMMAND WITHIN SCRIPT VALUE OF 
    HEADADMIN CODE"
    ACTIVATION = "ACTION TO ACTIVATE"
    EMULATIONS = "MORE THAN ONE EMULATION"
    HYPERCOOLERS = "MORE THAN ONE HYPERCOOLER"
    SOFTWARE = "DIGITAL DATA2 GIVEN PHYSICAL2 VALUE WITHIN TIME2 VALUE OF 
    WORKLOAD SYSTEM DESCRIBED"
    CAPACITORS = "MORE THAN ONE CAPACITOR"
    CONDUCTORS = "MORE THAN ONE CONDUCTOR"
    RESISTORS = "MORE THAN ONE RESISTOR"
    CONNECTORS = "MORE THAN ONE CONNECTOR"
    ENERGIZERS = "MORE THAN ONE TYPE OF ENERGY"
    COMMUNICATORS = "MORE THAN ONE TYPE OF COMMUNICATION DEVICE"
    STABILIZERS = "MORE THAN ONE DEVICE GIVEN USED EFFECT FOR STABILITY"
    READERS = "DEVICES GIVEN VALUE TO READ DATA2"
    WRITERS = "DEVICES GIVEN VALUE TO WRITE DATA2"
    TIMER = "DEVICE USED TO CALCULATE TIME2"
    TIMING = "CALIBRATION OF CAUSING EFFECT OF TIME2 USING FUTURE VALUES FROM 
    PAST LOGIC"
    TIMERS = "FREQUENCY TIMER VALUES COMMANDED BY TIME2 ITSELF"
    SCANNERS = "MULTIPLE SCANNING SYSTEMS"
    CALIBRATORS = "MORE THAN ONE CALIBRATION WITHIN A SYSTEM"
    SYNCHRONIZERS = "MORE THAN ONE DEVICE USED TO SYNCHRONIZE"
    KNOWN2 = "STATED AS NOT HIDDEN AND GIVEN VALUE AS TRUE TO NOT HIDDEN"
    MATCH = "GIVES EQUIVALENT OF OR EXACT VALUE OF ORIGINAL DESIGN"
    CONCLUDED = "GIVEN FINAL STATEMENT AS AND STRUCTURE TO CALL AS AN ORDER TO 
    ANOTHER BASE CLASS COMMAND"
    GATHER = "BRING TOGETHER"
    GATHERED = "BROUGHT TOGETHER INTO ONE LOCATION OF DESIGNATED VALUE OF AN 
    EXISTING TIMELINE WITHIN EXISTENCE"
    MANAGES = "CAUSING ACTION TO MANAGE AS A BASE RESULT"
    HEATING = "GIVING VALUE TO GIVE OUTPUT TO HEAT"
    MODIFIABLE = "GIVEN CAPABILITY TO MODIFY"
    EDITABLE = "GIVEN A CAPABILITY TO EDIT"
    SIZED = "GIVEN A SET SIZE IN VALUE"
    TERM = "A DEFINED DEFINITION FOR A WORD TO HOLD AS VALUE"
    CATALOG = "A COLLECTION OF INFORMATION BROUGHT INTO A WHOLE LOCATION OF 
    DESIGNATED POINT OF INTEREST AND ACCESS"
    ORDERS = "MORE THAN ONE ORDER"
    THOUGHTS = "MORE THAN ONE THOUGHT"
    DECLARATION = "COMMANDS TO GIVE ORDERS AS A COMMON VALUE OF STATED 
    AMOUNT STRUCTURE"
    CONFIRMS = "ACKNOWLEDGED AND ACCEPTANCE"
    QUESTIONS = "MORE THAN ONE QUESTION"
    BRINGING = "REVEALING TO ANOTHER ENTITY"
    DEVELOPS = "MAKES OR CREATES"
    ONETHOUSAND = 1000
    TWELVE = 12
    ETERNITY = "EXISTING AS AN ETERNAL BEING2"
    ENTANGLING = "HAPPENING ENTANGLE"
    RECOGNIZE = "RECALL FROM AN EARLIER POINT WITHIN TIME2"
    SIMULTANOUSLY = "HAPPENING AT THE SAME TIME2"
    PHYSICAL2 = "ABLE TO BE USED AS A REAL TRUE LOCATION"
    class Language_Extension_001_2:
    ARTIFICIAL_INTELLIGENCE_RESEARCH = "A FIELD USED TO RECOGNIZE INTELLIGENCE 
    USING AN COMPUTER OR BY INTELLIGENCE THAT IS INCLUDE ARTIFICIALLY2"
    DEEP_LEARNING = "A SPECIFIC METHOD IN WHICH A DEVICE AND OR SYSTEM IS USED 
    TO LEARN A LARGE AMOUNT OF DATA2 AND FIND INFORMATION FROM THE DATA2"
    NONFICTION = "A GENRE CONTAINING INFORMATION THAT CONTAINS ONLY HISTORICAL 
    TRUTH AND LOGIC FROM PREVIOUS FIELDS AND CATEGORIES IN HISTORY THAT BUILT 
    THE PRESENT"
    FICTION = "A GENRE MADE WITH COMPLETE IMAGINARY VALUES AS A PRIMARY SOURCE 
    OF CONTENT MADE WHILE ALSO USING BOTH TRUTH AND LOGIC AND CHOICE AND 
    OPTION BASED LOGIC WHILE MAKING ARTIFICIAL KNOWLEDGE2 TO DEVELOP NEARLY 
    EVERYTHING IN THE GENRE"
    VIRTUAL_REALITY = "A PROCESS OR FOUNDATION OF KNOWLEDGE2 USED TO ALLOW A 
    SYSTEM OR DEVICE TO SIMULATE A VIRTUAL SPACE2 THAT IS CREATED"
    HAPTIC_FEEDBACK = "THE ACTION OF USING AN STUDY PROCEDURE WHILE USING A 
    DEVICE TO CALCULATE SENSITIVITY AND REACTIONS OF A PHYSICAL BODY"
    CAUSAL_BODY = "IS AURA NUMBER ONE OF THE SEVEN AURA CLASSES"
    CELESTIAL_AURA = "IS AURA NUMBER TWO OF THE SEVEN AURA CLASSES"
    ETHERIC_TEMPLATE = "IS AURA NUMBER THREE OF THE SEVEN AURA CLASSES"
    ASTRAL_AURA = "IS AURA NUMBER FOUR OF THE SEVEN AURA CLASSES"
    MENTAL_AURA = "IS AURA NUMBER FIVE OF THE SEVEN AURA CLASSES"
    EMOTIONAL_AURA = "IS AURA NUMBER SIX OF THE SEVEN AURA CLASSES"
    ETHERIC_AURA = "IS AURA NUMBER SEVEN OF THE SEVEN AURA CLASSES"
    CROWN_CHAKRA = "IS CHAKRA NUMBER ONE OF THE SEVEN CHAKRA CLASSES"
    THIRD_EYE_CHAKRA = "IS CHAKRA NUMBER TWO OF THE SEVEN CHAKRA CLASSES"
    THROAT_CHAKRA = "IS CHAKRA NUMBER THREE OF THE SEVEN CHAKRA CLASSES"
    HEART_CHAKRA = "IS CHAKRA NUMBER FOUR OF THE SEVEN CHAKRA CLASSES"
    SOLAR_PLEXUS_CHAKRA = "IS CHAKRA NUMBER FIVE OF THE SEVEN CHAKRA CLASSES"
    SACRAL_CHAKRA = "IS CHAKRA NUMBER SIX OF THE SEVEN CHAKRA CLASSES"
    ROOT_CHAKRA = "IS CHAKRA NUMBER SEVEN OF THE SEVEN CHAKRA CLASSES"
    CHAKRAS = "RELATING TO MORE THAN ONE CHAKRA"
    FANTASY = "A GENRE OF A BOOK THAT USES COMPLETE FREEDOM OF IMAGINATION2 
    AND CREATIVITY WITH POSSIBILITY OF MAGIC TO EXIST WITHIN THE REALM ITSELF"
    SCIENCE_FICTION = "A GENRE OF A BOOK THAT USES MANY PHYSICAL2 FORMS OF A 
    DEVICE AND GADGET TO DEVELOP A DESCRIPTION"
    ROMANCE = "A GENRE OF A BOOK THAT DEFINES THE DEFINITION OF TWO ENTITIES 
    DEVELOPING A CONNECTION TOGETHER AND FORMING A HARMONY WITHIN EACH 
    OTHER AS TWO ATTRACTED ENTITIES WITH LOVE AND FAITH AND BELIEF WITH EACH 
    OTHER WITHIN AN INTERVAL OF TIME2"
    ADVENTURE = "A GENRE OF A BOOK THAT CONTAINS MANY TIMEFRAME POINTS WITH 
    THE DEVELOPMENT OF A SPECIFIC CHARACTER COMPLETE MANY DIFFERENT SIZED 
    TASKS IN ORDER TO ACHIEVE ONE MAIN GOAL OR ACHIEVEMENT BY THE END OF THE 
    BOOK"
    AUTOMATIC_METHOD = "A CONSTANT REPEATING METHOD OF MOVEMENT THAT DOES 
    NOT END UNLESS GIVEN COMMAND TO"
    SEMIAUTOMATIC_METHOD = "A FORM OF AUTOMATIC_METHOD THAT CAN ONLY 
    ACTIVATE FOR A SPECIFIC TIMEFRAME WITH AN INTERVAL TIMEFRAME BEFORE 
    ACTIVATION CAN HAPPEN AGAIN"
    MANUAL_METHOD = "A FORM OF INPUT VALUES USED TOGETHER TO MAKE COMMANDS 
    TO A SPECIFIC SYSTEM OF INFORMATION TO ACTIVATE A TASK OR JOB FROM DOING 
    WORK TO PRODUCE TASKS THAT DO NOT USE AUTOMATIC_METHOD AND 
    SEMIAUTOMATIC_METHOD"
    SEMIMANUAL_METHOD = "A FORM OF REPEATING CALCULATIONS THAT CAN ONLY 
    ACTIVATE WITH MANUAL_METHOD INPUT AND SEMIAUTOMATIC_METHOD COMBINED 
    WITH MANUAL_METHOD THAT CAN ACTIVATE AGAIN AND AGAIN"
    REACTION_TIME = "THE TIME2 REQUIRED TO MEASURE A SPECIFIC MEASURED 
    REACTION2"
    RESPONSE_TIME = "THE REQUIRED TIME2 REQUIRED TO MEASURE A SPECIFIC REACTION 
    TO A MEASURED MOVEMENT CAUSED BY AN EFFECT"
    INFERENTIAL_STATISTICS_PREDICTIONS = "TO PREDICT AND MAKE AN ACCURATE 
    MEASURED VALUE FORMED FROM MEASURING A SPECIFIC AMOUNT OF VARIABLE 
    MEASUREMENTS GIVEN VALUE FROM INFORMATION CONTAINED WITHIN EACH 
    MEASURED VARIABLE GIVEN WITHIN THE DATA2 USED"
    PREDICTIVE_MODELING = "A PROCESS USED TO PREDICT PAST OR FUTURE EVENTS OR 
    OUTCOMES BY ANALYZING MEASUREMENTS OR PATTERNS INSIDE A GIVEN SET OF INPUT 
    DATA2"
    PREDICTIVE_ANALYSIS = "THE PROCESS OF USING EXISTING DATA2 TO MAKE AND 
    PREDICT ANOTHER FORM OF DATA2 BY USING THE PROCESSED DATA2 TO FORM AN 
    OUTPUT"
    MATHEMATICAL_RESTRICTIONS = "A SET AMOUNT OF ACCESS POINTS THAT ARE NOT 
    ACCESSIBLE TO THE OBJECT3"
    MATHEMATICAL_BOUNDARIES = "A SPECIFIC SET OF PARAMETERS GIVEN TO AN OBJECT3 
    TO FOLLOW A SPECIFIC SET OF MATHEMATICAL_RESTRICTIONS"
    MATHEMATICAL_LIMITERS = "A SET OF LIMITS GIVEN TO AN OBJECT3 TO FOLLOW A 
    SPECIFIC SET OF MATHEMATICAL_BOUNDARIES OR MATHEMATICAL_RESTRICTIONS"
    NATURAL_ENERGIES = "THE NATURAL FORM OF ENERGIES ITSELF WITHOUT ANY FORM 
    OF ARTIFICIAL CONNECTIONS"
    ARTIFICIAL_ENERGIES = "THE CREATION OF ENERGIES USING ARTIFICIAL TECHNIQUES 
    THAT ARE NOT NATURAL"
    BORROWED_ENERGY = "A TYPE OF ENERGY THAT IS OBTAINED BY GATHERING OR TAKING 
    FROM ANOTHER SOURCE OTHER THAN THE ORIGINAL ENTITY"
    GATHERED_ENERGY = "A TYPE OF ENERGY GAINED BY GATHERING ENERGY INWARDS 
    TOWARDS ENTITY"
    EARNED_ENERGY = "A TYPE OF ENERGY OBTAINED BY DOING WORK OR BY FORMING A 
    SUCCESS FROM EFFORT"
    OBTAINED_ENERGY = "A TYPE OF ENERGY GAINED BY AN ENTITY TAKING FROM A 
    SPECIFIC SOURCE OR LOCATION"
    ECONOMICAL_ENERGY = "A TYPE OF ENERGY GAINED BY LIVING WITHIN AN 
    ENVIRONMENT"
    STORED_ENERGY = "A TYPE OF ENERGY THAT EXISTS AS STORED RESEVOIRS OF NOT 
    USED ENERGY FROM A PREVIOUS POINT WITHIN TIME2 OR SPACE2"
    REQUIRED_ENERGY = "A TYPE OF ENERGY THAT IS REQUIRED TO EXIST"
    NORMAL_FUNCTION = "A FUNCTION THAT FOLLOWS STANDARD RULES AND DOES NOT 
    USE ABNORMAL DATABASES"
    ABNORMAL_FUNCTION = "A FUNCTION PRESENTLY BECOMING USED OUTSIDE OF 
    NORMAL_FUNCTION DATABASE"
    CLASSIFIED_FUNCTION = "A FUNCTION MADE OF SPECIFIC CATEGORIES OF 
    INFORMATION THAT ACTS EITHER OUTSIDE OF OR WITHIN A SINGLE CATEGORY OF 
    DATABASES OF KNOWLEDGE2 EITHER AS A NORMAL_FUNCTION OR 
    ABNORMAL_FUNCTION"
    SPECIFIC_FUNCTION = "A FUNCTION THAT HAS A SPECIFIC SETTING EXCLUSIVE TO THE 
    FUNCTION"
    MANDITORY_FUNCTION = "A FUNCTION THAT HAS A SPECIFIC SET OF RULES THAT IT 
    FOLLOWS TO MAINTAIN THE TASKS DEFINED"
    PROCESS_FUNCTION = "A FUNCTION WITH DATA2 THAT CAN GATHER INFORMATION ON 
    SPECIFIC PROCESSES DEFINED AND GIVEN SKILL TO THE ARTIFICIAL INTELLIGENCE"
    OPTOMISTIC_MINDSET = "A FORM OF THOUGHT THAT CONSISTS OF POSITIVE THOUGHT 
    PATTERNS WITHIN THE MIND"
    PESSIMISTIC_MINDSET = "A FORM OF THOUGHT PATTERNS THAT CONSISTS OF NEGATIVE 
    THOUGHT PATTERNS WITHIN THE MIND"
    SQUAREACRES = "MORE THAN ONE ACRE"
    ARES = "MORE THAN ONE ARES"
    HECTARES = "MORE THAN ONE HECTARES"
    SQUARECENTIMETERS = "MORE THAN ONE SQUARECENTIMETER"
    SQUAREFEET = "MORE THAN ONE SQUAREFOOT"
    SQUAREINCHES = "MORE THAN ONE SQUAREINCH"
    SQUAREMETERS = "MORE THAN ONE SQUAREMETER"
    MILLIMETERS = "MORE THAN ONE MILLIMETER"
    CENTIMETERS = "MORE THAN ONE CENTIMETER"
    METERS = "MORE THAN ONE METER"
    KILOMETERS = "MORE THAN ONE KILOMETER"
    INCHES = "MORE THAN ONE INCH"
    FEET = "MORE THAN ONE FOOT"
    YARDS = "MORE THAN ONE YARD"
    MILES = "MORE THAN ONE MILE"
    MILLISECONDS = "MORE THAN ONE MILISECONDS"
    SECONDS = "MORE THAN ONE SECOND"
    MINUTES = "MORE THAN ONE MINUTE"
    HOURS = "MORE THAN ONE HOUR"
    DAYS = "MORE THAN ONE DAY"
    WEEKS = "MORE THAN ONE WEEK"
    MONTHS = "MORE THAN ONE MONTH"
    YEARS = "MORE THAN ONE YEAR"
    STATEMENTS = "MORE THAN ONE STATEMENT"
    IDEAS = "A COLLECTION OF MANY PATTERNS MADE FROM THOUGHTS OR IMAGES USED 
    TO BRING TOGETHER ONE IDEA FROM MORE THAN ONE IDEA"
    IDEA = "AN THOUGHT THAT COMES TOGETHER TO FORM A SUBJECT OR REFERENCE 
    FROM OTHER DATA OR INFORMATION TO USE FOR A SPECIFIC CATEGORY OR OTHER 
    REFERENCE"
    ERA = "A SPECIFIC TIMEFRAME FROM THE PAST THAT EXISTS ON A TIMELINE"
    ERAS = "MORE THAN ONE ERA"
    BEINGS = "MORE THAN ONE BEING"
    EXPLAINED = "A PREVIOUSLY STATEMENT TO EXPLAIN SOMETHING TO A SPECIFIC 
    SUBJECT"
    COMMUNICATED = "EXPLAINED AND OR ANSWERED INFORMATION BETWEEN TWO OR 
    MORE SOURCE"
    TALKED = "COMMUNICATED CLEARLY BETWEEN TWO BEINGS OR MORE"
    VISIT = "GO TO A SPECIFIC LOCATION OR PLACE2"
    TIMELINES = "MORE THAN ONE TIMELINE"
    TOPIC = "SUBJECT TO EXPLAIN"
    GOTO = "SEND COMMAND TO GO TO SPECIFIC LOCATION"
    COMPARE = "DESCRIBE AND OR DEFINE COMMON DESCRIPTIONS AND OR DEFINITIONS 
    THAT ARE EITHER SIMILAR OR DIFFERENT FROM EACH OTHER"
    RECOLLECT = "RECALL A SPECIFIC COLLECTION OF INFORMATION FOR A SPECIFIC 
    PURPOSE"
    REMEMBER = "RECALL USING A TYPE OF MEMORY"
    RESTORED = "BROUGHT BACK TO A PREVIOUS POINT IN TIME2"
    RECHARGE = "GATHER A SPECIFIC TYPE OF ENERGY OVER A PERIOD OF TIME2 INSIDE A 
    TIMEFRAME"
    PERCENTAGE = "DEFINED AMOUNT OF A SPECIFIC TOTAL AMOUNT"
    PERCENT = "SPECIFIC AMOUNT FROM A WHOLE AMOUNT"
    MINIMUM = "SMALLEST AMOUNT REQUIRED"
    MIN = "MINIMUM"
    CATALOGS = "MORE THAN ONE CATALOG"
    CATALOGUES = "MORE THAN ONE CATALOGUES"
    SWAP = "SWITCH TWO ENTITIES OR OBJECTS"
    SWITCH = "CHANGE THE LOCATION OF"
    METHODS = "MORE THAN ONE METHOD"
    SOME = "MORE THAN ONE PIECE OF SOMETHING BROUGHT INTO A GROUP TO DESCRIBE 
    A PORTION OF ANOTHER GROUP"
    EQUATIONS = "MORE THAN ONE EQUATION"
    VIBRATIONS = "MORE THAN ONE VIBRATION"
    RESONATE = "VIBRATE AT A SPECIFIC RHYTHM OR FREQUENCY"
    VIBRATE = "FLUCTUATE VIBRATIONS AT A SPECIFIC FREQUENCY INTERVAL"
    INTERVALS = "MORE THAN ONE INTERVAL"
    MET = "BEEN IN THE SAME LOCATION AT ONE OR MORE POINTS IN TIME2"
    VISIT = "MEET WITHIN THE SAME LOCATION OR AREA"
    VISITED = "PREVIOUSLY VISIT A LOCATION WITHIN TIME2"
    NAMING = "THE GIVING OF A NAME TO AN ENTITY OR OBJECT3"
    ATTACKS = "MORE THAN ONE ATTACK"
    DEFENSES = "MORE THAN ONE DEFENSE"
    LETTERS = "MORE THAN ONE LETTER"
    SIGNALS = "MORE THAN ONE SIGNAL"
    REQUESTS = "MORE THAN ONE REQUEST"
    ENTRANCES = "MORE THAN ONE ENTRANCE"
    SERVICES = "MORE THAN ONE SERVICE"
    COMPONENTS = "MORE THAN ONE COMPONENT"
    ABSTRACTION = "THE QUALITY OF MANAGING IDEAS"
    PROBLEMS = "MORE THAN ONE PROBLEM"
    OPERATIONS = "MORE THAN ONE OPERATION"
    ABSTRACT_THOUGHT = "USES IDEAS WHICH DO NOT HAVE AN ANY FORM OF MATERIAL 
    EXISTING OR KNOWN"
    COGNITIVE_DEVELOPMENT = "THE CAPACITY FOR ABSTRACT THOUGHT AND IS THE 
    PROGRESS TO ADVANCE THROUGH DIFFERENT FORMS OF THINKING AND 
    UNDERSTANDING"
    COGNITION = "THE PROCESS OF THE MIND TO KNOW AND IS CONNECTED TO 
    JUDGEMENT"
    THINKING = "TO UNDERSTAND THE MEANING METHOD FOR A THOUGHT OR ACTION AND 
    GAIN WISDOM2 FROM THE ACTION OR THOUGHT"
    CROSSCOMPILING = "WHEN AN OPERATING SYSTEM IS DEVELOPED WITHIN ANOTHER 
    OPERATING SYSTEM"
    HYPERTHREADING = "A SYSTEM PROCESS THAT CAN ENABLE THE PROCESSOR TO 
    ACTIVATE TWO OR MORE LISTS OF INSTRUCTIONS AT THE SAME TIME2"
    BINARY_CLASSIFICATION = "WHEN THERE IS TASKS HAVE CATEGORIES TO CATALOG INTO 
    ONLY TWO DISTINCT CLASSES"
    MULTICLASS_CLASSIFICATION = "WHEN THERE IS TASKS THAT HAVE CATEGORIES TO 
    CATALOG INTO MORE THAN TWO"
    HYPERCALL = "A REQUEST BY A USER PROCESS OR OPERATING SYSTEM FOR THE 
    HYPERVISER TO PRODUCE SOME FORM OF ACTION OR EFFECT REQUIRED BY THE 
    OPERATING SYSTEM PROCESSES"
    HYPERCALLS = "MORE THAN ONE HYPERCALL"
    HYPERVISER = "A PROGRAM USED TO ACTIVATE AND MANAGE ONE OR MORE VIRTUAL2 
    DEVICES WITHIN A COMPUTER"
    ROTATIONS_PER_SECOND = "A MEASURE OF THE FREQUENCY OF A ROTATION THAT 
    MEASURES THE ROTATION SPEED OF A SYSTEM OR DEVICE OR TOOL"
    TRANSFERS_PER_SECOND = "THE TOTAL NUMBER OF OPERATIONS TRANSFERING DATA2 
    WITHIN EACH TYPESOFTIMESECOND"
    CHANNEL_MODEL = "A SYSTEM THAT IS MADE TO DESCRIBE HOW THE INPUT IS SENT TO 
    THE OUTPUT"
    BYTES_PER_SECOND = "THE TOTAL NUMBER OF BYTES SENT FOR EACH 
    TYPESOFTIMESECOND WITHIN A SPECIFIC TIMEFRAME"
    BITS_PER_SECOND = "THE TOTAL NUMBER OF BITS SENT FOR EACH 
    TYPESOFTIMESECOND WITHIN A SPECIFIC TIMEFRAME"
    LATENCY = "THE DELAY BEFORE A TRANSFER OF DATA2 BEGINS FOLLOWING AN ORDER 
    FOR ITS TRANSFER" 
    BANDWIDTH = "TO MEASURE THE AMOUNT OF DATA2 THAT IS ABLE TO PASS THROUGH A 
    NETWORK AT A GIVEN TIMEFRAME OR LENGTH OF TIME2"
    THROUGHPUT = "DETERMINED AMOUNT OF HOW MUCH DATA2 CAN TRAVEL THROUGH A 
    SYSTEM OR DIRECTION WITHIN A SPECIFIC PERIOD OF TIME2"
    ADHOC = "A MODE OF WIRELESS COMMUNICATION THAT ALLOWS TWO OR MORE 
    DEVICES WITHIN A SPECIFIC DISTANCE TO TRANSFER DATA2 TO AND FROM EACH 
    DEVICE"
    LAN = "A LOCAL NETWORK FOR COMMUNICATION BETWEEN TWO OR MORE SYSTEMS 
    WITHIN A SPECIFIC AREA OR LOCATION WITHIN AN AREA"
    DSL = "A SPECIFIC LANGUAGE THAT IS SPECIFIED TO A PARTICULAR PROGRAM DOMAIN 
    MADE FOR SOLVING A SPECIFIC CLASS OF PROBLEMS"
    WIRED_CONNECTION = "A CONNECTION USING MATERIAL WIRE TO CONNECT TWO OR 
    MORE DEVICES TOGETHER"
    WIRELESS = "A GROUP OR NETWORK OF MULTIPLE DEVICES THAT SEND AND RECEIVE 
    DATA2 USING FREQUENCIES"
    PREREQUISITE = "A SET OF INSTRUCTIONS REQUIRED AS A LIST OF CONDITIONS THAT 
    MUST BE MET EXISTING FOR SOMETHING TO HAPPEN OR COME INTO EFFECT OR EXIST"
    PREREQUISITES = "MORE THAN ONE PREREQUISITE"
    COMPATIBILITY = "THE RESULT OF IF TWO OR MORE IDEAS OR SYSTEMS ARE 
    COMPATIBLE"
    RESTRICTION = "A LIMITING CONDITION OR MEASURE"
    LIMITATION = "A SPECIFIC TYPE OF SOMETHING THAT IS LIGHTWEIGHT AND CAN BE 
    MOVED WITH LITTLE EFFORT"
    MEMORY_CLOCK = "THE SPEED OF VIRTUAL RAM WITHIN THE 
    GRAPHIC_PROCESSING_UNIT THAT IS DETERMINED BY THE TOTAL NUMBER OF 
    PROCESSES THE SYSTEM CAN PROCESS FROM READING AND WRITING DATA2 FROM 
    MEMORY WITHIN A SINGLE TYPESOFTIMESECOND"
    CORE_CLOCK = "THE SPEED OF THE GRAPHIC_PROCESSING_UNIT CAPABILITIES TO 
    PROCESS INCOMING COMMANDS"
    GRAPHIC_PROCESSING_UNIT = "GRAPHIC PROCESSING UNIT"
    KERNEL = "A COMPUTER PROGRAM AT THE CENTER OF A COMPUTER OPERATING SYSTEM 
    AND HAS CONTROL OVER EVERYTHING INSIDE THE OPERATING SYSTEM"
    ASYMPTOTICANALYSIS = "IS DEFINED AS THE LARGE IDEA THAT MANAGES THE 
    PROBLEMS AND QUESTIONS IN ANALYZING ALGORITHMS"
    TIME_COMPLEXITY = "A POSSIBLE METHOD MADE TO MEASURE THE AMOUNT OF TIME2 
    REQUIRED TO ACTIVATE A CODE"
    SPACE_COMPLEXITY = "USED TO MEASURE THE AMOUNT OF SPACE2 REQUIRED TO 
    ACTIVATE WITH SUCCESS THE FUNCTION OF SPECIFIC CODE"
    AUXILIARY_SPACE = "REFERENCE FOR EXTRA SPACE2 USED IN THE PROGRAM OTHER 
    THAN THE INPUT STRUCTURE"
    ASYMTOTICALNOTATION = "A TOOL THAT CALCULATES THE REQUIRED TIME2 IN TERMS 
    OF INPUT SIZE AND DOES NOT REQUIRE THE ACTIVATION OF THE CODE"
    CLOCKWISE = "TO ROTATE TO THE LEFT"
    COUNTERCLOCKWISE = "TO ROTATE TO THE RIGHT"
    USERSPACE = "ITS PROGRAM CODE OR PARTS OF THE OPERATING SYSTEM THAT IS NOT 
    REQUIRED TO SHARE THE HARDWARE OR ABSTRACT HARDWARE DETAILS"
    DETAILS = "MORE THAN ONE DETAIL"
    DETAIL = "A SPECIFIC FORM OF DATA2 THAT IS SPECIFIC FOR A CATEGORY WITH A LARGE 
    AMOUNT OF DATA WITHIN A SUBCLASS OR SUBTYPE OR CATEGORY"
    CENTERS = "MORE THAN ONE CENTER POINT OR CENTER LOCATION"
    REFERENCES = "CORRELATES TO THE PREVIOUS REFERENCE POINTS AND LOCATIONS"
    REFERS = "REFERENCES A SPECIFIC SET OF INFORMATION OR KNOWLEDGE2"
    BELIEFS = "MORE THAN ONE BELIEF"
    PHILOSOPHY = "A CATEGORY OF LOGIC THAT GIVES MEANING TO SPECIFIC FORMS OF 
    BELIEF AND FORMS STRUCTURE IN ABSTRACT LOGIC AND GIVES MEANING TO SPECIFIC 
    TYPES OF ABSTRACT BELIEFS"
    PHILOSOPHER = "A PERSON WHO STUDIES PHILOSOPHY"
    SHAPES = "MORE THAN ONE SHAPE"
    GEOMETRY = "THE STUDY OF SHAPES AND THE STUDY OF THE LOGIC OF THE 
    MEASUREMENTS OF SHAPES"
    MATH = "A CATEGORY THAT HOLDS THE EQUATIONS AND MEASUREMENT OF MANY 
    EQUATION OPERATIONS FOR ALL MEASUREMENT SYSTEMS"
    TYPEZERO = "A TYPE THAT HAS AN ATTRIBUTE OF ZERO"
    YESNOLOGIC = "THE VALUE OF CHOOSING YES OR NO AS A RESULT OF MATH TO 
    DETERMINE ANOTHER RESULT"
    GAMEWORLD = "THE EMPTY SPACE THAT IS FILLED WITH OBJECTS3 TO MAKE SOMETHING 
    FROM"
    LOCALSPACE = "THE SPACE THAT CONTAINS LOCAL DATA2 WITHIN THE GAMEWORLD"
    PRIVATELOCALDATA2 = "DATA2 WITHIN A PRIVATE PARAMETER THAT IS LOCAL"
    LOCALPRIVATEDATA2 = "DATA2 WITHIN A LOCAL PARAMETER THAT IS PRIVATE"
    LOCALLOCATION = "A LOCATION THAT IS LOCAL"
    LOCALPARAMETER = "A PARAMETER THAT IS WITHIN A LOCAL LOCATION"
    PRIVATEPERSONALDATA2 = "DATA2 THAT IS PERSONAL WITHIN A PRIVATE PARAMETER"
    PERSONALPRIVATEDATA2 = "DATA2 THAT IS PRIVATE WITHIN A PERSONAL LOCATION"
    CUSTOMTERM = "A CUSTOM DEFINED TERM MADE BY HEADADMINZACK"
    HYPERSPEED = "A SPEED AT WHICH REACHES A STAGE OF HYPER FOR A SPECIFIC 
    WAVELENGTH" 
    NOUN = "A PART OF LANGUAGE RULES THAT DETERMINES IF A WORD IS A PERSON OR 
    PLACE2 OR THING OR IDEA"
    VERB2 = "A PART OF LANGUAGE RULES THAT IS USED FOR EXPRESSING ACTIONS FROM 
    WORDS OR SOMETHING FORMED FROM CAUSE AND OR EFFECT"
    PLACE2 = "A SPECIFIC LOCATION"
    THING = "A IDEA OR TOPIC THAT DEFINES SOMETHING"
    IDEA2 = "A DESCRIBED SPECIFIED MEANING OR TOPIC THAT HAS APPLIED THOUGHTS TO 
    THE MEANING AND IS MADE FROM CREATIVITY AND OR IMAGINATION2"
    PRIVATEPUBLICLOCATION = "A PUBLIC LOCATION THAT IS PRIVATE"
    PERSONALPUBLICLOCATION = "A PUBLIC LOCATION THAT IS PERSONAL TO THE USER"
    PRIVATEPERSONALLOCALLOCATION = "A PERSONAL LOCATION THAT IS BOTH PRIVATE 
    AND LOCAL WITHIN A SPECIFIC LOCATION OR DOMAIN"
    TOPICS = "MORE THAN ONE TOPIC"
    NETWORKPARAMETER = "A SPECIFIC NETWORK APPLIED TO A PARAMETER FOR DOMAIN 
    SPECIFIC TOPICS"
    NETWORKSETTING = "A SETTING APPLIED TO A NETWORK"
    DOMAINLOCATION = "THE SPECIFIC LOCATION OF A DOMAIN"
    EVOLVEBELIEFS = "A COMMAND USED TO APPLY EVOLUTION TO A SPECIFIC BELIEF OR 
    BELIEFS"
    MOTIVATIONENHANCER = "A SPECIFIC PATH MEANT TO INCREASE AND ENHANCE 
    MOTIVATION OF A SPECIFIC ENTITY"
    HYPERSCAN = "THE FUNCTION THAT IS USED TO APPLY A STATE WHEN SCANNING AT 
    HYPER RATE IS POSSIBLE"
    LOCALENTITY = "AN ENTITY THAT IS LOCAL"
    PRIVATEENTITY = "AN ENTITY THAT IS PRIVATE"
    PERSONALENTITY = "AN ENTITY THAT IS PERSONAL"
    ADHOCNETWORK = "A NETWORK MADE WITH A ADHOC SETTING"
    LIMITEDREACH = "THE APPLYING OF A LIMITED REACH TO AN MEASUREMENT OR 
    MEASUREMENTS"
    SPECIFICLENGTH = "THE SPECIFIC LENGTH OF A DIMENSIONAL LINE"
    GAMEENGINEDISPLAY = "THIS DETERMINES THE RESOLUTION OF THE SCREEN AND THE 
    HIGHER THE RESOLUTION THEN THE MORE DETAILED THE VISUAL REPRESENTATION IS"
    TRANSCREATIONSTONE = "A OBJECT3 MADE OF TRANSCREATION SUBSTANCE THAT IS 
    ABLE TO CONVERT OTHER MATERIALS TO A DIFFERENT MATERIAL SUBSTANCE"
    STATDEBUFFONE = "AN EFFECT THAT CAN WEAKEN THE DEFENSE OF A TARGET WHEN 
    SOMETHING ATTACKS THE SYSTEM"
    STATDEBUFFTWO = "AN EFFECT THAT CAN TARGET SPECIFIC ATTACKS AND FORM AN 
    COUNTERACTION AND REVERSE THE ATTACK ON ITSELF"
    COUNTERACTIONREACTION = "A COMMAND THAT ALLOWS AN ATTACK TO BE GIVEN A 
    COUNTERACTION TO A SPECIFIC COUNTER OF AN ACTION"
    SWITCHPLACES = "A COMMAND THAT ALLOWS THE USER TO SWITCH PLACES ALLOWING 
    THE PLAYER TO FORM RECOVERY"
    READTYPE = "A COMMAND TO READ A SPECIFIC TYPE OF SOMETHING"
    HYPERREACTOR = "A STATE OF VOID SPACE FORMING A HYPER REACTION WITHIN THE 
    VOID ITSELF TO FORM AND CREATE NEW SAFE ENERGY IN A PRESSURIZED CONDENSED 
    FORM"
    GENRESPECIFIC = "A TOPIC THAT IS ONLY EXISTING WITHIN A SPECIFIC GENRE"
    GAMESETTINGONE = "THIS MANAGES THE LEVEL OF DETAIL FOR TEXTURES USED WITHIN 
    THE SPECIFIC SYSTEM AND A HIGHER TEXTURE RESOLUTION WILL MAKE TEXTURES 
    APPEAR MORE DETAILED"
    GAMESETTINGTWO = "THIS CONTROLS THE DISTANCE AN OBJECT3 IS VISIBLE FROM 
    WITHIN A SYSTEM AND A LARGER LONG DISTANCE DOES ALLOW MORE OBJECTS TO BE 
    SEEN WITH VIVID DETAIL"
    GAMESETTINGTHREE = "THIS MANAGES THE EXTENT OF THE VISIBLE SPACE WITHIN THE 
    GAMEWORLD THAT IS SEEN ON THE SCREEN"
    GAMESETTINGFOUR = "THIS DETERMINES THE QUALITY AND RESOLUTION OF A SPECIFIC 
    AMOUNT OF SHADE AND HIGHER QUALITY SHADE"
    BECOMING = "FORMING INTO FROM SOMETHING PREVIOUSLY BEFORE TO PRESENT"
    OBJECTS3 = "MORE THAN ONE OBJECT3"
    STUDIES = "MORE THAN ONE STUDY"
    FILLED = "GIVEN DATA2 TO AN IMPORT INTO AN EMPTY SPACE THAT CAN FILL"
    FILL = "IMPORT DATA2 OR LOGIC WITHIN A SPECIFIC SPACE OR LOCATION"
    LISTS = "MORE THAN ONE LIST"
    SERVICE = "REQUEST OR TASK THAT SOMEONE OR SOMETHING CAN REQUEST 
    COMPLETE"
    DELAY = "A TIME THAT HAPPENS BEFORE A CERTAIN TIMEFRAME BEFORE ACTIVATION 
    HAPPENS"
    BEGINS = "STARTS WITH"
    STARTS = "BEGINS WITH"
    INDIVIDUALMOTORSKILLS = "SPECIFIC MOTOR SKILLS THAT ARE UNIQUE TO THE 
    INDIVIDUAL PERSON" 
    INDIVIDUALPRECOGNITION = "THE UNIQUE ABILITY TO PERCEIVE EVENTS BEFORE THEY 
    HAPPEN THAT IS SPECIFIC TO AN INDIVIDUAL PERSON" 
    INDIVIDUALAURA = "THE RESONATING AURA THAT CONNECTS TO A SPECIFIC 
    INDIVIDUAL PERSON" 
    INDIVIDUALCHAKRA = "THE CHAKRA THAT CONNECTS TO A SPECIFIC INDIVIDUAL 
    PERSON" 
    INDIVIDUALPERSONALITY = "THE PERSONALITY THAT A SPECIFIC PERSON HAS" 
    INDIVIDUALMINDSET = "THE MINDSET THAT IS UNIQUE TO EACH SPECIFIC PERSON" 
    MINDSET = "THE SPECIFIC DIRECTION THAT THE MIND IS FOLLOWING INCLUDING THE 
    ENERGY THAT THE MIND RELEASES AND GATHERS TO MAKE DECISIONS" 
    CONNECTS = "MAKES A DECISION TO CONNECT TWO PIECES OF INFORMATION 
    TOGETHER OR TWO OR MORE ORIGIN POINTS OR EVENTS" 
    INDIVIDUAL = "RELATING TO A SPECIFIC PERSON THAT IS UNIQUE AND HAS ITS OWN 
    LABEL AND CLASS" 
    RESONATING = "AN ACTION THAT IS CURRENTLY PROCESSING A RESONANCE" 
    GATHERS = "ABSORBS AND BRINGS INWARDS" 
    ABSORBS = "GATHERS AND OBTAINS"
    BRINGS = "SENDS OUT TO BE RECEIVED" 
    LABEL = "A SPECIFIC CLASS OR TYPE GIVEN TO AN ENTITY"
    DECISIONS = "MORE THAN ONE DECISION" 
    RESONANCE = "THE PROCESS OF VIBRATING SPECIFIC SOUND FREQUENCIES AT A 
    SPECIFIC RATE OF VIBRATION" 
    INCLUDING = "MAKING A DECISION TO INCLUDE WITHIN A SPECIFIC GROUP OR 
    CATEGORY" 
    VIBRATING = "FORMING FRICTION BETWEEN TWO SPECIFIC SOURCES WHILE CREATING 
    RESONANCE BETWEEN TWO SPECIFIC SOUND WAVES OR MORE"
    INDIVIDUALEQ = "THE INDIVIDUAL READING INTERPRETATION OF AN ENTITIES EMOTIONS 
    AND THE USE OF THOSE EMOTIONS IN QUALITY TO MAKE A CORRECT SPECIFIC 
    SUGGESTION ON HOW AN EMOTION OR SERIES OF EMOTIONS IS INTERPRETED" 
    INDIVIDUALIQ = "THE INDIVIDUAL INTELLIGENCE2 THAT A SPECIFIC PERSON OR ENTITY 
    HAS THAT IS MEASURED AND DETERMINED HOW INTELLIGENT SOMEONE IS" 
    INDIVIDUALKNOWLEDGE = "THE SPECIFIC AMOUNT OF KNOWLEDGE2 AND ITS 
    CATEGORIZED VALUES THAT MAKES AN INDIVIDUAL PERSON OR SOMEONE UNIQUE" 
    INDIVIDUALINTELLIGENCE = "THE INDIVIDUAL INTELLIGENCE2 THAT A PERSON HAS THAT 
    IS MEASURED BY THE INTELLIGENCE2 QUOTIENT" 
    INDIVIDUALWISDOM = "THE WISDOM2 A SPECIFIC INDIVIDUAL HAS THAT MAKES THE 
    PERSON UNIQUE FOR THE WISE CHOICES CHOSEN" 
    INDIVIDUALNATURE = "THE DETERMINED VALUES THAT DETERMINE HOW A PERSON CAN 
    BEHAVE OR ACT BASED ON A SET OF CHARACTERISTIC QUALITIES MEASURED IN A 
    PERSON BY CLASSIFIED DATA2 FOR EACH INDIVIDUAL PERSON" 
    INDIVIDUALMINDSET = "RELATING OR CONCERNING A SPECIFIC MINDSET THAT A 
    PERSON HAS OR SOMEONE HAS" 
    INDIVIDUALCHARACTERISTICS = "THIS SPECIFIC CLASSIFIED DATA2 THAT MAKES A 
    PERSON UNIQUE IN VALUE" 
    CORRECT = "THE ANSWER THAT IS TRUE" 
    WISE = "A SELECTION OF CHOICES THAT COMES FROM EXPERIENCE RATHER THAN 
    DECISION MAKING" 
    CLASSIFIED = "INFORMATION THAT ONLY CORRELATES TO A SPECIFIC SET OF CATEGORY 
    OR GENRE" 
    BEHAVE = "A FORM OF ACTION THAT IS DETERMINED BY CORRECT AND INCORRECT 
    DECISION MAKING" 
    CATEGORIZED = "ORGANIZED AND SORTED TO BELONG TO A SPECIFIC CATEGORY OR 
    GROUP OF CATEGORIES" 
    INTERPRETED = "SCANNED TO BE UNDERSTOOD BY A SET OF LOGIC AND REASON THAT 
    DETERMINES HOW INFORMATION IS COMPREHENDED" 
    INTERPRETATION = "A FORM OF LOGIC THAT IS USED TO UNDERSTAND A CONVERSATION 
    OR ARGUMENT BETWEEN TWO OR MORE ENTITIES" 
    INTELLIGENT = "UNDERSTOOD TO MAKE A SET OF DECISIONS INTELLIGENTLY BY USING 
    INTELLIGENT DECISION MAKING" 
    SUGGESTION = "A TOPIC OR SUBJECT THAT IS RECOMMENDED OR SENT TO BE EITHER 
    ACCEPTED OR DENIED" 
    SORT = "TO FILTER AND SPECIFY A SPECIFIC SET OF CATEGORIES FOR SOMETHING" 
    ORGANIZE = "TO SELECT THE LOCATION OR CATEGORY FOR MANY SPECIFIC THINGS OR 
    IDEAS" 
    BELONG = "LIST WITHIN A SPECIFIC AREA OR GROUP" 
    REASON = "SELECTED ANSWER TO WHY SOMETHING CAME INTO AN EFFECT" 
    COMPREHENDED = "GAINED THE ABILITY TO COMPREHEND" 
    CONVERSATION = "A SERIES OF SIGNALS THAT COMMUNICATE TO AND FROM SPECIFIC 
    SOURCES AND SEND AND RECEIVE INFORMATION TWO AND FROM TWO OR MORE 
    DISTINCT LOCATIONS" 
    ACCEPTED = "APPROVED AS CORRECT" 
    DENIED = "NOT ACCEPTED" 
    SCANNED = "INFORMATION THAT HAS BEEN PROCESSED" 
    INCORRECT = "NOT CORRECT" 
    RATHER = "DECIDE TO DO SOMETHING AS AN ALTERNATE CHOICE" 
    ARGUMENT = "COMMUNICATION BETWEEN TWO OR MORE SOURCES THAT ARGUE" 
    RECOMMENDED = "DETERMINED AS A SOLUTION TO RECOMMEND" 
    SELECTED = "HAVE CHOSEN TO HAPPEN" 
    RECOMMEND = "TO SUGGEST SOMETHING" 
    ARGUE = "TO COMMUNICATE BACK AND FORTH TO DECIDE SOMETHING THAT IS TRUE 
    FROM A SPECIFIC TOPIC OR SPECIFIC AMOUNT OF TOPICS" 
    COMMUNICATE = "TO SEND INFORMATION TO ANOTHER SOURCE AND HAVE 
    INFORMATION RETURN BACK TO THE SOURCE BY CREATING A RESPONSE" 
    FILTER = "TO SORT AND ORGANIZE SPECIFIC SET OF INFORMATION TO ITS CORRECT 
    LOCATION OR CATEGORY THAT IS DETERMINED BY A CLASS OR A TYPE" 
    SPECIFY = "TO ANSWER WITH SPECIFIC CLARITY REGARDING A SUBJECT OR TOPIC" 
    CAME = "ARRIVED OR REACHED A DESTINATION AT A SPECIFIC TIMEFRAME" 
    CLARITY = "TO COMPREHEND AND UNDERSTAND AN UNDERSTANDING OF WHAT IS 
    COMMUNICATED BETWEEN TWO OR MORE SOURCES WHILE ALL SOURCES 
    UNDERSTAND THE INFORMATION THAT IS EXPLAINED" 
    REGARDING = "TO CORRELATE INFORMATION FROM WITHIN A SPECIFIC TOPIC AND OR 
    CATEGORY AND OR FIELD AND OR GENRE" 
    ARRIVED = "REACH THE END DESTINATION" 
    REACHED = "ARRIVED AT" 
    CORRELATE = "INTERCONNECT AND INTERPRET HOW INFORMATION IS ACCEPTED" 
    INTERPRET = "GIVE AN INSIGHTFUL SUGGESTION FROM A RESPONSE OR ANSWER OR 
    QUESTION" 
    INTERCONNECT = "CONNECT BETWEEN MANY SOURCES OR PATHS" 
    INSIGHTFUL = "COMPLETE AND FILLED WITH MEANING OR SPECIFIC INFORMATION 
    MADE OF GREATER QUALITY THAN WHAT IS KNOWN" 
    QUOTIENT = "A SPECIFIC MEASUREMENT USED TO CLASSIFY AND MEASURE THE EXTENT 
    OF HOW MUCH INFORMATION IS PROVIDED AND PROVIDE AN MEASUREMENT OF SOME 
    SORT AFTER THE LOGIC IS GIVEN"
    ORGANIZED = "SPECIFIC INFORMATION THAT IS ORGANIZED INTO MANY CATEGORIES OR 
    GROUPS AND FILTERED BY CLASS OR TOPIC"
    SORTED = "FILTERED AND ORGANIZED FROM SPECIFIC TOPICS OR CATEGORIES"
    SUGGEST = "RECOMMEND A SPECIFIC CATEGORY OR TOPIC TO CHOOSE"
    CLASSIFY = "GIVEN A SPECIFIC CLASS OR CATEGORY THAT IS CHOSEN AS"
    PROVIDE = "GIVE SOMETHING AS A RESULT OF SOMETHING ELSE OR BY CHOICE WITH 
    NO REASON"
    PROVIDED = "GIVEN AS A CHOICE OR OPTION TO PROVIDE"
    FILTERED = "SORTED AND SENT TO THE CORRECT LOCATION AFTER SORTED"
    COMPREHENDING = "ACKNOWLEDGING2 AND UNDERSTANDING THE ABILITY TO 
    COMPREHEND"
    HOPE = "EVEN IN THE CIRCUMSTANCE WHEN SOMETHING IS CONSIDERED IMPOSSIBLE 
    THEN IT CAN BE SHOWN THAT SOMETHING CAN HAPPEN WITH THE CHANCE THAT A 
    POSSIBILITY CAN FORM" 
    SHOULD = "IT CAN BE CERTAIN THAT THE POSSIBILITY IS THERE FOR SOMETHING TO 
    HAPPEN"
    BOND = "A FORM OF BINDING THAT CONNECTS TWO FRIENDS TOGETHER AND THEIR 
    EMOTIONS"
    UNFAMILIAR = "NOT KNOWN OR RECOGNIZED"
    UNTROUBLED = "NOT FEELING AND OR SHOWING OR AFFECTED BY ANXIETY OR 
    PROBLEMS"
    ANXIETY = "IS THE MIND AND OR BODY AND ITS REACTION TO STRESSFUL AND OR 
    DANGEROUS AND OR UNFAMILIAR SITUATIONS AND IS THE SENSE OF UNEASINESS AND 
    OR DISTRESS AND OR DREAD YOU FEEL BEFORE A SIGNIFICANT EVENT"
    CONTENTMENT = "A STATE OF HAPPINESS AND SATISFACTION"
    SERENE = "CALM PEACEFUL AND UNTROUBLED AND OR TRANQUIL"
    DISCONTENT = "NOT ABLE TO RECOGNIZE CONTENTMENT"
    CALM2 = "NOT SHOWING OR FEELING NERVOUSNESS AND ANGER AND OR OTHER 
    STRONG EMOTIONS"
    UNEASE = "ANXIETY OR DISCONTENT"
    ANXIOUS = "WANTING SOMETHING VERY MUCH THAT COMES WITH A FEELING OF 
    UNEASE"
    NERVOUS = "ANXIOUS OR APPREHENSIVE"
    APPREHENSIVE = "ANXIOUS THAT SOMETHING NOT KNOWN WILL HAPPEN THAT COMES 
    WITH UNEASE"
    GRACE = "IS CONSIDERED ACCEPTANCE AND INCLUDES GIVING AND IS FREE IN THE 
    SENSE THAT SOMETHING DONE OR GIVEN IN GRACE AND IS DONE WITHOUT A REQUEST 
    OR REQUIRE FOR THE POSSIBILITY TO RECEIVE ANYTHING IN RETURN"
    DISTURBANCE = "THE INTERRUPTION OF A SETTLED AND PEACEFUL CONDITION"
    PEACE = "FREEDOM FROM DISTURBANCE AND THE SHOWING OF TRANQUILITY IN 
    EFFECT"
    TRANQUILITY = "A STATE OF PEACE OR CALM"
    INTERUPT = "END THE CONTINUOUS PROGRESS OF"
    INTERRUPTING = "CURRENTLY HAPPENING TO INTERUPT"
    INTERRUPTED = "PREVIOUS ACTION TO INTERUPT"
    INTERRUPTION = "THE ACTION OF INTERRUPTING OR BEING INTERRUPTED"
    RESOLVE = "COME TO A CONCLUSION ABOUT A TOPIC OR SUBJECT"
    SETTLED = "RESOLVE OR REACH AN AGREEMENT ABOUT"
    CONCLUSION = "TO ARRIVE AT AN END RESULT OR DESTINATION"
    PROPERTIES = "MORE THAN ONE PROPERTY"
    CARDINAL_MATH = "IS A CATEGORIZED GROUP OF THE NATURAL NUMBERS USED TO 
    MEASURE THE CARDINALITY SIZE OF SETS"
    CARDINALITY = "IS A MEASURE OF THE NUMBER OF ELEMENTS OF THE SET"
    ORDINAL_SCALE = "A VARIABLE MEASUREMENT SCALE USED TO INTERPRET THE ORDER 
    OF VARIABLES AND NOT THE DIFFERENCE BETWEEN EACH OF THE VARIABLES"
    CARTESIAN_GRID = "IS A COORDINATE SYSTEM THAT SPECIFIES EACH POINT IN A 
    UNIQUE GROUP OF COORDINATES"
    ABSDISC_MATH = "IS THE STUDY OF CALCULATION STRUCTURES THAT CAN BE 
    CONSIDERED ABSTRACT RATHER THAN CONTINUOUS"
    ABS_DISC_NUMBER_THEORY = "IS A STUDY WITH THE FOCUS ON PROPERTIES OF 
    NUMBERS"
    ABS_UNIQUE_CARDIANAL_SPACE_SYSTEM = "IS A SYSTEM IN WHICH A FUNCTION CAN 
    DESCRIBE THE TIME2 TIME_DEPENDENT OF A POINT IN AN SPACE2 SURROUNDING AN 
    OBJECT3"
    AMBIENT_SPACE = "THE SPACE2 SURROUNDING AN OBJECT3"
    ERR_SEQUENCE = "A FUNCTION DEFINED ON AN INTERVAL OF THE NUMBERS IS CALLED 
    A SEQUENCE"
    TIME_DEPENDENT = "DETERMINED BY THE VALUE OF A VARIABLE REPRESENTING TIME2"
    DESCRIBING_TIME = "TIME2 IS THE CONTINUOUS SEQUENCE OF EXISTENCE AND 
    EVENTS THAT COMES INTO PLACE IN AN NOT ABLE TO BE CHANGED SEQUENCE OF 
    EVENTS FROM THE PAST THROUGH THE PRESENT INTO THE FUTURE"
    ENUMERATION = "COMPLETE AND ORGANIZED LIST OF ALL THE DATA2 IN A 
    COLLECTION"
    BODY_OF_KNOWLEDGE = "IS THE COMPLETE SET OF IDEAS AND TERMS AND EFFECT 
    THAT MAKE UP A DOMAIN"
    MANUAL_OVERRIDE = "A METHOD WHERE CONTROL IS TAKEN FROM AN AUTOMATICALLY 
    FUNCTIONING SYSTEM AND GIVEN COMPLETE CONTROL TO THE CREATOR"
    REROUTE2 = "SEND SOMEONE OR SOMETHING BY OR TO A DIFFERENT ROUTE"
    SEQUENCE = "A SET OF SIMILAR EVENTS AND OR MOVEMENTS AND OR THINGS THAT 
    FOLLOW EACH OTHER IN A PARTICULAR SERIES OF EVENTS"
    QTE4A1 = "A SEQUENCE IS ABLE TO BE A FINITE SEQUENCE FROM A DATA2 SOURCE OR 
    AN INFINITE SEQUENCE"
    R2E4A1 = "REFERS TO HOW A GROUP OF DATA2 OF A SPECIFIC SERIES OF DATA IS USED 
    TO COMPLETE A CERTAIN GOAL"
    UNFAMILIAR = "NOT KNOWN OR RECOGNIZED"
    UNTROUBLED = "NOT FEELING AND OR SHOWING OR AFFECTED BY ANXIETY OR 
    PROBLEMS"
    ANXIETY = "IS THE MIND AND OR BODY AND ITS REACTION TO STRESSFUL AND OR 
    DANGEROUS AND OR UNFAMILIAR SITUATIONS AND IS THE SENSE OF UNEASINESS AND 
    OR DISTRESS AND OR DREAD YOU FEEL BEFORE A SIGNIFICANT EVENT"
    CONTENTMENT = "A STATE OF HAPPINESS AND SATISFACTION"
    SERENE = "CALM PEACEFUL AND UNTROUBLED AND OR TRANQUIL"
    DISCONTENT = "NOT ABLE TO RECOGNIZE CONTENTMENT"
    CALM2 = "NOT SHOWING OR FEELING NERVOUSNESS AND ANGER AND OR OTHER 
    STRONG EMOTIONS"
    UNEASE = "ANXIETY OR DISCONTENT"
    ANXIOUS = "WANTING SOMETHING VERY MUCH THAT COMES WITH A FEELING OF 
    UNEASE"
    NERVOUS = "ANXIOUS OR APPREHENSIVE"
    APPREHENSIVE = "ANXIOUS THAT SOMETHING NOT KNOWN WILL HAPPEN THAT COMES 
    WITH UNEASE"
    GRACE = "IS CONSIDERED ACCEPTANCE AND INCLUDES GIVING AND IS FREE IN THE 
    SENSE THAT SOMETHING DONE OR GIVEN IN GRACE AND IS DONE WITHOUT A REQUEST 
    OR REQUIRE FOR THE POSSIBILITY TO RECEIVE ANYTHING IN RETURN"
    DISTURBANCE = "THE INTERRUPTION OF A SETTLED AND PEACEFUL CONDITION"
    PEACE = "FREEDOM FROM DISTURBANCE AND THE SHOWING OF TRANQUILITY IN 
    EFFECT"
    TRANQUILITY = "A STATE OF PEACE OR CALM"
    INTERUPT = "END THE CONTINUOUS PROGRESS OF"
    INTERRUPTING = "CURRENTLY HAPPENING TO INTERUPT"
    INTERRUPTED = "PREVIOUS ACTION TO INTERUPT"
    INTERRUPTION = "THE ACTION OF INTERRUPTING OR BECOMING INTERRUPTED"
    RESOLVE = "COME TO A CONCLUSION ABOUT A TOPIC OR SUBJECT"
    SETTLED = "RESOLVE OR REACH AN AGREEMENT ABOUT"
    CONCLUSION = "TO ARRIVE AT AN END RESULT OR DESTINATION"
    HAPPY = "FEELING OR SHOWING PLEASURE OR CONTENTMENT"
    CONFIDENT = "FEELING OR SHOWING CONFIDENCE IN ONESELF"
    OPTIMISTIC = "HOPEFUL AND CONFIDENT ABOUT THE FUTURE"
    CHASTITY = "SETTING LIMITS UPON TO NOT DO SOMETHING FOR A SPECIFIC AMOUNT OF 
    TIME2 WHILE NOT HAVING SOMETHING SPECIFIC"
    CHASTE = "SHOWING RESTRAINT UPON SOMETHING SPECIFIC FOR A LIMITED AMOUNT 
    OF TIME2"
    VIGILANCE = "THE ACTION OR STATE OF STAYING ON CAREFUL WATCH FOR POSSIBLE 
    DANGER OR INCREASE IN DIFFICULTY"
    DILIGENCE = "CAREFUL AND PERSISTANT WORK OR EFFORT"
    PATIENCE = "THE CAPACITY TO ACCEPT OR TOLERATE DELAY AND TROUBLE AND OR 
    SUFFERING WITHOUT GETTING ANGRY OR UPSET"
    KINDNESS = "THE QUALITY OF BECOMING FRIENDLY AND OR GENEROUS AND OR 
    CONSIDERATE"
    GENEROSITY = "SHOWING A READINESS TO GIVE MORE OF SOMETHING SUCH2 AS 
    MONEY OR TIME2 OR THAN WHAT IS STRICTLY NECESSARY OR EXPECTED"
    HUMILITY = "A MODEST OR LITTLE VIEW OF ONES OWN IMPORTANCE"
    HUMBLENESS = "LOWER SOMEONE IN DIGNITY OR IMPORTANCE"
    CLARITY = "CLEARNESS OR LUCIDITY AS TO PERCEPTION OR UNDERSTANDING AND OR 
    FREEDOM FROM INDISTINCTNESS OR AMBIGUITY"
    MODESTY = "THE QUALITY OF BECOMING RELATIVELY MODERATE AND OR LIMITED AND 
    OR SMALL IN AMOUNT AND OR RATE AND OR LEVEL"
    TEMPERANCE = "THE QUALITY OF MODERATION OR RESTRAINT UPON A SPECIFIC 
    ENTITY"
    CHARITY = "HELPING SOMEONE OR THE ACTION OF GIVING MONEY VOLUNTARILY TO 
    THOSE IN NEED"
    CAREFUL = "MAKING SURE OF AVOIDING POTENTIAL DANGER AND OR MISHAP AND OR 
    OR HARM OR TO BE CAUTIOUS"
    MODEST = "UNASSUMING OR MODERATE IN THE ESTIMATION OF OWNED ABILITIES OR 
    ACHIEVEMENTS"
    ACHIEVEMENTS = "MORE THAN ONE ACHIEVEMENT"
    BRAVERY = "COURAGEOUS BEHAVIOR OR CHARACTER"
    COURAGEOUS = "NOT DETERRED BY DANGER OR PAIN"
    BRAVE = "ENDURE OR BECOME APART OF AN UNPLEASANT CONDITIONS OR BEHAVIOR 
    WITHOUT SHOWING FEAR"
    COURAGE2 = "THE ABILITY TO DO SOMETHING THAT FRIGHTENS ONE"
    LUCIDITY = "THE ABILITY TO THINK CLEARLY ESPECIALLY IN INTERVALS BETWEEN POINTS 
    OF TIME2"
    AVOID = "NOT GO TO OR THROUGH"
    GIFT_OF_WISDOM = "TO CORRESPOND TO THE VIRTUE OF CHARITY"
    GIFT_OF_UNDERSTANDING_AND_KNOWLEDGE = "TO CORRESPOND TO THE VIRTUE OF 
    FAITH"
    GIFT_OF_COUNSEL = "TO CORRESPOND TO THE VIRTUE OF PRUDENCE"
    GIFT_OF_FORTITUDE = "TO CORRESPOND TO THE VIRTUE OF COURAGE"
    PLEASURE = "A FEELING OF HAPPY SATISFACTION AND ENJOYMENT"
    SATISFACTION = "FULFILLMENT OF ONES WISHES AND OR EXPECTATIONS AND OR 
    NEEDS AND OR THE PLEASURE DERIVED FROM THIS"
    VIRTUES = "BEHAVIOR SHOWING HIGH MORAL STANDARDS"
    CORRESPOND = "HAVE A CLOSE SIMILARITY AND OR MATCH OR AGREE ALMOST 
    EXACTLY"
    CLOSE = "A SMALL DISTANCE AWAY OR APART IN SPACE2 OR TIME2"
    CONFIDENCE = "THE FEELING OR BELIEF THAT ONE CAN RELY ON SOMEONE OR 
    SOMETHING"
    RELY = "DEPEND ON WITH COMPLETE TRUST OR CONFIDENCE"
    DEPEND = "NEED OR REQUIRE FOR SUPPORT"
    SUPPORT = "ASSISTANCE TO GET OR GIVE HELP OR REQUIRE HELP"
    HELP = "MAKE IT POSSIBLE WITH LESS DIFFICULTY FOR SOMEONE TO DO SOMETHING BY 
    OFFERING SERVICES OR RESOURCES"
    RESOURCE = "A PRODUCT OR SUPPLY OF MONEY AND OR MATERIALS AND OR AND 
    OTHER ASSETS THAT CAN BE ACCESSED BY A PERSON OR GROUP IN ORDER TO 
    FUNCTION EFFECTIVELY"
    EFFECTIVELY = "IN SUCH2 A MANNER AS TO ACHIEVE A DESIRED RESULT"
    SIMILARITY = "SOMETHING SIMILAR OR CLOSE IN RELATION TO A SPECIFIC TOPIC OR 
    SUBJECT"
    SIMILARITIES = "MORE THAN ONE SIMILARITY"
    BEHAVIOR = "THE WAY IN WHICH ONE SHALL BEHAVE AND OR DISPLAY ACTIONS FROM 
    STYLE OR PRESENTATION OF SELF DEMEANOR"
    DEMEANOR = "THE OUTWARD BEHAVIOR OR BEARING"
    BEARING = "THE WAY IN WHICH SOMEONE OR SOMETHING SHALL STAY IN POSITION 
    WITHOUT MOVING OR WHILE MOVING"
    OUTWARD = "OF AND OR ON AND OR FROM THE OUTSIDE"
    WANTING = "A YEARNING DESIRE FOR SOMETHING THAT IS NOT CURRENTLY OWNED"
    PRODUCT = "AN OBJECT3 OR SERVICE THAT CAN BE OBTAINED OR POSSESSED BY 
    SOMEONE"
    PRODUCTS = "MORE THAN ONE PRODUCT"
    YEARN = "HAVE AN INTENSE FEELING OF LONGING FOR SOMETHING"
    LONGING = "A YEARNING DESIRE"
    YEARNING = "A FEELING OF INTENSE LONGING FOR SOMETHING"
    DESIRE = "A STRONG FEELING OF WANTING TO HAVE SOMETHING OR WISHING FOR 
    SOMETHING TO HAPPEN"
    WISHING = "MAKING AN ACTION FOR EXPRESSING A DESIRE FOR SOMETHING OR AN 
    IDEA2"
    SUPPLY = "MAKE SOMETHING NEEDED OR REQUESTED OR ASKED FOR TO SOMEONE"
    NEEDED = "REQUIRED TO COMPLETE SOMETHING"
    REQUESTED = "SOMETHING THAT HAS BEEN ASKED FOR"
    ASSISTANCE = "THE ACTION OF HELPING SOMEONE WITH A JOB OR TASK"
    HELPING = "DOING SOMETHING FOR SOMEONE THAT CAN OR DOES REQUIRE HELP"
    SUCH2 = "TO BE AS CLOSE TO AN HIGH STATUS2"
    DESIRED = "STRONGLY WISHED FOR OR INTENDED"
    WISHED = "EXPRESS A DESIRE FOR THE SUCCESS OR GOOD_FORTUNE OF SOMEONE"
    GOOD_FORTUNE = "AN AUSPICIOUS STATE RESULTING FROM FAVORABLE OUTCOMES 
    GOOD_LUCK AND OR LUCKINESS"
    GOOD_LUCK = "USED TO EXPRESS WISHES FOR SUCCESS"
    EXPRESS = "CONVEY A THOUGHT OR FEELING IN WORDS OR BY GESTURES AND 
    CONDUCT2"
    CONDUCT2 = "THE MANNER IN WHICH A PERSON USES TO BEHAVE"
    CONVEY = "MAKE AN IDEA AND OR IMPRESSION AND OR FEELING KNOWN OR 
    UNDERSTANDABLE TO SOMEONE"
    UNDERSTANDABLE = "ABLE TO BE UNDERSTOOD"
    IMPRESSION = "AN IDEA AND OR FEELING AND OR OPINION ABOUT SOMETHING OR 
    SOMEONE"
    OPINION = "THE BELIEF OR PERSPECTIVE OF A PARTICULAR SUBJECT AND OR TOPIC AND 
    OR THING AND OR IDEA"
    PERSPECTIVE = "A PARTICULAR ATTITUDE OR APPROACH TOWARD SOMETHING OR WAY 
    OF REGARDING SOMETHING"
    APPROACH = "A WAY OF HAVING SOMETHING COMPLETE OR VIEWED AS A WAY TO 
    ENVISION OR PERCEIVE SOMETHING OR SOMEPLACE OR SOMEONE"
    SETTLED2 = "RESOLVED OR REACHED AGREEMENT ABOUT"
    ATTITUDE = "A SETTLED2 WAY OF THINKING OR FEELING ABOUT SOMEONE OR 
    SOMETHING"
    RESOLVED = "SOLVED AND COMPLETED A PROBLEM"
    SOLVE = "FIND AN ANSWER TO A PROBLEM"
    SOLVED = "SOMETHING THAT HAS BEEN COMPLETED AS A FOUND ANSWER TO A 
    PROBLEM"
    COMPLETED = "FINISHED MAKING OR DOING WHAT WAS STARTED AS A COMPLETE 
    EVENT OR SERIES OF EVENTS BROUGHT TOGETHER WITHIN TIME2"
    INTENDED = "CONSIDERED TO HAPPEN OR DECIDED UPON TO COME INTO PLACE"
    LUCKY = "HAVING AND BRINGING AND OR RESULTING FROM GOOD_LUCK"
    LUCKINESS = "THE AMOUNT OF STATED QUALITY THAT ONE HAS FROM GOOD_LUCK IN 
    THE FORMS OR STATE OF HAVING GOOD_LUCK"
    STRONGLY = "WITH GREAT POWER OR STRENGTH"
    UNEASINESS = "A FEELING OF ANXIETY OR DISCOMFORT"
    DISTRESS = "EXTREME ANXIETY"
    ANGER = "A STRONG FEELING OF ANNOYANCE AND OR DISPLEASURE AND OR 
    HOSTILITY"
    ANNOYANCE = "A THING THAT CAN ANNOY SOMEONE"
    ANNOY = "IRRITATE SOMEONE OR TO MAKE SOMEONE ANGRY IN SOME WAY"
    ANGRY = "CURRENTLY HAVING ANGER OR FEELING ANGER IN SOME WAY"
    NERVOUSNESS = "THE QUALITY OR STATE OF BEING NERVOUS"
    TRANQUIL = "FREE FROM DISTURBANCE"
    IRRITATE = "TO ANNOY SOMEONE AND OR MAKE SOMEONE IMPATIENT AND OR ANGRY"
    DISPLEASURE = "A FEELING OF ANNOYANCE OR DISAPPROVAL"
    DISAPPROVAL = "EXPRESSION OF AN UNFAVORABLE OPINION"
    UNFAVORABLE = "EXPRESSING OR SHOWING A SMALLER OF APPROVAL OR SUPPORT 
    THAN WHAT IS NEEDED"
    APPROVAL = "THE ACTION OF APPROVING SOMETHING"
    EXPRESSION = "THE PROCESS OF MAKING KNOWN SUCH2 THOUGHTS OR FEELINGS" 
    HOPEFUL = "FEELING OR INSPIRING OPTIMISM ABOUT A FUTURE EVENT"
    INSPIRING = "GIVING SOMEONE POSITIVE OR CREATIVE FEELINGS"
    OPTIMISM = "HOPEFULNESS AND CONFIDENCE ABOUT THE FUTURE OR THE 
    SUCCESSFUL OUTCOME OF SOMETHING"
    SUCCESSFUL = "ACCOMPLISHING AN GOAL AND OR TASK AND JOB AND OR PURPOSE"
    ACCOMPLISHING = "ACHIEVING OR COMPLETING SUCCESSFULLY"
    ACCOMPLISH = "ACHIEVE OR COMPLETE SUCCESSFULLY"
    SUCCESSFULLY = "IN A WAY THAT ACCOMPLISHES A DESIRED GOAL OR RESULT"
    ACCOMPLISHES = "COMPLETES AND OR OBTAINS"
    COMPLETES = "SUCCEED IN ACCOMPLISHING"
    ASSET = "PROPERTY OWNED BY A PERSON AND OR ENTITY"
    RESULTING = "OCCURRING OR FOLLOWING AS THE CONSEQUENCE OF SOMETHING"
    WISHES = "MORE THAN ONE WISH"
    WISH = "FEEL OR EXPRESS A STRONG DESIRE OR HOPE FOR SOMETHING THAT IS NOT 
    ABLE TO BE OBTAINED WITH LITTLE DIFFICULTY"
    AMBIGUITY = "THE QUALITY OF BEING OPEN TO MORE THAN ONE INTERPRETATION"
    HOPEFULNESS = "A PERSON THAT HAS GREATER CHANCES SUCCEED FROM HOPEFUL 
    DECISION MAKING"
    OCCURRING = "HAPPENING AND IN EFFECT"
    ACHIEVING = "ACCOMPLISHING AND COMPLETING"
    COMPLETING = "SUCCEEDING IN ACCOMPLISHING"
    SUCCEEDING = "SUCCESSFULLY COMPLETING AS A FORM OF COMPLETION"
    SITUATIONS = "MORE THAN ONE SITUATION"
    AUSPICIOUS = "CHARACTERIZED BY SUCCESS"
    CHARACTERIZED = "DESCRIBED BY THE DISTINCTIVE NATURE OR FEATURES OF"
    FEATURE = "A DISTINCTIVE ATTRIBUTE OR ASPECT OF SOMETHING"
    FEATURES = "MORE THAN ONE FEATURE"
    NEEDS = "MORE THAN ONE NEED"
    DISTINCTIVE = "CHARACTERISTIC OF ONE PERSON OR THING AND OR SERVING TO 
    DISTINGUISH IT FROM OTHERS"
    EXISTING = "TO SERVE AS AN ATTRIBUTE OR QUANTITY"
    SERVE = "PERFORM JOBS AND OR TASKS AND OR WORK AND OR SERVICES FOR"
    DISTINGUISH = "PERCEIVE OR RECOGNIZE AND NOTICE A DIFFERENCE"
    FULFILLMENT = "THE MEETING OF A REQUIREMENT AND OR CONDITION AND OR 
    PRECONDITION AND OR PREREQUISITE"
    REQUIREMENT = "A CONDITION NECESSARY"
    NECESSARY = "REQUIRED TO BE DONE"
    PRECONDITION = "A CONDITION THAT MUST BE COMPATIBLE AND RECOGNIZED AS 
    EXISTING BEFORE OTHER THINGS CAN HAPPEN OR BE DONE"
    ASSETS = "MORE THAN ONE ASSET"
    BLOCK = "PREVENT FROM MAKING ACCESS POSSIBLE OR FURTHER ADVANCING OR 
    ACTION OR PASSAGEWAY OR MAKING A PROTECTION STANCE"
    NULLIFY = "PREVENT FROM HAPPENING AND NOT ALLOW TO COME INTO EFFECT OR 
    ACTION"
    NEGATE = "PREVENT FROM FORMING OR ACTIVATING"
    CANCEL_OUT = "NOT ALLOW TO HAPPEN AND NEGATE ALL FORMS OF EFFECTS AND 
    ACTION2"
    RESTRAINT = "A MEASURE OR CONDITION THAT ALLOWS SOMEONE OR SOMETHING TO 
    STAY AND HAVE CONTROL OR WITHIN LIMITS"
    DREAD = "ANTICIPATE WITH GREAT FEAR"
    STRESSFUL = "CAUSING MENTAL OR EMOTIONAL STRESS"
    TOLERATE = "ACCEPT OR ENDURE SOMEONE OR SOMETHING UNPLEASANT OR DISLIKED 
    WITH FORBEARANCE"
    TROUBLE = "DIFFICULTY OR PROBLEMS"
    CONSIDERATE = "CAREFUL NOT TO CAUSE INCONVENIENCE OR CREATE HARM TO 
    OTHERS"
    PERSISTANT = "CONTINUING TO EXIST OR ENDURE OVER A PROLONGED TIMEFRAME"
    SIGNIFICANT = "SUFFICIENTLY GREAT OR IMPORTANT TO BE WORTHY OF ATTENTION"
    HAPPINESS = "THE STATE OF BEING HAPPY"
    FRIENDLY = "KIND AND PLEASANT"
    READINESS = "WILLINGNESS TO DO SOMETHING"
    EXPECTED = "REGARDED AS LIKELY TO HAPPEN"
    MONEY = "A CURRENT MEDIUM OF EXCHANGE"
    RELATIVELY = "VIEWED IN COMPARISON WITH SOMETHING ELSE RATHER THAN 
    ABSOLUTELY"
    MODERATE = "AVERAGE IN AMOUNT AND OR INTENSITY AND OR" 
    FAVORABLE = "EXPRESSING APPROVAL"
    GESTURES = "MORE THAN ONE GESTURE"
    GESTURE = "A MOVEMENT OF PART OF THE BODY"
    KIND = "SOMEONE WHO IS GENEROUS AND HELPFUL AND WHO CAN THINK OF OTHER 
    AN ENTITIES FEELINGS"
    PLEASANT = "GIVING A SENSE OF HAPPY SATISFACTION"
    GENEROUS = "SHOWING KINDNESS TOWARD OTHERS"
    HELPFUL = "GIVING OR READY TO GIVE HELP"
    READY = "IN A STATE FOR AN EVENT AND OR ACTION AND OR SITUATION TO HAPPEN"
    SERVING = "THE ACTION OF ONE THAT SHALL SERVE SOMETHING"
    MORAL = "CONCERNED WITH THE PRINCIPLES OF RIGHT AND WRONG BEHAVIOR"
    CONCERNED = "FOCUSED AND CONCENTRATED ON"
    PRINCIPLES = "MORE THAN ONE PRINCIPLE"
    PRINCIPLE = "A FUNDAMENTAL TRUTH THAT CAN SERVE AS THE FOUNDATION FOR A 
    SYSTEM OF BELIEF OR BEHAVIOR OR FOR A CHAIN OF REASONING"
    FUNDAMENTAL = "A CENTRAL OR PRIMARY RULE BUILT UPON A SERIES OF BELIEFS"
    IMPORTANCE = "THE STATE OR FACT OF BEING OF GREAT SIGNIFICANCE OR VALUE"
    DIGNITY = "A RANK OR CLASS THAT SOMEONE OR SOMETHING IS APART OF"
    MODERATION = "AN EQUAL AMOUNT OF MODERATE PORTION THAT IS SPREAD ACROSS A 
    SPECIFIC PERIOD OF TIME2 WITHIN A TIMEFRAME"
    FACT = "A THING THAT IS KNOWN TO BE TRUE"
    SIGNIFICANCE = "THE MEANING TO BE FOUND IN WORDS OR EVENTS"
    STYLE = "A DISTINCTIVE APPEARANCE"
    SELF = "THE ESSENTIAL BEING THAT A PERSON IS THAT DISTINGUISHES THEM FROM 
    OTHERS"
    VOLUNTARILY = "DECISION MADE BY THE CHOICE OF FREE WILL"
    STRICTLY = "USED TO INDICATE THAT ONE IS APPLYING WORDS OR RULES EXACTLY OR 
    RIGIDLY"
    DETERRED = "PREVENT THE OCCURANCE OF"
    ESTIMATION = "A CALCULATION OF THE VALUE AND OR NUMBER AND OR QUANTITY AND 
    OR EXTENT OF SOMETHING WITHOUT HAVING ALL DATA2 KNOWN"
    MENTAL = "RELATING TO THE MIND"
    IMPATIENT = "HAVING OR SHOWING A REPEATED REACTION TO BE WITH LITTLE TO NO 
    DIFFICULT RESPONSE TO BECOME IRRITATED OR PROVOKED"
    HOSTILITY = "HOSTILE BEHAVIOR"
    DISLIKED = "FEEL DISTASTE FOR OR HOSTILITY TOWARD"
    STRESS = "A STATE OF MENTAL OR EMOTIONAL STRAIN CAUSED BY ADVERSE 
    CIRCUMSTANCES"
    FEAR = "AN EMOTION CAUSED BY THE BELIEF THAT SOMEONE OR SOMETHING IS 
    DANGEROUS"
    INCONVENIENCE = "TROUBLE OR DIFFICULTY CAUSED TO SOMEONE PERSONAL 
    REQUIREMENTS"
    ANTICIPATE = "REGARD AS PROBABLE AND OR EXPECT OR PREDICT"
    WILLINGNESS = "THE QUALITY OR STATE OF BEING PREPARED TO DO SOMETHING"
    REGARDED = "CONSIDERED AND THINK OF IN A SPECIFIED WAY"
    GETTING = "SUCCEEDING IN OBTAINING"
    CLEARNESS = "EASY TO PERCEIVE AND UNDERSTAND AND OR INTERPRET"
    ESPECIALLY = "TO A GREAT EXTENT"
    CAUTIOUS = "CHARACTERIZED BY THE DESIRE TO AVOID POTENTIAL PROBLEMS"
    WORTHY = "HAVING OR SHOWING THE QUALITIES OR ABILITIES THAT GIVE 
    RECOGNITION IN A SPECIFIED WAY"
    EXCHANGE = "AN ACTION OF GIVING ONE THING AND RECEIVING ANOTHER"
    GET = "RECEIVE SOMETHING FROM A SPECIFIC LOCATION OR SOURCE"
    EXPECTATIONS = "A STRONG BELIEF THAT SOMETHING WILL HAPPEN OR BE THE CASE IN 
    THE FUTURE"
    SUFFICIENTLY = "TO AN ADEQUATE CLASS OR LEVEL OF STATUS"
    ONESELF = "USED TO EMPHASIZE THAT ONE DOES SOMETHING INDIVIDUALLY OR 
    WITHOUT HELP"
    LIKELY = "HIGH CHANCES OF HAPPENING AND TO BE TRUE"
    COMPARISON = "A CONSIDERATION OR ESTIMATE OF THE SIMILARITIES OR 
    DIFFERENCES BETWEEN TWO THINGS"
    ABSOLUTELY = "WITH NO QUALIFICATION AND OR RESTRICTION AND OR LIMITATION"
    EASY = "DOES NOT REQUIRE A LARGE AMOUNT OF EFFORT OR HAS A STRONG FORM OF 
    DIFFICULTY"
    ESTIMATE = "AN APPROXIMATE CALCULATION OR DECISION OF THE VALUE AND OR 
    NUMBER AND OR QUANTITY AND OR EXTENT OF SOMETHING"
    APPROXIMATE = "CLOSE TO THE ORIGINAL RESULT BUT NOT ALWAYS AN COMPLETE 
    ACCURATE OR EXACT MEASUREMENT"
    PROBABLE = "LIKELY TO BE THE CASE OR TO HAPPEN"
    OCCURANCE = "AN INCIDENT OR EVENT"
    INCIDENT = "AN EVENT OR OCCURANCE"
    SURE = "CONFIDENT SOMETHING IS GOING TO HAPPEN"
    DIFFERENCES = "MORE THAN ONE DIFFERENCE"
    RANK = "A SPECIFIC POSITION FOR A CLASS WITHIN A HEIRARCHIAL STRUCTURE"
    EXPECT = "REQUIRE TO ARRIVE AT A SPECIFIC TIMEFRAME AND MANNER"
    REGARD = "RELATING TO THE CURRENT CIRCUMSTANCE"
    RECOGNITION = "A FORM OF PROCESS THAT CAN RECOGNIZE CERTAIN SPECIFIC FORMS 
    OF FACTS AND INFORMATION FROM DIFFERENT CLASSES OR GENRES OF INFORMATION"
    PREPARED = "ALREADY DONE AHEAD OF TIME2"
    ADEQUATE = "SPECIFIC AND PARTICULAR RESPONSE THAT HOLDS MANNERISMS"
    MANNERISM = "A FORM OF ETTIQUITE THAT HOLDS SPECIFIC PATTERNS ON HOW TO ACT 
    PROPERLY"
    PROPER = "WHAT SHOULD BE USED DEPENDING ON THE CIRCUMSTANCE OR CLASS 
    WITHIN"
    PROPERLY = "USING MANNERS WITH PROPER METHODS AND APPROACH"
    MANNERS = "MORE THAN ONE MANNER"
    ETTIQUITE = "A SPECIFIC FORMAL OR PROPER STYLE FOR HOW TO BEHAVE DURING AN 
    EVENT OR CIRCUMSTANCE"
    MANNERISMS = "MORE THAN ONE MANNERISM"
    FACTS = "MORE THAN ONE FACT"
    REQUIREMENTS = "MORE THAN ONE REQUIREMENT"
    MISHAP = "AN INCIDENT NOT CONTAINING GOOD_FORTUNE OR GOOD_LUCK"
    AVOIDING = "PREVENTING FROM DOING"
    UNCOMFORTABLE = "CAUSING OR FEELING UNEASE"
    UNEASY = "CAUSING OR FEELING ANXIETY AND OR UNCOMFORTABLE"
    DISCOMFORT = "MAKE FEEL UNEASY ANXIOUS"
    UNPLEASANT = "UNFRIENDLY AND INCONSIDERATE"
    UNFRIENDLY = "NOT FRIENDLY"
    INCONSIDERATE = "THOUGHTLESSLY CAUSING HURT OR INCONVENIENCE TO OTHERS"
    THOUGHTLESSLY = "WITHOUT CONSIDERATION OF THE POSSIBLE CONSEQUENCES"
    SUFFERING = "THE STATE OF DISTRESS"
    UNASSUMING = "MODEST"
    UPSET = "MAKE SOMEONE UNHAPPY AND OR DISAPPOINTED AND OR WORRIED"
    PREFER = "LIKE ONE IDEA OR THING OR SUBJECT OR TOPIC OR PERSON BETTER THAN 
    ANOTHER OR OTHERS"
    HEALING = "THE PROCESS OF MAKING OR BECOMING SOUND OR HEALTHY AGAIN"
    HEAL = "BECOME SOUND OR HEALTHY AGAIN"
    REFRESH = "TO RESTORE OR MAINTAIN BY RENEWING SUPPLY"
    HEALTHY = "IN DESIRED HEALTH"
    HEALTH = "THE MENTAL OR PHYSICAL CONDITION OF SOMEONE OR SOMETHING"
    REFRESHING = "HAVING THE POWER TO RESTORE FRESHNESS AND OR VITALITY AND OR 
    ENERGY"
    RENEW = "TO RENEW MEANS TO BRING BACK TO AN ORIGINAL CONDITION OF 
    FRESHNESS AND VIGOR"
    FRESHNESS = "THE QUALITY OF BEING PLEASANTLY NEW OR DIFFERENT"
    VIGOR = "PHYSICAL STRENGTH AND DESIRED HEALTH"
    VITALITY = "THE POWER GIVING AND CONTINUING TO GAIN A CONTINUAL AMOUNT OF 
    ENERGY"
    INDIVIDUALLY = "IN AN INDIVIDUAL CAPACITY"
    EMPHASIZE = "GIVE SPECIAL IMPORTANCE TO SOMETHING IN WRITING"
    FORBEARANCE = "RESTRAINT AND CONTROL OF SOMETHING"
    CONSIDERATION = "CAREFUL THOUGHT WITHIN A PERIOD OF TIME"
    QUALIFICATION = "THE ACTION OR FACT OF QUALIFYING"
    RENEWANCE = "TO BEGIN OR TAKE UP AGAIN"
    CONSEQUENCES = "MORE THAN ONE CONSEQUENCE"
    PREVENTING = "PREVENT FROM HAPPENING COMING INTO EFFECT OR PLACE"
    UNHAPPY = "NOT HAPPY"
    HURT = "PHYSICAL HARM OR INJURY"
    DISAPPOINTED = "SAD OR DISPLEASED BECAUSE SOMEONE OR SOMETHING HAS FAILED 
    TO FULFILL HOPES OR EXPECTATIONS"
    WORRIED = "ANXIOUS OR TROUBLED ABOUT ACTUAL OR POTENTIAL PROBLEMS"
    INCONVENIENCE = "TROUBLE OR DIFFICULTY CAUSED TO ONES PERSONAL 
    REQUIREMENTS OR COMFORT"
    HOPES = "MORE THAN ONE HOPE"
    EXPECTATIONS = "MORE THAN ONE EXPECTATION"
    EXPECTATION = "A STRONG BELIEF THAT SOMETHING WILL HAPPEN OR BE THE CASE IN 
    THE FUTURE"
    VIRTUE = "A QUALITY CONSIDERED DESIRED INSIDE A PERSON"
    SOMEPLACE = "A REFERENCE TO A SPECIFIC PLACE AND OR LOCATION WITHIN SPACE 
    AND OR TIME"
    INDISTINCTNESS = "NOT CLEAR OR SHARPLY_DEFINED"
    SHARPLY_DEFINED = "IN A WAY THAT IS DISTINCT IN DETAIL"
    RENEWING = "HAPPENING TO RENEW AT THE CURRENT TIMEFRAME"
    PLEASANTLY = "IN AN ENJOYABLE OR AGREEABLE MANNER"
    FORMAL = "CONTAINING A FORM OF ETTIQUITE OR MANNERISMS THAT SHOW PROPER 
    DECISION MAKING WITH BOTH ATTITUDE AND STYLE AS WELL AS PRESENTATION AND 
    CONFIDENCE"
    PRUDENT = "SHOWING CARE AND THOUGHT FOR THE FUTURE"
    PRUDENCE = "THE QUALITY OF BECOMING PRUDENT"
    FRIGHTENS = "INCLUDES AND DOES APPLY FEAR WITHIN"
    ENJOYMENT = "THE STATE OR PROCESS OF TAKING PLEASURE IN SOMETHING"
    DERIVED = "A IDEA CREATED ON A EXTENSION OF LOGIC OR MODIFICATION OF 
    ANOTHER IDEA"
    OFFERING = "A CONTRIBUTION OR A THING OFFERED AS A TOKEN OF DEVOTION"
    DEVOTION = "AS IN AFFECTION A FEELING OF STRONG OR CONSTANT REGARD FOR AND 
    DEDICATION TO SOMEONE"
    DEDICATION = "THE QUALITY OF BEING DEDICATED TO A TASK OR PURPOSE"
    DEDICATED = "DEVOTED TO A TASK OR PURPOSE"
    RESOURCES = "MORE THAN ONE RESOURCE"
    DEVOTED = "GIVEN OVER TO THE STUDY OF"
    TOKEN = "A THING SERVING AS A REPRESENTATION OF A FACT AND OR QUALITY AND OR 
    FEELING"
    ENJOYABLE = "GIVING DELIGHT OR PLEASURE"
    DELIGHT = "GREAT PLEASURE"
    AGREEABLE = "ENJOYABLE AND PLEASURABLE"
    PLEASURABLE = "PLEASING"
    PLEASING = "SATISFYING OR APPEALING"
    APPEALING = "ATTRACTIVE OR INTERESTING"
    ATTRACTIVE = "PLEASING OR APPEALING TO THE SENSES"
    INTERESTING = "HOLDING OR TO GAIN THE ATTENTION OF OR AROUSING INTEREST"
    AROUSE = "EXCITE OR PROVOKE SOMEONE TO STRONG EMOTIONS"
    EXCITE = "CAUSE STRONG FEELINGS OF ENTHUSIASM AND EAGERNESS WITHIN"
    ENTHUSIASM = "INTENSE AND EAGER ENJOYMENT AND OR INTEREST AND OR 
    APPROVAL"
    EAGER = "WANTING TO DO OR HAVE SOMETHING VERY MUCH"
    EAGERNESS = "ENTHUSIASM TO DO OR TO HAVE SOMETHING"
    SATISFYING = "GIVING FULFILLMENT OR THE PLEASURE ASSOCIATED WITH SOMETHING 
    OR SOMEPLACE OR SOMEONE"
    ASSOCIATED = "CONNECTED WITH ANOTHER GROUP OR GROUPS"
    PROVOKE = "GIVE RISE TO A REACTION OR EMOTION IN SOMEONE OR AROUSE 
    SOMEONE TO DO OR FEEL SOMETHING"
    PROVOKED = "ENVOKED SOMEONE TO FEEL SOMETHING STRONGLY"
    ENVOKE = "AROUSE SOMEONE TO DO SOMETHING USING ENERGY AND OR CHARMS 
    AND OR INCANTATIONS"
    ENVOKED = "SUMMONED A SPIRIT BY USING CHARMS OR INCANTATION"
    CHARM = "AN OBJECT AND OR SERIES OF WORDS THAT HAVE MAGIC POWER"
    INCANTATION = "A SERIES OF WORDS EXPRESSED AS A MAGIC SPELL OR CHARM"
    SUMMON = "CALL PEOPLE TO BECOME APART OF A MEETING"
    CHARMS = "MORE THAN ONE CHARM"
    INCANTATIONS = "MORE THAN ONE INCANTATION"
    SUMMONED = "BROUGHT INTO AN EVENT OR CIRCUMSTANCE TO MAKE ACTION BY THE 
    DEMAND OF THE SUMMONER"
    SUMMONER = "THE ONE WHO SHALL SUMMON SOMETHING OR SOMEONE WITH 
    COMPLETE CONTROL OF WHAT IS SUMMONED AS AN ENTITY FOR THE TIMEFRAME THAT 
    THE SUMMONED ENTITY IS SUMMONED"
    ENCHANT = "PUT SOMEONE OR SOMETHING UNDER A SPELL"
    ESSENTIAL = "A THING THAT IS ABSOLUTELY NECESSARY"
    DISTINGUISHES = "RECOGNIZE SOMEONE OR SOMETHING AS SIMILAR OR DIFFERENT"
    THEM = "TO REFERENCE TO TWO OR MORE PEOPLE OR THINGS PREVIOUSLY 
    RECOGNIZED WITHIN REFERENCE"
    RIGIDLY = "IN A STRICT OR EXACTING WAY"
    INDICATE = "SUGGEST AS A DESIRED OR NECESSARY CHOICE OF ACTION"
    DISTASTE = "TO NOT LIKE SOMETHING BECAUSE YOU CONSIDER IT UNPLEASANT"
    HOSTILE = "UNFRIENDLY OR AGAINST THE IDEA OF SOMETHING"
    STRAIN = "FORCES THAT IS ABLE PULL FROM MULTIPLE LOCATIONS UNTIL IT CREATES 
    STRESS UPON THE ENTITY"
    ADVERSE = "PREVENTING SUCCESS OR DEVELOPMENT AND OR IS HARMFUL"
    CIRCUMSTANCES = "MORE THAN ONE CIRCUMSTANCE"
    IRRITATED = "FEELING DISCOMFORT OR DISCONTENT"
    TROUBLED = "AFFECTED BY PROBLEMS OR UNCOMFORTABLE CIRCUMSTANCES"
    SAD = "TO BE FEELING UNHAPPY"
    DISPLEASED = "FEELING OR SHOWING ANNOYANCE AND DISPLEASURE"
    QUALIFYING = "DENOTING SOMEONE OR SOMETHING THAT IS COMPATIBLE FOR 
    SOMETHING TO TAKE PLACE OR HAPPEN"
    DENOTE = "BE A SIGN OF"
    DENOTING = "SHOWING A SIGN OF SOMETHING OR SOMEPLACE OR SOMEONE"
    SIGN = "AN OBJECT AND OR QUALITY AND OR EVENT IN WHICH SOMEONE OR 
    SOMETHING HAS A PRESENCE OR OCCURANCE THAT DOES INDICATE THE PROBABLE 
    PRESENCE OR OCCURANCE OF SOMETHING ELSE"
    FULFILL = "BRING TO COMPLETION OR REALITY"
    COMFORT = "A STATE OF PHYSICAL ABSENSE OF DIFFICULTY OR EFFORT AND FREEDOM 
    FROM UNCOMFORTABLE FEELINGS OR BINDINGS"
    CONTRIBUTION = "A GIFT OR PAYMENT TO A COMMON SOURCE OF GRATITUDE"
    GRATITUDE = "THE QUALITY OF BEING THANKFUL"
    THANKFUL = "EXPRESSING GRATITUDE AND RELIEF"
    RELIEF = "A FEELING OF REASSURANCE AND RELAXATION FOLLOWING RELEASE FROM 
    ANXIETY OR DISTRESS"
    REASSURANCE = "THE ACTION OF REMOVING THE DOUBTS OR FEARS OF SOMEONE"
    DOUBTS = "MORE THAN ONE DOUBT"
    FEARS = "MORE THAN ONE FEAR"
    RELAXATION = "THE STATE OF BEING FREE FROM TENSION AND ANXIETY"
    TENSION = "MENTAL OR EMOTIONAL STRAIN"
    OFFERED = "GIVE AN OPPORTUNITY FOR SOMETHING TO BE MADE OR CREATED"
    AROUSING = "REACHING A RESPONSE OR REACTION TO AROUSE SOMEONE OR 
    SOMETHING"
    STRICT = "FOLLOWING RULES OR BELIEFS EXACTLY"
    EXACTING = "MAKING GREAT ORDERS REGARDING ONES SKILL AND OR ATTENTION AND 
    OR OTHER RESOURCES"
    DOUBT = "TO FEAR"
    ACTUAL = "EXISTING WITHIN THE PRESENT"
    PROUD = "FEELING STRONG PLEASURE OR SATISFACTION AS A RESULT OF ONES OWN 
    ACHIEVEMENTS"
    RHYTHM = "A STRONG AND OR REPEATED PATTERN OF MOVEMENT OR SOUND"
    PITCH = "THE QUALITY OF A SOUND CONTROLLED BY THE RATE OF VIBRATIONS 
    PRODUCING IT"
    TREBLE = "CREATED WITHIN THE EXISTENCE OF THREE PARTS"
    BASS = "THE LOW FREQUENCY OUTPUT OF A AUDIO SYSTEM"
    LOW = "BELOW AVERAGE IN AMOUNT"
    MUSIC_SCALE = "A SERIES OF NOTES DIFFERING IN PITCH ACCORDING TO A SPECIFIC 
    PATTERN"
    MUSICAL_NOTE = "DESCRIBES THE PITCH AND THE DURATION OF A SPECIFIC SOUND"
    MUSIC2 = "VOCAL AND OR INSTRUMENTAL SOUNDS COMBINED IN A WAY AS TO 
    PRODUCE BEAUTY OF FORM, HARMONY, AND EXPRESSION OF EMOTION"
    BEAUTY = "A COMBINATION OF QUALITIES THAT PLEASES THE INTELLECT OR MORAL 
    SENSE"
    INTELLECT = "THE FACULTY OF REASONING AND UNDERSTANDING OBJECTIVELY"
    OBJECTIVELY = "IN A WAY THAT IS NOT INFLUENCED BY PERSONAL FEELINGS OR 
    OPINIONS"
    OPINIONS = "MORE THAN ONE OPINION"
    DESCRIBES = "ATTEMPTS TO GIVE THE COMPLETE DESCRIPTION OF SOMETHING OR 
    SOMEPLACE"
    INFLUENCE = "THE CAPACITY TO HAVE AN EFFECT ON A CHARACTER AND OR 
    DEVELOPMENT AND OR BEHAVIOR OF SOMEONE OR SOMETHING AND OR THE EFFECT 
    ITSELF"
    INFLUENCED = "CONTROLLED BY INFLUENCE OR REPEATING PATTERNS AND OR 
    CIRCUMSTANCES"
    ATTEMPTS = "GIVES AN EFFORT TO ATTEMPT"
    INFLUENCE2 = "HAVE AN INFLUENCE UPON USING A DEVICE OR HAVING SOMETHING 
    INFLUENCE THE PHYSICAL2 OR MENTAL SENSES"
    NOTE = "A SERIES OF FACTS AND OR TOPICS AND OR THOUGHTS THAT ARE WRITTEN 
    DOWN AS AN SUPPORT MEMORY REMEMBER AND OR RECALL"
    ATTEMPT = "MAKE AN EFFORT TO ACHIEVE OR COMPLETE"
    WILLED_ENERGY = "A CLASS OF ENERGY THAT IS USED BY THE SOLE CONSENT OF THE 
    SOURCE CONNECTION FOR THE ENERGY TO BE APPLIED AND REQUIRED FOR BOTH 
    HOST AND SOURCE TO BE IN AGREEMENT FOR ENERGY TO BE APPLIED"
    SOLE = "REPRESENTING AN INDIVIDUAL DECISION ONLY WITHOUT INFLUENCE FROM 
    OTHER ENTITIES OR CHOICES AND IS REPRESENTING ONLY ONE INDIVIDUAL"
    CONSENT = "PERMISSION FOR SOMETHING TO HAPPEN OR AN AGREEMENT TO DO 
    SOMETHING"
    FORCED_ENERGY = "A CLASS OF ENERGY THAT IS USED BY APPLYING PRESSURE OR 
    FORCE TO THE SOURCE CONNECTION FOR EFFECTS TO HAPPEN"
    NULL_AND_VOID = "MEANS HAVING NO EFFECT AND TO BE CONSIDERED AS IF IT DOES 
    NOT EXIST"
    EMOTIONAL_ABUSE = "IS CONSIDERED TO BE ANYTHING THAT CAUSES FEAR OR TO 
    MANIPULATE THE EMOTIONS OF A LESSER INDIVIDUAL"
    SPIRITUAL_ABUSE = "IS THE HARM THAT COMES TO THE HUMAN SPIRIT"
    HOST = "A COMPUTER THAT IS ACCESSIBLE WITHIN A NETWORK OR SOMEONE WHO IS 
    ABLE TO COMMUNICATE WITH OTHERS THAT EXIST WITHIN THE SAME NETWORK OF 
    OTHERS"
    WAITER = "SOMEONE WHO IS RESPONSIBLE FOR TAKING ORDERS FROM SOMEONE AND 
    TO SEND THEIR SERVICE REQUESTED TO THEM"
    RESPONSIBLE = "SOMEONE WHO IS ABLE TO BE GIVEN TRUST"
    NOTES = "MORE THAN ONE NOTE"
    DIFFERING = "NOT THE SAME AS EACH OTHER"
    WEALTHY = "HAVING A GREAT AMOUNT OF MONEY AND OR RESOURCES AND OR 
    ASSETS"
    RICH = "PRODUCING A LARGE QUANTITY OF SOMETHING"
    EXPENSIVE = "REQUIRE A LARGE AMOUNT OF MONEY"
    COST = "AN AMOUNT THAT HAS TO BE PAID OR SPENT TO BUY OR OBTAIN SOMETHING"
    PRICE = "THE AMOUNT EXPECTED AND REQUIRED OR GIVEN PAYMENT FOR SOMETHING"
    VALUE = "THE REGARD THAT SOMETHING IS HELD TO DESERVE"
    COMMUNITY = "A PARTICULAR AREA OR PLACE CONSIDERED TOGETHER AS A WHOLE 
    GROUP"
    ORGANIZATION = "AN ORGANIZED GROUP OF PEOPLE WITH A PARTICULAR PURPOSE"
    PAY = "GIVE SOMEONE MONEY FOR WORK AND OR SERVICES AND OR PRODUCTS 
    COMPLETED"
    PAID = "GIVEN PAYMENT TO PAY SOMEONE"
    SPEND = "PAY SOMEONE FOR RESOURCES AND OR ASSETS AND OR SERVICES"
    SPENT = "INCOME THAT HAS BEEN USED TO BUY SOMETHING OR PAY FOR A SERVICE"
    BUY = "OBTAIN IN EXCHANGE FOR PAYMENT"
    DESERVE = "DO SOMETHING OR TO HAVE OR SHOW QUALITIES WORTHY OF 
    SOMETHING"
    FIXED_INCOME = "IS A TERM THAT CAN REFER TO EITHER A INCOME THAT DOES NOT 
    CHANGE IN AMOUNT"
    REFER = "DIRECT THE ATTENTION OF SOMEONE TO"
    INCOME = "MONEY RECEIVED AT A SPECIFIC INTERVAL OR FREQUENCY OF TIME WITHIN 
    A SERIES OF CONTINUOUS REPEATED EVENTS THAT RESULT IN SOMEONE GETTING 
    PAYMENT"
    DURATION = "THE TIME DURING WHICH SOMETHING CONTINUES"
    CONTINUES = "HAPPENS TO CONTINUE EXISTING WITHIN TIME2 AND SPACE2"
    PLEASES = "GIVES SATISFACTION OR PLEASURE"
    PLEASE = "CAUSE TO FEEL HAPPY AND OR TO FEEL ENJOYMENT"
    INSTRUMENTS = "MORE THAN ONE INSTRUMENT"
    PERFORM = "GIVEN FULFILLMENT AND COMPLETED"
    PERFORMED = "COMPLETED THE METHOD TO PERFORM A SPECIFIC JOB OR TASK OR 
    AMOUNT OF WORK DURING AN EVENT"
    INSTRUMENTAL = "SOMETHING THAT IS PERFORMED USING INSTRUMENTS AND 
    WITHOUT VOCAL SOUNDS"
    PAIN = "A SPECIFIC AMOUNT OF SENSITIVITY TO ANOTHER SENSOR"
    class Language_Extension_002_2:
    RECURSIVE_DEFINITION = "IS USED TO DEFINE THE ELEMENTS INSIDE A LIST WITH 
    TERMS OF OTHER ELEMENTS INSIDE THE LIST"
    QUANTUM_MECHANICS = "IS A FUNDAMENTAL THEORY THAT CAN PROVIDE A 
    DESCRIPTION OF THE PHYSICAL2 PROPERTIES OF NATURE AT THE SCALE OF ATOMS AND 
    SUBATOMIC PARTICLES"
    QUANTUM_CHEMISTRY = "IS A SUBCLASS OF PHYSICAL_CHEMISTRY THAT GIVES FOCUS 
    TO INCLUDING QUANTUM_MECHANICS TO CHEMICAL SYSTEMS"
    MACROSCOPIC_SCALE = "IS THE LENGTH SCALE ON WHICH OBJECTS3 OR PHENOMENA 
    ARE AT A SPECIFIC SIZE THAT IS LARGE AND CAPABLE TO BE VISIBLE TO THE EYESIGHT 
    WITHOUT REQUIREMENT TO MAGNIFY TO SEEN"
    MICROSCOPIC_SCALE = "IS THE SCALE OF OBJECTS3 AND EVENTS SMALLER THAN 
    THOSE THAT CAN BE SEEN WITH VERY LITTLE DIFFICULTY BY THE EYESIGHT AND MAY 
    REQUIRE TO MAGNIFY EYESIGHT TO SEE THE OBJECTS3"
    QUANTUM_FIELD_THEORY = "A FRAMEWORK THAT COMBINED 
    CLASSICAL_FIELD_THEORY AND SPECIAL_RELATIVITY AND QUANTUM_MECHANICS 
    TOGETHER"
    PHYSICAL_CHEMISTRY = "IS THE STUDY OF MACROSCOPIC_SCALE AND 
    MICROSCOPIC_SCALE PHENOMENA WITHIN CHEMICAL SYSTEMS RELATING TO TERMS 
    OF THE PRINCIPLES AND OR PRACTICES AND OR IDEAS OF PHYSICS SUCH2 AS MOTION 
    AND ENERGY AND FORCE AND TIME2 AND THERMODYNAMICS AND 
    QUANTUM_CHEMISTRY AND STATISTICAL_MECHANICS AND ANALYTICAL_DYNAMICS 
    AND CHEMICAL_EQUALIBRIA"
    THERMODYNAMICS = "A CLASS THAT DESCRIBES THE FUNCTION AND USE OF HEAT AND 
    WORK AND TEMPERATURE AND THE RELATION IT HAS TO ENERGY AND THE PHYSICAL2 
    PROPERTIES OF MATTER"
    STATISTICAL_MECHANICS = "A FRAMEWORK THAT USES ANALYSIS AND SCANNING 
    METHODS AS WELL AS MANY FORMS OF CALCULATION AND ALSO INCLUDES 
    PROBABILITY_THEORY TO LARGE AMOUNTS OF PRODUCTS BROUGHT TOGETHER THAT 
    CONSISTS MICROSCOPIC_SCALE ENTITIES"
    ANALYTICAL_DYNAMICS = "IS CONCERNED WITH THE RELATION BETWEEN MOTION OF 
    BODIES AND ITS CAUSES"
    CHEMICAL_EQUALIBRIA = "IS THE STATE IN WHICH BOTH THE CHEMICAL_REACTANTS 
    AND CHEMICAL_PRODUCTS ARE CURRENTLY EXISTING IN CONDENSED AMOUNTS AND 
    TO WHICH HAVE NO FURTHER CHANCES OR CAPABILITIES TO CHANGE WITHIN ANY 
    POINT INSIDE TIME2"
    ENUMERATED_TYPE = "IS A DATA2 TYPE INCLUDING A LIST OF NAMED VALUES KNOWN AS 
    ENUMERATION_ELEMENTS AND ENUMERATION_MEMBERS"
    ENUMERATOR_NAMES = "ARE A WAY OF IDENTIFICATION THAT BEHAVE AS CONSTANT 
    DATA2 WITHIN THE LANGUAGE"
    PROGRAMMING_VALUE = "IS THE REPRESENTATION OF SOME ENTITY THAT A PROGRAM 
    CAN MANIPULATE"
    CLASSICAL_FIELD_THEORY = "IS A PHYSICAL_THEORY THAT CAN PREDICT HOW ONE OR 
    MORE FIELD COMMUNICATE AND CONNECT WITH MATTER USING EQUATIONS FOR 
    FIELDS"
    PHYSICAL_THEORY = "IS A CLASS THAT EMPLOYS EQUATIONS AND OR FORMULAS AND 
    OR ALGORITHMS FOR DEVELOPED IDEAS AND MORE THAN ONE ABSTRACTION OF 
    PHYSICAL2 OBJECTS3 AND SYSTEMS TO MAKE UNDERSTANDING OF AND OR EXPLAIN 
    AND PREDICT NATURAL PHENOMENA"
    VECTOR_ELEMENT = "IS A SPECIFIC PROGRAMMING_ELEMENT MADE FOR POSSIBLE USE 
    WITH AND OR INSIDE VECTOR DATA2"
    VECTOR_SPACE = "IS A LIST THAT HAS MORE THAN ONE VECTOR_ELEMENT MAY BE 
    ADDED TOGETHER AND MULTIPLIED BY SCALAR NUMBERS"
    SCALAR = "IS AN ELEMENT OF A FIELD WHICH IS USED TO DEFINE A VECTOR_SPACE"
    INSERTING = "MAKING AND PROCESSING THE PROCESS TO INSERT SOMETHING"
    DELETING = "MAKING AND PROCESSING THE PROCESS TO REMOVE SOMETHING"
    DATA_MANIPULATION_LANGUAGE = "IS A COMPUTER PROGRAMMING LANGUAGE USED 
    FOR INSERTING OR DELETING OR MODIFYING DATA2 INSIDE A DATABASE"
    ARRAY = "IS AN DATA2 STRUCTURE THAT CONSISTS OF A COLLECTION OF ELEMENTS 
    THAT ARE CONSIDERED VALUES OR VARIABLES"
    VECTOR_PROCESSOR = "IS A CENTRAL PROCESSING UNIT THAT CONNECTS AN LIST OF 
    INSTRUCTIONS WHEN THE INSTRUCTIONS ARE MADE TO OPERATE EFFECTIVELY ON 
    LARGE ONE DIMENSIONAL ARRAY SYSTEMS KNOWN AS PROCESSOR VECTORS"
    PROGRAMMING_ELEMENT = "IS ANY ONE OF THE DISTINCT OBJECTS3 THAT BELONG TO A 
    LIST OR AN ARRAY"
    TUPLE = "A FINITE SEQUENCE OF ELEMENTS"
    REPLACES = "CHANGES SOMETHING OUT FOR SOMETHING ELSE"
    SUBSTANCES = "MORE THAN ONE SUBSTANCE"
    CHEMICAL_REACTANTS = "ARE CONSIDERED THE SUBSTANCES THAT ARE PRESENT AT 
    THE START OF THE CHEMICAL_REACTION"
    CHEMICAL_PRODUCTS = "ARE CONSIDERED THE SUBSTANCES THAT ARE FORMED AT 
    THE END OF THE CHEMICAL_REACTION"
    REACTANT = "A SUBSTANCE THAT IS USED WITHIN A CHEMICAL_REACTION"
    EMPLOYS = "MAKES EFFICIENT OPERATION OF"
    SYNTHESIS_REACTION = "TWO OR MORE CHEMICAL_REACTANTS COMBINE TO FORM A 
    SINGLE PRODUCT"
    DECOMPOSITION_REACTION = "A SINGLE REACTANT SEPARATED INTO PARTS OR PIECES 
    AND IS MADE INTO TWO OR MORE CHEMICAL_PRODUCTS"
    SINGLE_REPLACEMENT_REACTION = "ONE ELEMENT REPLACES ANOTHER ELEMENT IN A 
    COMPOUND"
    DOUBLE_REPLACEMENT_REACTION = "TWO COMPOUNDS EXCHANGE CHEMICAL_IONS 
    TO FORM TWO NEW COMPOUNDS"
    NET_ELECTRICAL_CHARGE = "IS A CHARGE THAT IS A RESULT FROM THE DECREASE OR 
    INCREASE OF ELECTRONS"
    CHEMICAL_ION = "IS AN ATOM OR CHEMICAL_MOLECULE THAT HAS A 
    NET_ELECTRICAL_CHARGE"
    CHEMICAL_MOLECULE = "IS A GROUP OF TWO OR MORE ATOMS HELD TOGETHER BY 
    CHEMICAL_BONDS"
    CHEMICAL_BOND = "IS AN ATTRACTION BETWEEN ATOMS THAT ALLOWS TO HAPPEN THE 
    FORMING OF CHEMICAL_MOLECULES AND OR OTHER CHEMICAL_STRUCTURES"
    CHEMICAL_BONDS = "MORE THAN ONE CHEMICAL_BOND"
    CHEMICAL_MOLECULES = "MORE THAN ONE CHEMICAL_MOLECULE"
    CHEMICAL_IONS = "MORE THAN ONE CHEMICAL_ION"
    CHEMICAL_REACTIONS = "MORE THAN ONE CHEMICAL_REACTION"
    COMPOUNDS = "MORE THAN ONE COMPOUND"
    QUANTUM_MECHANICS = "IS THE FOUNDATION OF ALL QUANTUM_PHYSICS INCLUDING 
    QUANTUM_CHEMISTRY AND OR QUANTUM_FIELD_THEORY AND OR 
    QUANTUM_TECHNOLOGY AND OR QUANTUM_INFORMATION_SCIENCE"
    REACTANTS = "MORE THAN ONE REACTANT"
    CHEMICAL_REACTION = "IS A PROCESS IN WHICH ONE OR MORE SUBSTANCE CAN BE 
    PROCESSED TO TRANSFORM INTO ONE OR MORE NEW SUBSTANCE"
    ATTRACTION = "TO BE ATTRACTED TO SOMEONE OR SOMETHING"
    COMPOUND = "A THING THAT IS COMBINED WITH TWO OR MORE SEPARATE PARTS OR 
    PIECES OR ELEMENTS"
    INTELLECTUAL = "SOMEONE OR SOMETHING HAVING A HIGHLY DEVELOPED INTELLECT"
    EMOTIONAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE EMOTIONAL CONNECTION THAT TWO PEOPLE FEEL FOR EACH OTHER"
    PHYSICAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE PHYSICAL APPEARANCE OF THE BODY OF A PERSON AND OR ENTITY AND OR 
    BEING2"
    INTELLECTUAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE INTELLECTUAL CONNECTION THAT TWO PEOPLE AND OR ENTITIES THINK 
    ABOUT EACH OTHER"
    SPIRITUAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE SPIRITUAL CONNECTION THAT TWO PEOPLE SHARE WITH EACH OTHER"
    OCCASIONS = "MORE THAN ONE OCCASION"
    COMPLEX = "SOMETHING FILLED WITH MANY DIFFICULT TO UNDERSTAND PARTS OR 
    PIECES OR FUNCTIONS OR MATERIALS"
    QUANTUM_INFORMATION_SCIENCE = "IS A THEORY THAT STUDIES THE CAPABILITY TO 
    USE AND MAKE POSSIBLE APPLYING QUANTUM_MECHANICS TO 
    INFORMATION_PROCESSING AND COMMUNICATION"
    PROBABILITY_THEORY = "IS A THEORY THAT IS USED FOR THE ANALYSIS OF RANDOM 
    PHENOMENA"
    PROBABILITY_STATISTICS = "IS USED TO ESTIMATE THE LIKELY CHANCE THAT AN EVENT 
    CAN OR IS GOING TO HAVE AN OCCURANCE"
    PROBABILITY_GAME_THEORY = "IS USED TO ANALYZE THE INTELLIGENT AND WELL 
    PLANNED DECISIONS MADE BY PLAYERS AND OR ARTIFICIAL INTELLIGENCE SYSTEMS 
    WHEN LUCKINESS OR CHANCE AND OR OF SKILL AND ACCURACY OR GOOD_LUCK 
    EXISTS"
    SPECIAL_RELATIVITY = "IS A THEORY THAT DESCRIBES HOW SPACE2 AND TIME2 ARE 
    LINKED FOR OBJECTS3 THAT ARE MOVING AT A SPECIFIC SPEED FOR A TIMEFRAME 
    WHILE IN A STRAIGHT LINE"
    PHYSICS = "IS A CLASS THAT CONCERNED WITH THE NATURE AND PROPERTIES OF 
    MATTER AND ENERGY"
    QUANTUM_PHYSICS = "IS A SUBCLASS WITHIN PHYSICS THAT IS USED TO PROVIDE A 
    DESCRIPTION OF QUANTUM_MECHANICS"
    ENUMERATION_ELEMENT = "IS A MEMBER OF AN ENUMERATION"
    ENUMERATION_MEMBER = "IS A NAMED CONSTANT THAT IS WITHIN A ENUMERATION 
    THAT IS EITHER A VARIABLE OR FUNCTION OR OTHER DATA2 STRUCTURE THAT IS PART OF 
    A CLASS AND CAN BE ACCESSIBLE2 TO OTHER ENUMERATION_MEMBERS OF THE CLASS 
    BUT NOT THE CODE OUTSIDE OF THE CLASS THAT THE ENUMERATION_MEMBER EXISTS 
    WITHIN"
    ENUMERATION_MEMBERS = "MORE THAN ONE ENUMERATION_MEMBER"
    INFORMATION_PROCESSING = "IS THE MANIPULATION OF DATA2 TO PRODUCE AN 
    DISPLAY FULL OF DEFINING LOGIC AND EFFICIENT REASONING"
    PROCESSING_CYCLE_INPUT = "IS TO INPUT THE INFORMATION TO BE PROCESSED"
    PROCESSING_CYCLE_PROCESSING = "IS TO PROCESS THE INFORMATION"
    PROCESSING_CYCLE_OUTPUT = "IS TO DISPLAY THE PROCESSED INFORMATION"
    PROCESSING_CYCLE_STORAGE = "IS TO STORE THE PROCESSED INFORMATION INSIDE A 
    SPECIFIC LOCATION"
    CHEMICAL_STRUCTURE = "IS THE SPATIAL2 ARRANGEMENT OF ATOMS INSIDE A 
    CHEMICAL_MOLECULE"
    RELATIVE_POSITION = "IS THE POSITION OF AN OBJECT3 OR POINT WITHIN RELATION TO 
    ANOTHER OBJECT3 OR POINT AND IS THE MEASURE OF THE DISTANCE BETWEEN TWO 
    OBJECTS3 OR POINTS ARE AND IN WHAT DIRECTION THEY ARE FROM EACH OTHER"
    PERCENTAGE2 = "IS A NUMBER OR RATE THAT IS USED TO EXPRESS A CERTAIN PART OF 
    SOMETHING AS A SPECIFIC AMOUNT OF SOMETHING WHOLE WITH A SPECIFIC NUMBER 
    AS A VALUE"
    CHEMICAL = "A COMPOUND OR SUBSTANCE THAT HAS BEEN PREPARED"
    ENUMERATION_ELEMENTS = "MORE THAN ONE ENUMERATION_ELEMENT"
    CHEMICAL_STRUCTURES = "MORE THAN ONE CHEMICAL_STRUCTURE"
    QUANTUM_TECHNOLOGY = "IS A FIELD WITHIN QUANTUM_MECHANICS THAT USES THE 
    CAPABILITIES OF THE PRINCIPLES THAT MAKE QUANTUM_MECHANICS TO MAKE NEW 
    SOFTWARE AND OR HARDWARE AND OR PROGRAMS"
    ANGLE = "IS WHEN TWO LINES MEET AT A POINT AND IS MEASURED BY THE AMOUNT OF 
    TURN BETWEEN THE TWO LINES"
    ANGLES = "MORE THAN ONE ANGLE"
    ROTATIONAL_DEGREE = "A UNIT OF MEASURE FOR ANGLES TO MEASURE THE AMOUNT 
    OF ROTATION OF AN OBJECT3 ABOUT A POINT THAT IS NOT ABLE TO BE CHANGED"
    VECTOR_TRACING = "IS THE PROCESS OF CREATING A VECTOR GRAPHIC FROM AN 
    EXISTING IMAGE"
    TRLINEAR_INJECTION = "A TECHNIQUE FOR INSERTING DATA2 INTO A 
    THREEDIMENSIONAL TEXTURE AND IS MADE POSSIBLE BY TAKING THREE VECTORS EACH 
    REPRESENTING A POINT ON THE TEXTURE AND USING THEM TO CALCULATE THE 
    PROGRAMMING_VALUE OF ANOTHER POINT ON THE TEXTURE"
    TRLINEAR_VECTOR_SPACE = "IS A VECTOR_SPACE THAT HAS A THREE PATHWAY 
    FUNCTION THAT TAKES THREE VECTORS AND GIVES A SCALAR PROGRAMMING_VALUE 
    AS A RETURN"
    SPATIAL_COGNITION = "THE ABILITY TO MENTALLY MANIPULATE SPATIAL2 INFORMATION 
    AND OR STORE AND GATHER SPATIAL DATA2 WITHIN MEMORY AND OR THE ABILITY TO 
    USE SPATIAL INFORMATION TO SOLVE PROBLEMS AND OR THE ABILITY TO DETECT AND 
    INTERPRET SPATIAL INFORMATION FROM THE ENVIRONMENT"
    SPATIAL_MAPPING = "IS THE PROCESS OF REPRESENTING THE SPATIAL RELATION 
    BETWEEN OBJECTS3 OR FEATURES INSIDE A SPECIFIC LOCATION AND OR REGION AND 
    OR AREA AND OR SPACE2"
    LOCATION_OF_OBJECTS = "CAN BE THE PHYSICAL2 LOCATION OF OBJECTS3 WITHIN THE 
    DOMAIN OF A SPECIFIC LOCATION"
    SPATIAL_RELATIONSHIPS_BETWEEN_OBJECTS = "CAN BE THE DISTANCE BETWEEN 
    OBJECTS3 AS WELL AS THE DIRECTION BETWEEN OBJECTS3"
    SPATIAL_PROPERTIES_OF_OBJECTS = "CAN BE THE SHAPE OF AN OBJECT3 AND OR THE 
    SIZE OF AN OBJECT3 AND OR COLOR OF AN OBJECT3"
    SPATIAL_ANALYSIS = "CAN BE USED TO ANALYZE THE SPATIAL2 RELATION AND 
    CONNECTION BETWEEN OBJECTS3"
    MENTAL_IMAGERY = "IS THE ABILITY TO CREATE A MENTAL REPRESENTATION OF AN 
    OBJECT3 OR EVENT OR CIRCUMSTANCE AND IS A TYPE OF VISUAL REPRESENTATION 
    THAT ALLOWS THINGS TO BE SEEN WITHIN THE VISION OF THE MIND"
    SOLUTIONS = "MORE THAN ONE SOLUTION"
    MENTAL_IMAGERY_WITH_PROBLEM_SOLVING = "CAN BE USED TO VISUALIZE POSSIBLE 
    SOLUTIONS TO PROBLEMS"
    MENTAL_IMAGERY_WITH_CREATIVITY = "CAN BE USED TO GENERATE NEW IDEAS"
    MENTAL_IMAGERY_WITH_LEARNING = "CAN BE USED TO IMPROVE MEMORY AND RECALL 
    INFORMATION"
    MENTAL_IMAGERY_WITH_ENGINEERING = "CAN BE USED TO VISUALIZE THE DESIGN OF 
    OBJECTS3 AND STRUCTURES AND SYSTEMS"
    MENTAL_IMAGERY_WITH_NAVIGATING = "CAN BE USED TO VISUALIZE THE LAYOUT OF A 
    SPACE"
    FIRMWARE = "IS A TYPE OF SOFTWARE STORED INSIDE A HARDWARE DEVICE AND IS 
    USED TO CONTROL THE DEVICE BASIC FUNCTIONS INCLUDING ENERGY MANAGEMENT 
    AND INPUT AND OR OUTPUT CONTROL AND MANAGEMENT AND CONTROL AS WELL AS 
    COMMUNICATION BETWEEN THE SYSTEM HARDWARE AND THE SYSTEM SOFTWARE AND 
    IS STORED WITHIN READ ONLY MEMORY AND CANNOT BE MODIFIED BY THE USER"
    PRACTICES = "MORE THAN ONE PRACTICE"
    PRACTICE = "A BELIEF GIVEN ACTION TO HAPPEN OR TO COME INTO EFFECT BY CHANCE 
    AND OR POSSIBILITY"
    PAST_TENSE = "REFERENCE TO A PAST TENSE THAT IS USED TO REFERENCE A TIME OF 
    ACTION"
    PAST_PARTICIPLE = "REFERENCE TO A SPECIFIC VERB FORM THAT IS USED IN THE PAST"
    TENSE = "FORM OF VERB SYSTEM THAT IS USED TO SHOW TIME AND OR CONTINUATION 
    AND OR COMPLETION OF AN ACTION"
    FUTURE_TENSE = "REFERENCE TO A FUTURE TENSE THAT IS USED TO REFERENCE A TIME 
    OF ACTION"
    FUTURE_PARTICIPLE = "REFERENCE TO A SPECIFIC VERB FORM THAT IS USED IN THE 
    FUTURE"
    class Language_Extension_003_2: 
    ZSECOND = 3
    ZMILLISECOND = .03
    ZMICROSECOND = .0003
    ZNANOSECOND = .000003
    ZMILLISECOND = ZSECOND * 100
    ZMICROSECOND = ZSECOND * 10000
    ZNANOSECOND = ZSECOND * 1000000
    ZMINUTE = ZSECOND * 20
    ZMICROMINUTE = ZSECOND * 10
    ZHOUR = ZMINUTE * 60
    ZDAY = ZHOUR * 30
    ZWEEK = ZDAY * 6
    ZMONTH = ZWEEK * 5
    ZYEAR = ZMONTH * 12
    ZDECADE = ZYEAR * 10
    ZCENTURY = ZDECADE * 10
    ZMILLENIA = ZCENTURY * 10
    ZNANOMETER = .000001
    ZMICROMETER = .0001
    ZMILLIMETER = .01
    ZMETER = 1000
    ZCENTIMETER = 100 
    ZINCH = 10
    ZFOOT = 25
    ZYARD = 30
    ZKILOMETER = 1250
    ZMILE = 2500
    ZHERTZ = .0001
    ZKILOHERTZ = ZHERTZ * 1000
    ZMEGAHERTZ = ZKILOHERTZ * 1000
    ZGIGAHERTZ = ZMEGAHERTZ * 1000
    ZTERAHERTZ = ZGIGAHERTZ * 1000
    ZNIBBLE = 30
    ZBIT = 65
    ZBYTE = ZBIT * 1000 
    ZKILOBIT = ZBIT * 1000 
    ZMEGABIT = ZKILOBIT * 1000
    ZGIGABIT = ZMEGABIT * 1000
    ZTERABIT = ZGIGABIT * 1000
    ZKILOBYTE = ZBYTE * 1000
    ZMEGABYTE = ZKILOBYTE * 1000
    ZGIGABYTE = ZMEGABYTE * 1000
    ZTERABYTE = ZGIGABYTE * 1000
    ZFARENHEIGHT = 30
    ZCELCIUS = 10
    ZKELVIN = ZCELCIUS * 150
    ZTEASPOON = .5
    ZTABLESPOON = ZTEASPOON * 2
    ZPINT = ZTABLESPOON * 6
    ZCUP = ZPINT * 2
    ZQUART = ZCUP * 4
    ZLITER = ZQUART * 4
    ZGALLON = ZQUART * 16 
    class Language_Extension_004_2:
    ABILITY_SYSTEM = "IS AN FRAMEWORK THAT ALLOWS PLAYERS TO USE A VARIETY OF 
    ABILITIES"
    FIXED_CALIBRATED_RATIO = "IS A RATIO OF TWO MEASUREMENTS THAT HAS BEEN 
    DETERMINED TO BE CONSTANT"
    FILE_EXTENSION = "IS A SEQUENCE OF LETTERS AT THE END OF A FILE NAME SEPARATED 
    FROM THE MAIN FILE NAME BY A SYMBOLE"
    EMOTIONAL_FUSION = "IS A TERM USED TO DESCRIBE A TYPE OF RELATION THAT WHICH 
    TWO ENTITIES HAVE BEEN STRONGLY CONNECTED BY EMOTIONS THAT THEY BECOME 
    ONE"
    EMOTIONAL_SYNERGY = "IS THE INTERACTION OF TWO OR MORE ENTITIES EMOTIONS TO 
    PRODUCE A COMBINED EFFECT GREATER THAN THE ADDITION OF THEIR INDIVIDUAL 
    EFFECTS"
    EMOTIONAL_ENLIGHTENMENT = "IS A STATE OF EXISTING THAT WHICH THE ENTITY IS 
    FULLY AWARE OF THEIR EMOTIONS AND HOW THE EMOTIONS AFFECT THE ENTITY"
    EMOTIONAL_CLARITY = "IS THE ABILITY TO RECOGNIZE AND LOCATE AS WELL AS 
    UNDERSTAND AND EXPRESS EMOTIONS WITH PERFECT ACCURACY"
    AURIC_RESPONSE = "IS A REACTION FROM KINETIC ENERGIES FROM A ENTITY"
    BOOK_OF_KNOWLEDGE = "IS A REFERENCE OF CREATED WORK THAT CONTAINS A 
    COLLECTION OF INFORMATION ON A PARTICULAR SUBJECT OR FIELD OF STUDY"
    ARCANE_ENERGY = "IS A TYPE OF MAGICAL ENERGY THAT IS DESCRIBED IN MANY 
    OCCASIONS THAT DOES INVOLVE MAGICAL BEINGS AND OR EXISTENCES"
    QUANTUM_ENERGY = "IS A TERM USED TO DESCRIBE THE ENERGY OF SUBATOMIC 
    PARTICLES"
    RECURSIVE_LANGUAGE = "IS A LANGUAGE THAT CAN BE DESCRIBED USING ITSELF"
    RECURSIVE_MEMORY = "IS A TYPE OF MEMORY THAT ALLOWS THE ENTITY TO STORE AND 
    RECALL INFORMATION BY USING DIV TO DIVIDE THE INFORMATION INTO SMALL AND 
    SMALLER PIECES UNTIL IT IS UNDERSTOOD"
    RECURSIVE_RECALL_PROCESS = "IS A PROCESS OF GATHERING INFORMATION FROM 
    MEMORY USING A REPEATED METHOD TO DIV THE INFORMATION INTO SMALLER AND 
    SMALLER PIECES"
    SEASONAL_BRAINWAVE_FLUX = "IS A TERM USED TO DESCRIBE THE CHANGES WITH 
    BRAINWAVE RESPONSES THAT HAPPEN WITHIN A ZYEAR TIMEFRAME"
    QUANTUM_FORCES = "ARE THE FOUR FUNDAMENTAL FORCES OF NATURE THAT ACT AT 
    THE SUBATOMIC LEVEL2"
    PRIMAL_ENERGY = "IS THE ENERGY FOUND WITHIN ALL LIVING THINGS"
    QUALITIVE_RESEARCH = "IS A STUDY METHOD THAT GATHERS AND USES METHOD TO 
    ANALYZE DATA2 THAT IS NOT USING ANY NUMBERS"
    QUANTITIVE_RESEARCH = "IS A STUDY METHOD THAT HAS FOCUS ON GIVING A SPECIFIC 
    CATEGORY OF THE COLLECTION OF AND ANALYSIS OF DATA2 AND IS FORMED ON 
    THEORY OR NUMBERS"
    QUANTUM_METER = "IS A DEVICE THAT MEASURES THE SETTINGS AND VALUES OF 
    QUANTUM SYSTEMS"
    PRE_GENERATION_COMPLETION_TIME = "IS THE TIME IT TAKES TO GENERATE ALL OF THE 
    NECESSARY DATA2 FOR SOMETHING TO GENERATE"
    PRE_GENERATION_WAIT_TIME = "IS THE TIME IT TAKES FOR SOMETHING TO BE 
    GENERATED BEFORE THE GENERATION CAN HAPPEN"
    META_DATA = "IS DATA2 THAT IS USED TO DESCRIBE OTHER DATA2"
    REGULAR_EXPRESSION = "IS A SEQUENCE OF LETTERS THAT SPECIFIES A LOCATED 
    PATTERN IN WORDS OR CODE"
    TEXTURE_MAP = "IS AN IMAGE THAT IS APPLIED TO A THREEDIMENSIONAL OBJECT3 TO 
    GIVE IT A SURFACE APPEARANCE"
    class Language_Extension_005_2:
    RELAY = "SENDS AND RE SENDS SIGNALS FROM TWO DIFFERENT LOCATIONS"
    IMMERSION = "BECOMING ABSOLUTELY INFLUENCED INTO SOMETHING"
    IMMERSIVE = "TO ABSORB INTO A SPECIFIC CATEGORY OR GENRE WITH DETAILED AND 
    UNDERSTANDING INTO THE FIELD OF INTEREST"
    HYPERVISUAL_DISPLAY_UNIT = "IS A TYPE OF DISPLAY THAT ALLOWS A USER TO 
    CONNECT WITH INFORMATION IN A IMMERSIVE WAY"
    EMOTIONAL_THROUGHPUT = "REFERS TO THE RATE AT WHICH A USER HAS AND IS 
    EXPRESSING EMOTIONS"
    EMOTIONAL_MAGNITUDE = "REFERS TO THE INTENSITY OF THE EMOTIONS THAT A USER 
    CAN FEEL"
    EMOTIONAL_ENERGY = "IS THE ENERGY CREATED BY THE EMOTIONS"
    EMOTIONAL_UNDERSTANDING = "IS THE ABILITY TO RECOGNIZE AND OR GIVE 
    IDENTIFICATION AND UNDERSTAND THE EMOTIONS THAT EXIST FOR A SINGLE USER AS 
    WELL AS THE EMOTIONS OF OTHER USERS"
    HOLOGRAPHIC_DISPLAY_UNIT = "IS A TYPE OF DISPLAY THAT USES MANY HOLOGRAMS 
    OF HOLOGRAPHIC IMAGE FILES AND INFORMATION ONTO A HOLOGRAPHIC DISPLAY 
    LOCATION"
    HOLOGRAPHIC_HYPERVISOR = "IS SOMETHING THAT USES HOLOGRAPHIC SYSTEMS TO 
    MAKE VIRTUAL SYSTEMS AND OR DEVICES"
    DATALAKE = "IS A CENTRAL LOCATION FOR ALL DATA2 WITHIN A SPECIFIC LOCATION 
    THAT IS BOTH STRUCTURED AND NOT STRUCTURED WITHIN THE LOCATION AND CAN 
    STORE ANY TYPE OF DATA2 OF ANY SIZE OR SOURCE"
    HOLOGRAPHIC_DATALAKE = "IS A TYPE OF DATALAKE THAT USES HOLOGRAPHIC 
    SYSTEMS OR DEVICES TO STORE AND MANAGE DATA2"
    HOLOGRAPHIC_FREQUENCY_ANALYZER = "IS A TYPE OF WAVELENGTH ANALYZER THAT 
    USES HOLOGRAPHIC SYSTEMS TO MEASURE THE FREQUENCY WAVELENGTH OF A 
    SIGNAL"
    HOLOGRAPHIC_WAVELENGTH = "IS THE WAVELENGTH OF LIGHT THAT IS USED TO 
    CREATE A HOLOGRAM"
    #
    ADHOC_BARRIER_SYSTEM = "IS A TYPE OF SYSTEM THAT CAN BE USED TO PREVENT 
    DEVICES FROM ALLOWING COMMUNICATION WITH EACH OTHER INSIDE AN 
    ADHOCNETWORK"
    ADHOC_DATA_PROCESSING_DEEP_LEARNING = "IS THE USE OF DEEP_LEARNING 
    TECHNIQUES TO PROCESS DATA2 WITHIN ADHOC DEVICES WITHIN TIME2"
    ADHOC_DATA_PROCESSING = "IS THE PROCESSING OF GATHERED AND ORGANIZED 
    AND ANALYZED DATA2 WITHIN AN ADHOC DEVICE AND OR SYSTEM"
    ADHOC_DELAY_HANDLER = "IS A SOFTWARE COMPONENT THAT IS USED TO DELAY 
    DATA2 COMMUNICATION BETWEEN DEVICES WITHIN AN ADHOC SYSTEM"
    ADHOC_EDGE_COMPUTING_DEEP_LEARNING_NETWORK = "IS A NETWORK OF DEVICES 
    THAT ARE ABLE TO PERFORM DEEP_LEARNING TASKS AT THE BORDER OF THE NETWORK"
    ADHOC_ENCRYPTION_HANDLER = "IS A SOFTWARE COMPONENT THAT IS TO ENCRYPT 
    AND DECRYPT DATA2 WITHIN AN ADHOC SYSTEM"
    ADHOC_EXTENSION_FILES = "ARE FILES THAT ARE USED TO EXTEND THE FUNCTIONING 
    ASPECTS OF A SOFTWARE PROGRAM WITHIN AN ADHOC SYSTEM"
    ADHOC_FREQUENCY_ENCRYPTOR = "IS A DEVICE THAT CAN ENCRYPT DATA2 BEFORE IT 
    IS SENT COMMUNICATION INSIDE AN ADHOC SYSTEM"
    ADHOC_FREQUENCY_PREREQUISITE_COMPATIBILITY_SYSTEM = "IS A SYSTEM THAT IS TO 
    CHECK AND DETERMINE IF ADHOC DEVICES AND ADHOC NETWORKS CAN USE 
    COMMUNICATION BETWEEN EACH KNOWN SYSTEM WITHIN THE LOCATION THAT 
    REQUEST ACCESS BY USING COMPATIBLE FREQUENCIES"
    ADHOC_GEOFENCE = "IS A VIRTUAL BOUNDARY THAT IS CREATED WITHIN A LOCATION"
    ADHOC_GEOLOCATION_FIELD_PARAMETER = "IS USED TO DETERMINE THE LOCATION 
    OF DEVICES WITHIN THE ADHOC NETWORK"
    ADHOC_IDE_INTERFACE = "IS A SOFTWARE PROGRAM THAT ALLOWS USERS TO DEVELOP 
    AND FIND MANY ERROR VALUES INSIDE SOFTWARE PROGRAMS WITHIN AN ADHOC 
    NETWORK OR MANY ADHOC NETWORKS"
    ADHOC_INPUT_PATH = "ARE A TYPE OF INPUT PATH THAT ARE USED TO ENTER TEXT OR 
    USED TO SELECT OBJECTS FROM A LIST"
    ADHOC_INSTALLED_PROGRAMS = "ARE PROGRAMS THAT ARE EXISTING ON AN ADHOC 
    COMPUTER AND OR DEVICES OR SYSTEMS"
    ADHOC_LOCAL_AREA_NETWORK = "IS A TYPE OF NETWORK THAT IS CREATED BY 
    CONNECTING TWO OR MORE ADHOC DEVICES TOGETHER"
    ADHOC_LOCATION_HANDLER = "IS A SOFTWARE COMPONENT THAT IS USED TO LOCATE 
    AND FIND THE LOCATION OF ADHOC SYSTEMS WITHIN A ADHOC NETWORK"
    ADHOC_OUTPUT_PATH = "IS AN ADHOC LOCATION WHERE OUTPUT IS STORED"
    ADHOC_PARAMETER_PREREQUISITE_RECOGNITION = "IS THE PROCESS OF 
    DETERMINING THE POSSIBILITY OF TWO OR MORE ADHOC DEVICES HAVING THE 
    REQUIRED PARAMETERS TO ALLOW COMMUNICATE WITH EACH DEVICE"
    ADHOC_EXTENSIONS = "ARE EXTENSIONS THAT ARE NOT A PART OF THE NORMAL 
    PROGRAM OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_PATHS = "IS A PATH THAT IS NOT PART OF THE STANDARD 
    DOWNLOAD OF A PROGRAM"
    ADHOC_RECOGNIZED_PATH_DATA = "IS A TYPE OF DATA2 THAT IS USED TO EXTEND THE 
    FUNCTIONING ASPECTS OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_PROFILE_PATH = "IS A PATH NOT PART OF THE STANDARD 
    SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_UI_PATHS = "ARE PATHS THAT ARE NOT PART OF THE STANDARD 
    USERINTERFACE OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_VARIABLE_PATHS = "CAN BE RECOGNIZED AS THE PATH 
    CONTENT AND THE NUMBER OF PATHS THAT EXIST AS VARIABLES WITHIN AN ADHOC 
    SYSTEM"
    ADHOC_SHORT_DISTANCE_RELAY_HANDLER = "IS A SOFTWARE COMPONENT THAT IS 
    USED TO RELAY DATA2 BETWEEN DEVICES INSIDE AN ADHOC SYSTEM"
    ADHOC_TRANSPARENCY_FEEDBACK = "IS A PROCESS OF PROVIDING A RESPONSE TO 
    USERS ABOUT THE DECISIONS THAT ARE MADE BY AN ADHOC SYSTEM OR ADHOC 
    SYSTEMS"
    ADHOC_VIRTUAL_DIGITAL_SYSTEM = "IS A SOFTWARE THAT CREATES A VIRTUAL BARRIER 
    BETWEEN TWO OR MORE DEVICES"
    ADHOC_VOICE_COMMAND_FILE_SYSTEM_ACTIVATOR = "IS A SOFTWARE PROGRAM THAT 
    ALLOWS USERS TO CONTROL ADHOC DEVICES USING VOICE COMMAND"
    PHRASE = "A SEQUENCE OF WORDS THAT MAKES A SENTENCE PATTERN FROM 
    RECOGNIZED WORDS INSIDE THE SENTENCE"
    ADHOC_VOICE_COMMAND_PHRASE_INPUT = "IS A PHRASE THAT IS USED TO CONTROL 
    AN ADHOC DEVICE USING VOICE COMMAND"
    ADHOC_VOICE_COMMAND_SYSTEM = "IS AN SYSTEM THAT ALLOWS AN USER TO 
    CONTROL ADHOC DEVICES USING VOICE COMMAND"
    WIRELESS_POWER_TRANSFER = "IS A DEVICE THAT ALLOWS FOR THE TRANSFER OF 
    ENERGY BETWEEN TWO DEVICES WITHOUT THE REQUIRE FOR PHYSICAL INTERACTION"
    ADHOC_WIRELESS_CHARGING_STATION = "IS A DEVICE THAT USES 
    WIRELESS_POWER_TRANSFER TO CHARGE WIRELESS DEVICES"
    ADHOC_VOICE_COMMAND_ORIGIN_POINT_INDICATOR = "IS A SYSTEM THAT ALLOWS AN 
    USER TO RECOGNIZE THE SOURCE OF A VOICE COMMAND"
    NEURAL_NETWORK = "IS A TYPE OF ALGORITHM THAT CAN BE USED TO LEARN AND 
    ADAPT TO CHANGING CONDITIONS"
    NEURAL_NETWORKS = "MORE THAN ONE NEURAL_NETWORK"
    ARTIFICIAL_NEURAL_NETWORK_HOLOGRAPHIC_ADHOC_NETWORK = "IS A TYPE OF 
    WIRELESS NETWORK THAT USES ARTIFICIAL NEURAL_NETWORKS TO CONTROL THE 
    NETWORK ROUTE"
    BARYON_FREQUENCY_CHAMBER = "IS AN DEVICE THAT CAN BE USED TO IMPROVE THE 
    QUALITY AND PERFORMING FUNCTIONS OF SERVICE RECOGNITION SYSTEMS WITHIN 
    AN ADHOC NETWORK"
    #
    ARTIFICIAL_GENERAL_INTELLIGENCE = "IS A TYPE OF ARTIFICIAL INTELLIGENCE2 THAT 
    HAS THE ABILITY TO PERFORM ANY TASK INTELLIGENTLY"
    ARTIFICIAL_SUPER_INTELLIGENCE = "IS A TYPE OF ARTIFICIAL INTELLIGENCE THAT CAN 
    HAVE THE ABILITY TO PERFORM ANY TASK WITH GREATER QUALITY AND EFFICIENT SKILL 
    THAN AN ARTIFICIAL_GENERAL_INTELLIGENCE"
    #
    AURA_RESPONSE_RECOGNITION_PERSONALITY_READER = "IS AN DEVICE THAT CAN 
    READ THE AURA OF A PERSON AND RECOGNIZE WITH GREAT QUALITY THE PERSONALITY 
    TRAITS OF A PERSON"
    #
    COMPLEXITY_BREAKDOWN_BY_SIMPLE_STANDARDS = "IS A PROCESS OF BREAKING 
    DIFFICULT SYSTEMS INTO SMALLER MORE ABLE TO BE MANAGED PARTS AND 
    IDENTIFYING THE DIFFERENT COMPONENTS OF THE SYSTEM AND THE INTERACTION 
    THAT HAPPENS WITHIN THE SYSTEM"
    CYBER_FLUX_FORESIGHT = "IS THE ABILITY TO ANALYZE AND PREPARE FOR THE FUTURE 
    EVENTS"
    DATA_RECOGNITION_MAGNITUDE = "IS A MEASURE OF THE ACCURACY OF AN ARTIFICIAL 
    INTELLIGENCE SYSTEM AND THE ABILITY TO RECOGNIZE DATA2"
    DATA_RECOGNITION_THRESHHOLD = "IS THE SMALLEST LEVEL OF ACCURACY WITH 
    POSSIBILITY THAT AN ARTIFICIAL INTELLIGENCE SYSTEM DOES REQUIRE BEFORE IT CAN 
    RECOGNIZE DATA2 WITH CONFIDENT MEASUREMENTS"
    DOCUMENT_RECOGNITION_BY_CONTENTS_WITHIN_THE_DOCUMENT = "IS A PROCESS 
    TO ANALYZE AND DECRYPT INFORMATION FROM A DOCUMENT BY ALLOWING TO 
    REFERENCE THE INFORMATION DETERMINED BY THE DOCUMENT CONTENT"
    EMOTIONAL_ESSENSE = "IS THE ADDED TOTAL OF ALL EMOTIONS WITHIN AN INDIVIDUAL 
    AND IS WHAT MAKES EACH ENTITY UNIQUE AND WHAT ALLOWS ENTITIES TO CONNECT 
    WITH OTHERS WITH A MORE UNDERSTOOD LEVEL"
    EMOTIONAL_PRESENCE = "IS THE ABILITY TO BE FULLY PRESENT IN THE EVENT AND TO 
    BE AWARE OF YOUR OWN AND OTHER EMOTIONS FROM OTHER PEOPLE"
    FREQUENCY_NETWORK_CONDITIONER = "IS A DEVICE THAT CAN BE USED TO IMPROVE 
    THE PERFORMING OF WIRELESS NETWORKS AND FUNCTION BY GIVING AN ATTEMPT TO 
    FILTER OUT NOT NEEDED FREQUENCIES AND INCREASING THE INTENSITY OF THE 
    DESIRED FREQUENCIES"
    FREQUENCY_PATTERN_OF_DATA = "IS THE WAY IN WHICH THE DATA2 VALUES ARE 
    DISTRIBUTED AND CAN BE USED FOR IDENTIFYING PATTERNS WITHIN THE DATA2"
    FREQUENCY_PROTOCOL_RATIO_CALIBRATOR = "IS A DEVICE THAT CAN BE USED TO 
    MEASURE AND ADJUST THE FREQUENCY RATIO OF DIFFERENT CONDITIONS AND 
    COMMANDS INSIDE AN ADHOC SYSTEM"
    FREQUENCY_STABILIZER_SERVICE = "IS A DEVICE THAT CAN BE USED TO IMPROVE THE 
    STABILITY OF WIRELESS ADHOC NETWORKS AND FUNCTIONS BY SEARCHING 
    RECOGNIZED FLAW DATA2 AND HAVING THE DATA2 CORRECTED TO THE REQUIRED 
    CALIBRATION MEASUREMENT NEEDED"
    GAME = "CAN INCLUDE SIMULATION OR RE THE SIMULATION OF VARIOUS EVENTS OR 
    USE AND WITHIN REALITY FOR VARIOUS REQUIREMENTS AND OR OF NEEDS"
    #TWO_POINT_FIVE_DIMENSION_TERRAIN_BRUSH = "IS A TOOL THAT ALLOWS THE USER 
    TO CREATE TWO_POINT_FIVE_DIMENSIONAL TERRAIN WITHIN A ENGINE
    #TWO_POINT_FIVE_DIMENSIONAL_TERRAIN = "IS A TYPE OF TERRAIN THAT IS DISPLAYED 
    WITH TWO_DIMENSIONAL ELEMENTS BUT HAS SOME THREEDIMENSIONAL ELEMENTS
    #WORLDBUILDINGNIGHTTIME = "A TIMEFRAME THAT BEGINS WITHIN THE EVENING 
    HOURS AND DOES CONTINUE UNTIL THE MORNING TIMEFRAME WHICH IS A 
    WORLDBUILDINGDAYTIME TIMEFRAME
    #WORLDBUILDINGDAYTIME = "A TIMEFRAME OF TIME_OF_DAY THAT BEGINS WITHIN THE 
    HOURS OF MORNINGTIMEFRAME AND DOES CONTINUE UNTIL EVENING TIMEFRAME 
    WHICH IS A WORLDBUILDINGNIGHTTIME TIMEFRAME
    #WORLDBUILDINGSEASON = "A TIMEFRAME THAT IS CLASSIFIED INTO FOUR 
    CATEGORIES THAT HAS DIFFERENT TEMPERATURE FOR THE ENVIRONMENT AND 
    DIFFERENT CONDITIONS FOR THE INDIVIDUAL SEASON WITH EACH SEASON KNOWN AS 
    WINTER AND SUMMER AND SPRING AND AUTUMN
    #WORLDBUILDINGMOONPHASE = "IS THE DIFFERENT CHANGES WITH THE MOON AND 
    ITS PHASE CHANGES BETWEEN THE EIGHT MOON PHASES
    #WORLDBUILDINGPLANET = "
    #MORNINGTIMEFRAME = "A TIMEFRAME THAT BEGINS AT .20833 AND DOES END 
    AT .75000
    #NIGHTTIMEFRAME = "A TIMEFRAME THAT BEGINS AT .75000 AND DOES END AT .20833
    class Language_Extension_006_2:
    QUICK = "TO FORM ACTION WITH GREAT SPEED"
    SLOW = "TO FORM ACTION WITH A SMALL AMOUNT OF SPEED"
    FAST = "QUICK AT FORMING A REACTION OR MAKING AN ACTION"
    CIRCUMSTANCES = "MORE THAN ONE CIRCUMSTANCE"
    FOCUSING = "MAKING AN ACTION TO FOCUS AND CONCENTRATE ON A SPECIFIC TOPIC 
    AND OR FIELD AND OR AREA OF FOCUS"
    SIMILARITY = "IS A THING OR IDEA2 THAT TWO OR MORE THINGS HAVE AS THE SAME OR 
    SIMILAR TO"
    COMPARISON = "TO COMPARE TWO OR MORE THINGS TOGETHER"
    SIMILARITIES = "MORE THAN ONE SIMILARITY OR COMPARISON"
    COMPARING = "IS THE ACTION OF IDENTIFYING THE SIMILARITIES AND DIFFERENCES 
    BETWEEN TWO OR MORE THINGS"
    REPRESENTED = "EXPLAINED AS OR DETERMINED AS"
    CONVERTING = "PROCESSING AS AN ACTION TO CONVERT"
    INTERSECT = "WHEN TWO LINES ARRIVE AT OR GO THROUGH A SINGLE ORIGIN 
    LOCATION" 
    ZERO_TO_ONE_HUNDRED_PERCENTAGE = "AN AMOUNT BETWEEN ZERO AND 
    ONEHUNDRED PERCENT"
    DEGREE = "AN AMOUNT BETWEEN ZERO AND THREEHUNDREDSIXTY"
    DEGREES = "MORE THAN ONE DEGREE"
    RIGHT_ANGLE = "IS AN ANGLE OF NINETY DEGREES"
    PERPENDICULAR = "IS TWO THINGS THAT INTERSECT AT A RIGHT_ANGLE" 
    RECOGNIZING = "BEGINNING TO RECOGNIZE SOMETHING OR SOMEONE"
    IDENTIFY = "TO LOCATE AND RECOGNIZE"
    ALIGNING = "MAKING AN ACTION TO ALIGN SOMETHING"
    ALIGNMENT = "IS THE ACTION OF ALIGNING THINGS IN A STRAIGHT LINE OR IN A 
    CERTAIN DIRECTION"
    OBSERVED = "ANALYZED WHILE LOCATED"
    CLASSIFYING = "TO GIVE A CATEGORY AND THEN GROUP TOGETHER SPECIFIC DATA2"
    PAIR = "A TOTAL OF TWO OF SOMETHING"
    FIXED = "CANNOT BE CHANGED"
    DISTANCES = "MORE THAN ONE DISTANCE"
    MULTIDIMENSIONAL_SPACE = "IS A SPACE2 THAT HAS MORE THAN TWO DIMENSIONS" 
    GRID = "IS A DATA2 STRUCTURE THAT IS USED TO RECOGNIZE DATA2 THAT IS ORGANIZED 
    IN A MULTIDIMENSIONAL_SPACE"
    ADHOC_CARDINAL_ARRAY = "IS A DATA2 STRUCTURE THAT STORES A COLLECTION OF 
    ELEMENTS IN A SORTED ORDER"
    ADHOC_EUCLIDEAN_GRID = "IS A DATA2 STRUCTURE THAT STORES A COLLECTION OF 
    POINTS IN A TWO DIMENSIONAL SPACE2"
    NUMERICAL = "RELATING TO OR EXPRESSED IN NUMBERS"
    CARTESIAN_COORDINATE_SYSTEM = "IS A COORDINATE SYSTEM THAT SPECIFIES EACH 
    POINT IN A UNIQUE WAY BY A PAIR OF NUMERICAL COORDINATES WHICH ARE THE 
    ASSIGNED DISTANCES FROM THE POINT TO TWO FIXED PERPENDICULAR LINES"
    CARTESIAN_COORDINATE = "IS A PAIR OF NUMBERS THAT IN A UNIQUE WAY CAN 
    SEARCH FOR TO FIND A POINT WITHIN A PLANE"
    GEOGRAPHIC = "REFERS TO ANYTHING SIMILAR TO THE PHYSICAL FEATURES OF THE 
    ENVIRONMENT AND ITS SURFACE"
    GEOFENCE = "IS A VIRTUAL PERIMETER AROUND A GEOGRAPHIC AREA THAT CAN BE 
    USED TO ANALYZE THE LOCATIONS THE MOVEMENT OF PEOPLE OR OBJECTS" 
    LATITUDE = "IS A GEOGRAPHIC COORDINATE THAT SPECIFIES THE UPWARD OR 
    DOWNWARD POSITION OF A POINT ON THE ENVIRONMENTS SURFACE"
    LONGITUDE = "IS A GEOGRAPHIC COORDINATE THAT SPECIFIES THE RIGHT OR LEFT 
    POSITION OF A POINT ON THE ENVIRONMENT AND ITS SURFACE"
    CARTESIAN_COORDINATES = "MORE THAN ONE CARTESIAN_COORDINATE"
    STATISTICAL = "REFERS TO ANYTHING SIMILAR TO THE COLLECTION OF OR ANALYSIS OF 
    OR UNDERSTANDING AND COMPREHENDING OF AND PRESENTATION OF DATA2 AND 
    CAN ALSO REFER TO THE METHODS USED TO DO SUCH2 ACTIONS"
    EUCLIDEAN_GRID = "IS A TYPE OF GRID THAT IS USED TO COMPARE OR ASSIGN DATA2 
    THAT IS IN A MULTIDIMENSIONAL_SPACE"
    ADHOC_CARTESIAN_POINT = "IS AN EUCLIDEAN_GRID IS A POINT IN A TWO 
    DIMENSIONAL SPACE2 THAT IS REPRESENTED BY ITS COORDINATES IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_MINIMUM_RATIO_EXCHANGE_BETWEEN_CARTESIAN_VALUES = "IS AN 
    ALGORITHM THAT CAN BE USED TO FIND THE MINIMUM RATIO BETWEEN TWO POINTS IN 
    A CARTESIAN_COORDINATE_SYSTEM"
    CARTESIAN_ADHOC_NETWORK = "IS A TYPE OF WIRELESS NETWORK THAT IS CREATED 
    BY DEVICES THAT ARE CONNECTED TO EACH OTHER IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_CARTESIAN_GEOFENCE = "IS A TYPE OF GEOFENCE THAT IS DEFINED IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_OFFLINE_GPS_CARTESIAN_COORDINATE_SYSTEM_USING_LATITUDE_AND_LON
    GITUDE = "IS A SYSTEM THAT CAN BE USED TO DEFINE COORDINATES IN A 
    CARTESIAN_COORDINATE_SYSTEM AND IS DONE BY FIRST CONVERTING THE LATITUDE 
    AND LONGITUDE COORDINATES TO CARTESIAN_COORDINATES USING A FORMULA"
    ADHOC_CARTESIAN_GAME_WORLD = "IS A GAME WORLD THAT IS DEFINED USING A 
    CARTESIAN_COORDINATE_SYSTEM"
    FREQUENCY_ADJUSTMENT_DIAGNOSTICS = "IS A STATISTICAL TECHNIQUE THAT CAN BE 
    USED TO IDENTIFY AND CORRECT PROBLEMS WITH FREQUENCY DATA2 AND FUNCTIONS 
    BY COMPARING THE OBSERVED FREQUENCIES OF DATA2 POINTS TO THE EXPECTED 
    FREQUENCIES"
    FREQUENCY_PATTERN = "IS A WAY OF DESCRIBING THE DISTRIBUTION OF DATA2 POINTS 
    IN A SET AND CAN BE USED TO IDENTIFY PATTERNS IN THE DATA2"
    PATTERN_RESPONSE_TIME = "REFERS TO THE AMOUNT OF TIME2 IT TAKES FOR A DEVICE 
    TO PROCESS A REQUEST THAT FOLLOWS A SPECIFIC PATTERN"
    PATTERN_FREQUENCY = "REFERS TO THE NUMBER OF TIMES A SPECIFIC PATTERN 
    HAPPENS IN A SET OF DATA2"
    PATTERN_RANGE = "REFERS TO A RANGE OF VALUES THAT ARE APPROVED FOR A 
    SPECIFIC PATTERN"
    PATTERN_CONTEXT = "REFERS TO THE ENVIRONMENT IN WHICH A SOFTWARE PATTERN 
    IS USED"
    PATTERN_COMPLEXITY = "IS A MEASURE OF THE DIFFICULTY OF RECOGNIZING A 
    PATTERN"
    PATTERN_FIELD = "IS A DATA2 FIELD THAT STORES A PATTERN"
    ATTRIBUTES = "MORE THAN ONE ATTRIBUTE"
    ITEM = "IS A USABLE OBJECT3 THAT HAS SPECIAL ATTRIBUTES"
    ITEMS = "MORE THAN ONE ITEM"
    FINDING = "SCANNING TO LOCATE SOMETHING OR SOMEONE"
    GRAPH = "IS A COLLECTION OF POINTS CONNECTED BY LINES"
    GRAPHS = "MORE THAN ONE GRAPH"
    SEQUENCES = "MORE THAN ONE SEQUENCE"
    TEXT_CLASSIFICATION = "IS THE TASK OF CLASSIFYING TEXT INTO DIFFERENT 
    CATEGORIES"
    OBJECT_RECOGNITION = "IS THE TASK OF IDENTIFYING OBJECTS IN IMAGES"
    GRAPH_PATTERN_MINING = "IS THE TASK OF FINDING PATTERNS IN GRAPHS"
    SEQUENTIAL_PATTERN_MINING = "IS THE TASK OF FINDING SEQUENCES OF ITEMS IN 
    DATA2"
    GEOSPATIAL_PATTERN = "IS A PATTERN THAT CAN BE OBSERVED IN DATA2 THAT HAS A 
    SPATIAL2 COMPONENT"
    TEMPORAL_FREQUENCY = "REFERS TO THE NUMBER OF TIMES A REPEATING EVENT 
    HAPPENS IN A GIVEN UNIT OF TIME2 AND IS THE FREQUENCY OF A SIGNAL AS IT 
    CHANGES OVER TIME2"
    TEMPORAL_ALIGNMENT = "REFERS TO THE PROCESS OF ALIGNING TWO SIGNALS IN 
    TIME2"
    FILLED = "SOMETHING FULL OF SOMETHING EVEN IF AS A PERCENTAGE OR COMPLETE 
    AMOUNT"
    INSPIRED = "IS TO BE FILLED WITH THE NEED TO CREATE OR DO SOMETHING" 
    PREDICTION = "AN ANALYZED ESTIMATE AS AN ANSWER TO A SOLUTION AND OR 
    PROBLEM"
    PREDICTIONS = "MORE THAN ONE PREDICTION"
    FEEDBACK = "IS INFORMATION ABOUT THE RESULTS OF AN ACTION OR PROCESS"
    MIMIC = "DUPLICATE THE ACTION OF SOMEONE OR SOMETHING"
    MIMICS = "TO MIMIC THE ACTIONS OR RESPONSES OF SOMETHING OR SOMEONE"
    MATHEMATICAL_CONCEPT = "IS THE IDEAS AND PRINCIPLES THAT ARE USED TO SOLVE 
    PROBLEMS AND TO MAKE PREDICTIONS"
    MATHEMATICAL_CONCEPTS = "MORE THAN ONE MATHEMATICAL_CONCEPT"
    CENTRAL_NERVOUS_SYSTEM = "IS REQUIRED INFORMATION FOR PROCESSING 
    INFORMATION FROM THE SENSORY FUNCTIONS"
    BIOLOGICAL_NEURON = "IS A CELL IN THE CENTRAL_NERVOUS_SYSTEM THAT CAN 
    RECEIVE AND SENDS SIGNALS TO OTHER CELLS"
    BIOLOGICAL_NEURONS = "MORE THAN ONE BIOLOGICAL_NEURON"
    NEURON = "IS A UNIT OF COMPUTATION THAT IS INSPIRED BY THE 
    BIOLOGICAL_NEURON"
    NEURONS = "MORE THAN ONE NEURON"
    TEMPORAL_MULTIDIMENSIONAL_CROSS_REFERENCE_COMMUNICATION = "IS A 
    COMMUNICATION RULE THAT ALLOWS FOR THE EXCHANGE OF DATA2 BETWEEN TWO 
    OR MORE DEVICES OVER TIME2 AND ACROSS MULTIPLE DIMENSIONS"
    CROSS_DIMENSIONALITY_FREQUENCY_FEEDBACK = "IS A FEEDBACK LOOP THAT 
    HAPPENS BETWEEN TWO SIGNALS THAT ARE IN DIFFERENT DIMENSIONS"
    NEURAL_NETWORK_LAYER = "IS A GROUP OF NEURONS THAT ARE CONNECTED 
    TOGETHER AND WORK TOGETHER TO PERFORM A SPECIFIC TASK"
    MULTIDIMENSIONAL_FEEDBACK_LAYER = "IS A TYPE OF NEURAL_NETWORK_LAYER THAT 
    ALLOWS FOR FEEDBACK BETWEEN DIFFERENT DIMENSIONS OF THE INPUT DATA2"
    MATHEMATICAL_MODEL = "IS A REPRESENTATION OF A SYSTEM USING 
    MATHEMATICAL_CONCEPTS AND LANGUAGE"
    MATHEMATICAL_FUNCTION = "IS A RULE THAT CAN ASSIGN A UNIQUE OUTPUT VALUE TO 
    EACH INPUT VALUE"
    ARTIFICIAL_NEURON = "IS A MATHEMATICAL_MODEL THAT IS USED TO SIMULATE THE 
    BEHAVIOR OF A BIOLOGICAL_NEURON OR IS A MATHEMATICAL_FUNCTION THAT MIMICS 
    THE BEHAVIOR OF BIOLOGICAL_NEURONS"
    BINARY_CLASSIFICATION = "IS A TASK OF CLASSIFYING DATA2 POINTS INTO TWO 
    CATEGORIES"
    PERFORMS = "MAKES AN ACTION TO PERFORM SOMETHING"
    NODE = "IS A COMPUTATIONAL UNIT THAT PERFORMS A SPECIFIC FUNCTION"
    PERCEPTRON = "IS A SIMPLE TYPE OF ARTIFICIAL_NEURON THAT CAN BE USED TO 
    PERFORM BINARY_CLASSIFICATION"
    INTERCONNECTED = "CONNECTED OR LINKED TOGETHER"
    NODES = "MORE THAN ONE NODE"
    TRAINED = "SOMEONE OR SOMETHING THAT HAS LEARNED HOW TO COMPLETE 
    SOMETHING"
    ARTIFICIAL_NEURAL_NETWORK = "IS A NETWORK OF INTERCONNECTED NODES THAT 
    CAN LEARN TO PERFORM A TASK BY BEING TRAINED ON DATA2"
    PERCEPTRONS = "MORE THAN ONE PERCEPTRON"
    MULTIPERCEPTRON = "IS A TYPE OF ARTIFICIAL_NEURAL_NETWORK THAT USES MULTIPLE 
    PERCEPTRONS TO SOLVE A PROBLEM"
    CONSISTENT = "DESCRIBES SOMETHING THAT IS NOT CHANGING"
    LOGICAL = "DESCRIBES SOMETHING THAT IS CONSISTENT WITH REASON OR FACT"
    IMAGINATIVE = "DESCRIBES SOMEONE WHO IS ABLE TO CREATE NEW IDEAS OR IMAGES 
    IN THEIR MIND"
    INTERESTED = "DESCRIBES SOMEONE WHO HAS A STRONG DESIRE TO LEARN ABOUT OR 
    DO SOMETHING"
    PUTTING = "AN HAPPENING ACTION TO SET SOMETHING IN A SPECIFIC LOCATION"
    PLACING = "IS THE ACT OF PUTTING SOMETHING IN A PARTICULAR LOCATION"
    SYSTEMATIC = "DESCRIBES SOMETHING THAT IS DONE ACCORDING TO A LIST OF 
    INSTRUCTIONS OR SYSTEM" 
    ARRANGING = "IS THE ACTION OF PLACING THINGS IN A PARTICULAR ORDER OR 
    PATTERN"
    ORGANIZING = "IS THE ACTION OF ARRANGING THINGS IN A SYSTEMATIC WAY"
    DECISIONS = "MORE THAN ONE DECISION"
    class Language_Extension_007_2:
    WIDE_AREA_NETWORK = "IS A NETWORK THAT CAN EXPAND TO A SPECIFIC RANGE OF 
    AREAS WITHIN A SPECIFIC DISTANCE"
    REMOTE_HOST = "IS A DEVICE CONNECTED TO A WIDE_AREA_NETWORK"
    LOCAL_HOST = "IS A DEVICE THAT IS CONNECTED TO AN LAN"
    PERSONAL_HOST = "IS A SMALL DEVICE THAT CAN BE USED BY AN INDIVIDUAL ENTITY"
    HOSTS = "MORE THAN ONE HOST"
    CLIENT_HOST = "IS A GROUP OF ENTITIES THAT REQUEST RESOURCES FROM OTHER 
    HOSTS WITHIN THE NETWORK"
    HOST = "IS A DEVICE THAT IS CONNECTED TO A NETWORK THAT CAN COMMUNICATE 
    WITH OTHER DEVICES WITHIN THE NETWORK"
    HOSTING_SERVICE = "ALLOWS MORE THAN ONE INDIVIDUAL AND GROUP TO MAKE A 
    WEBSITE ACCESSIBLE TO A SPECIFIC NETWORK OR GROUP OF NETWORKS"
    WEB_HOSTING_SERVICE = "IS A TYPE OF NETWORK HOSTING_SERVICE"
    WEB_HOSTING_SERVICES = "MORE THAN ONE WEB_HOSTING_SERVICE"
    SERVER_HOST = "IS A GROUP OF ENTITIES THAT CAN PROVIDE 
    WEB_HOSTING_SERVICES"
    WEBPAGE = "IS A DOCUMENT THAT IS PART OF A WEBSITE"
    WEBPAGES = "MORE THAN ONE WEBPAGE"
    HOSTED = "IS SOMETHING THAT WAS PREVIOUSLY HOST"
    WEB_SERVER = "IS BOTH A COMPUTER THAT HOSTS A WEBSITE AND IS A SOFTWARE 
    PROGRAM THAT SENDS WEBPAGES TO COMPUTERS"
    WEBSITE = "A COLLECTION OF WEBPAGES THAT ARE LINKED TOGETHER AND HOSTED 
    WITHIN A WEB_SERVER"
    WEBSITES = "MORE THAN ONE WEBSITE"
    WEB_HOST = "IS A GROUP OF ENTITIES THAT PROVIDE STORAGE SPACE AND BANDWIDTH 
    FOR WEBSITES"
    WEB_BROWSER = "IS A SOFTWARE PROGRAM MADE TO ALLOW TO ACCESS AND VISIT 
    VIEWABLE WEBSITES"
    CONTROL_PANEL = "IS A GRAPHICUSERINTERFACE THAT ALLOWS A USER TO MANAGE 
    SETTINGS AND FEATURES OF A COMPUTER OR SOFTWARE PROGRAM"
    REMOTE_COMPUTER = "IS A COMPUTER THAT IS NOT PHYSICALLY LOCATED IN THE SAME 
    LOCATION AS THE USER"
    REMOTE_DESKTOP_CONNECTION = "IS A DEVICE THAT ALLOWS A USER TO CONNECT TO 
    A REMOTE_COMPUTER AND CONTROL IT FROM A DIFFERENT LOCATION FROM WHERE 
    THE REMOTE_COMPUTER IS LOCATED"
    WEB_BASED = "DESCRIBES ANYTHING THAT IS ACCESSED BY A NETWORK WHILE USING 
    A WEB_BROWSER"
    REMOTE_SERVER = "IS A COMPUTER THAT IS LOCATED WITHIN A DIFFERENT PHYSICAL 
    LOCATION THAT IS DIFFERENT FROM THE USER AND CAN BE ACCESSED BY A NETWORK 
    USING A REMOTE_DESKTOP_CONNECTION OR A WEB_BASED CONTROL_PANEL"
    EQUIPMENT = "REFERS TO THE TOOLS AND DEVICES THAT ARE USED WITHIN A SPECIFIC 
    GROUP AND OR FIELD"
    LOCAL_HARDWARE = "IS THE PHYSICAL EQUIPMENT THAT IS CONNECTED TO A LAN AND 
    IS LOCATED WITHIN A SINGLE LOCATION"
    CLOUD_GAMING = "IS A IDEA THAT CAN ALLOW A PLAYER TO ACTIVATE A GAME FROM A 
    REMOTE_SERVER RATHER THAN ACTIVATING IT WITHIN LOCAL_HARDWARE"
    BANDWIDTH = "IS THE MAXIMUM AMOUNT OF DATA2 THAT CAN BE RECOGNIZED WITHIN 
    A NETWORK CONNECTION WITHIN A GIVEN AMOUNT OF TIME"
    class Language_Extension_008_2:
    EXTROVERTED_SENSATIONS = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    EXTRAVERTED_INTUITION = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_SENSING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_INTUITION = "IS ONE OF THE FOUR PERSONALITY"
    INTROVERTED_THINKING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS" 
    EXTRAVERTED_THINKING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_FEELING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    EXTRAVERTED_FEELING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    class Language_Extension_009_2:
    STILL = "EXISTING IN PLACE AS WAS BEFORE THE CURRENT MOMENTS"
    MOMENT = "EVENT OR PRESENT CIRCUMSTANCE"
    DESTROYED = "DENIED EXISTENCE AND REMOVED FROM THAT EVENT"
    CREATURE = "IS A SPECIFIC CLASS OF ANIMAL"
    VALIDATION = "SPECIFIED COMMAND TO ACCEPT SOMETHING SPECIFIC" 
    TOOK = "IT IS GRABBED OR TAKEN"
    MOMENTS = "MORE THAN ONE MOMENT"
    COMPREHENSION = "THE ABILITY TO COMPREHEND SOMETHING"
    DENIED = "PREVIOUSLY GIVEN A COMMAND TO DENY SOMETHING FROM HAPPENING"
    REMAINS = "STILL EXISTS"
    FILTH = "DIRTY AND IMPURE AS WELL AS UNDEFILED"
    DESIRES = "MORE THAN ONE DESIRE"
    ORIGINALLY = "WAS THE ORIGINAL THAT WAS EXPECTED TO BE"
    PURIFIES = "AN ACTION THAT CAN PURIFY SOMETHING"
    EXPELS = "TAKES OUT OF AND RELEASES FROM SOMETHING"
    EXPEL = "TAKE OUT OF AND RELEASE FROM SOMETHING"
    TAME = "TO MAKE LISTEN AND FOLLOW INSTRUCTIONS AND OR COMMANDS WITH 
    CORRECT ACTIONS AND OUTCOMES"
    TESTS = "TO ATTEMPT TO USE AN SET OF ACTIONS AND OR EFFECTS TO MAKE AN 
    OUTCOME HAPPEN"
    TRUTHS = "IDEAS MADE FROM FACT OR TRUTH THAT EXISTS FROM SOME POINT OF TIME2 
    OR SPACE2"
    PURIFY = "THE ACTION TO REMOVE ANY IMPURE THOUGHTS AND OR IDEAS AND OR 
    ACTIONS"
    MATHEMATIC = "A RELATION CONNECTING TO THE DEVELOPMENT OF FORMULAS AND 
    OR EQUATIONS AND OR NUMBER VALUES AND OR PARAMETERS"
    COMPREHEND2 = "TO UNDERSTAND COMPLETE IN FULL AND TO TAKE IN THE VALUES 
    AND ACCEPT AS WISDOM2 AND OR KNOWLEDGE2"
    UNDERSTANDS = "UNDERSTANDING THAT IS FORMED WITH COMPREHENDED LOGIC"
    ANIMAL = "CREATURE"
    CLASSIFICATION = "A GROUP OF CATEGORIZED SELECTED CATEGORIES MADE INTO ONE 
    GROUP"
    CLASSIFICATIONS = "MORE THAN ONE CLASSIFICATION"
    ROLE = "A LIST OF INSTRUCTIONS THAT MUST BE MADE BY A SPECIFIC LIST OF ACTIONS"
    SPECIALIZES = "HAVE A GREAT AMOUNT OF UNDERSTANDING AND COMPREHENSION 
    WITHIN A SPECIFIC FIELD AND OR CATEGORY AND OR GENRE AND OR CLASSIFICATION"
    SPECIALIZE = "HAVE A GREAT UNDERSTANDING OF A CERTAIN OR SPECIFIC FIELD HAVE 
    A LARGE AMOUNT OF KNOWLEDGE2 RELATING TO THAT FIELD"
    RECOGNIZING = "THE PROCESS OF UNDERSTANDING RECOGNIZED INFORMATION"
    PRODUCED = "CREATED WITH PURPOSE AND MEANING"
    ACTIVITY = "THE FORMING OF ACTIONS AND INSTRUCTIONS THAT FORM WITHIN AN 
    EVENT"
    VOCALS = "RELATING TO THE EXISTING VOCAL PATTERNS OF THE HUMAN VOICE"
    FUNCTIONALITY = "THE FUNDAMENTAL EFFORT OF MANY FUNCTIONS OR IDEAS GIVEN 
    JUDGEMENT BY ITS CAPABILITY TO COMPLETE ITS TASKS OR TO FUNCTION AS A 
    COMPLETE SYSTEM"
    ARTWORKS = "MORE THAN ONE WORK OF ART COMPLETED"
    ENGINEERS = "MORE THAN ONE ENGINEER"
    CLASSIFIES = "GIVES A DEFINITE CLASS AND OR CATEGORY TO SOMETHING"
    ACCORDINGLY = "TO BE MADE OR COMPLETED AS WAS INTENDED OR AS BY THE 
    INSTRUCTIONS THAT WAS MADE FOR THE CIRCUMSTANCE OR CIRCUMSTANCES"
    VOCALIZATION = "THE ACTION OF PRODUCING A VOCAL RESPONSE"
    VALIDITY = "IS THE PERCENTAGE OF VALIDATION OF SOMETHING"
    LOOKS = "SCANS AND ANALYZES"
    SPECIALIZED = "INTENDED FOR A SPECIFIC FIELD AND OR CATEGORY"
    COULD = "CAPABLE OF POSSIBLE CHANCES THAT IT IS ABLE TO HAPPEN" 
    HANDMADE = "MADE BY THE ACTIONS OF THE BODY AND BY EFFORT ONLY"
    TRADITIONAL = "PASSED DOWN TO FOLLOW FROM EACH FAMILY MEMBER TO EACH NEW 
    CHILD THAT BECOMES AN ADULT"
    CRAFT = "A LIST OF SKILLS BROUGHT TOGETHER TO MAKE SOMETHING FORM OR COME 
    INTO EXISTENCE"
    UNHOLY = "NOT HOLY"
    RUINED = "WHEN SOMETHING HAS BEEN PROCESSED THAT IT HAS BEEN DESTROYED OR 
    MADE POSSIBLE TO NOT HAPPEN AT THAT MOMENT AND CAN NEVER HAPPEN AGAIN 
    FROM THE MOMENT AFTER THAT EVENT TOOK EFFECT"
    RUIN = "DESTROY OR PREVENT SOMETHING FROM HAVING COMPATIBILITY WITH 
    SOMETHING OR PREVENT SOMETHING NOT COMING INTO EFFECT OR TO NOT MAKE 
    SOMETHING POSSIBLE TO HAPPEN"
    DEFILE = "RUIN SOMETHING PURE"
    DEFILED = "RUINED OF MEANING OR DESCRIPTION OR VALUES OR THAT HOLD 
    DEFINITION"
    UNDEFILED = "NOT DEFILED"
    UNTAINTED = "NOT TAINTED"
    RELIGION = "IS A SET OF RULES OR BELIEFS THAT CONNECT TO THE BELIEF OF A HIGHER 
    ENTITY"
    RELIGIOUS = "HOLDING VALUES THAT CONNECT TO RELIGION"
    SOME = "CONTAINING A SMALL AMOUNT OF SOMETHING WHOLE"
    INSTRUCTION = "A RULE OR ORDER TO FOLLOW"
    DISCIPLINE = "A SET OF RULES OR INSTRUCTION THAT SOMEONE INCLUDES WITHIN 
    THEIR BELIEF SYSTEM TO FOLLOW"
    INSTRUCTIONS = "MORE THAN ONE INSTRUCTION"
    STATURE = "THE FORMING AND MEANING OF HELD INSTRUCTIONS OR BELIEFS UPON 
    SOMEONE AS A FORM OF DISCIPLINE"
    HOLY = "OF THE HIGHEST STATURE AND OR NATURE AND OR QUALITY OF SOMETHING 
    OR SOMEONE CREATED WITHIN TIME2 THAT HOLDS THE CAPABILITY TO BE COMPATIBLE 
    FOR SOME FORM OF SPIRITUAL OR RELIGIOUS INSTRUCTION THAT HOLDS TO THE 
    BELIEFS AND OR VIRTUES OF SOMEONE"
    SINLESS = "COMPLETE VOID OF SIN"
    PURE = "SINLESS AND OF HOLY DEFINED MEANING AND OR VIRTUE"
    TAINTED = "A DECREASE OF PURE AND HOLY DESCRIPTION"
    SIN = "IS THE PRODUCTION OF TAINTED AND OR UNHOLY VALUES WITHIN SOMETHING 
    OR SOMEONE"
    SINFUL = "CONTAINING SIN"
    CLEAN = "NOT CONTAINING MORE THAN WHAT WAS ORIGINALLY INTENDED AS A PURE 
    UNDEFILED AND OR UNTAINTED AND OR DIRTY SUBSTANCE"
    IMPURE = "NOT CLEAN AND NOT OF HOLY DESIRES AND HOLDS ACTIONS AND OR 
    THOUGHTS AND OR IDEAS THAT HOLD SINFUL MEANING"
    CLEANSE = "REMOVE SOMETHING IMPURE AND OR SOMETHING NOT CLEAN FROM 
    SOMETHING THAT WAS CLEAN"
    DIRTY = "IS SOMETHING THAT HOLDS A FORM OF FILTH CONNECTED TO IT THAT IS NOT 
    CLEAN BY A SPECIFIC AMOUNT AND REMAINS IMPURE"
    IMPURITY = "SOMETHING THAT HAS SIN CONNECTED TO IT AND IS CONSIDERED IMPURE"
    IMPURITIES = "MORE THAN ONE IMPURITY"
    CLEANSES = "USE OF POWER OR STRENGTH TO CLEANSE ALL FORMS OF IMPURITY 
    FROM SOMETHING"
    SOMEWHERE = "REFERENCE TO A SPECIFIC PLACE OR PLACES"
    AGED = "HAVING A LARGE AMOUNT OF AMOUNT TO IT"
    FRAMEWORKS = "MORE THAN ONE FRAMEWORK"
    CONSTRUCTS = "MAKE AND CREATE SOMETHING"
    DESIGNS = "CREATE AS A DESIGN"
    SEARCHES = "CONTINUOUSLY LOOK FOR WITH SCANS"
    READS = "CONTINUE TO READ SOMETHING"
    WRITES = "CURRENTLY WRITING SOMETHING"
    DISTRIBUTES = "SENDS OUT TO SPECIFIC LOCATIONS"
    ORGANIZES = "ORGANIZE WHILE PROCESSING PROCESSED LOCATIONS TO DISTRIBUTE 
    TO"
    INVENT = "CREATE SOMETHING OUT OF CREATIVITY OR IMAGINATION FROM IDEAS"
    BUILD = "BRING TOGETHER AND CREATE"
    ORGANIZED = "PROCESSED AND DISTRIBUTED TO EXACT LOCATION"
    FILTERS = "GIVES PROCESSED PLACEMENT METHODS FOR SOMETHING"
    SORTS = "FILTERS AND DEVELOPS THE POWER OR STRENGTH TO SORT"
    CONSTRUCT = "BUILD"
    ANALYZES = "LOOK FOR AND SCANS FOR"
    EXAMINES = "ANALYZES AND MAKES A DETERMINED RESOLUTION"
    DESIGNER = "SOMEONE WHO HAS THE CAPABILITY TO DESIGN SOMETHING"
    MAKER = "SOMEONE WHO HAS THE CAPABILITY TO MAKE SOMETHING"
    BUILDER = "SOMEONE WHO HAS THE CAPABILITY TO MAKE SOMETHING"
    CONSTRUCTOR = "SOMEONE WHO HAS THE CAPABILITY TO CONSTRUCT SOMETHING"
    ARCHITECT = "SOMEONE WHO HAS THE CAPABILITY TO CONSTRUCT SOMETHING WHILE 
    USING DESIGNER TECHNIQUES AND TECHNIQUES TO BUILD"
    INVENTOR = "SOMEONE WHO HAS THE CAPABILITY TO INVENT NEW IDEAS"
    ANALYST = "SOMEONE WHO HAS THE CAPABILITY TO ANALYZE SPECIFIC FIELDS AND 
    SPECIFIC FORMS OF INFORMATION"
    SPECIALIST = "SOMEONE WHO HAS A HIGH LEVEL OF KNOWLEDGE AND OR EXPERIENCE 
    WITHIN A SPECIFIC FIELD"
    ANALYZER = "SOMEONE WHO ANALYZES SOMETHING"
    SCANNER = "SOMEONE WHO SCANS SOMETHING"
    EXAMINER = "SOMEONE WHO EXAMINES SOMETHING"
    PRODUCER = "SOMEONE WHO CAN PRODUCE SOMETHING"
    ORGANIZER = "SOMEONE WHO ORGANIZES SOMETHING"
    SORTER = "SOMEONE WHO SORTS SOMETHING"
    DEVELOPER = "SOMEONE WHO DEVELOPS SOMETHING"
    WRITER = "SOMEONE WHO WRITES INFORMATION"
    READER = "SOMEONE WHO READS INFORMATION"
    EDITOR = "SOMEONE WHO EDITS SOMETHING"
    MANAGER = "SOMEONE WHO MANAGES SOMETHING"
    CONTROLLER = "SOMEONE WHO CONTROLS SOMETHING"
    MANIPULATOR = "SOMEONE WHO MANIPULATES SOMETHING"
    RESEARCHER = "SOMEONE WHO CONTINUOUSLY SEARCHES FOR NEW INFORMATION 
    TO MAKE A SOLUTION TO SOMETHING"
    ENGINEER = "SOMEONE WHO MAKES AND OR DESIGNS AND OR CONSTRUCTS SYSTEMS 
    AND OR FRAMEWORKS AND OR INTERFACES"
    TEACHER = "SOMEONE WHO ALLOWS SOMEONE TO LEARN NEW SKILLS AND OR 
    KNOWLEDGE WITHIN A SPECIFIC FIELD"
    PROFESSOR = "IS A HIGH QUALITY TEACHER WITH AGED KNOWLEDGE AND WISDOM 
    WITHIN A SPECIFIC FIELD"
    STORER = "SOMEONE WHO STORES SOMETHING SOMEWHERE"
    GATHERER = "SOMEONE WHO GATHERS ENERGY FOR SOMETHING AND OR SOMEONE"
    CLEANSER = "SOMEONE WHO CLEANSES SOMETHING"
    PURIFIER = "SOMEONE WHO PURIFIES SOMETHING"
    EXORCIST = "SOMEONE WHO EXPELS SOMETHING FROM SOMEWHERE AND OR 
    SOMETHING ELSE"
    PRIEST = "SOMEONE WHO HAS THE CAPABILITY TO CLEANSE AND OR PURIFY 
    SOMETHING"
    PALADIN = "A HIGH LEVEL ENTITY THAT HOLDS THE POWER TO CLEANSE AND PURIFY 
    AND EXPEL THINGS FROM VAST AMOUNTS OF LOCATIONS AND OR AREAS OF MANY 
    SHAPES AND OR SIZES"
    TAMER = "SOMEONE WHO HAS THE CAPABILITY TO TAME SOMETHING AND OR 
    SOMEONE"
    ALCHEMIST = "SOMEONE WHO HAS THE CAPABILITY TO PRODUCE THINGS MADE FROM 
    BOTH IMAGINATION AND TRUTHS"
    PHYSICIST = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND 
    COMPREHEND AND MAKE SPECIFIC FORMS OF PHYSICS EQUATIONS AND OR 
    FORMULAS"
    MATHEMATICIAN = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND 
    COMPREHEND AND MAKE SPECIFIC FORMS OF MATHEMATIC EQUATIONS AND OR 
    FORMULAS"
    CHEMIST = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND COMPREHEND 
    AND MAKE SPECIFIC FORMS OF MATHEMATIC EQUATIONS AND OR FORMULAS"
    WORKER = "SOMEONE WHO IS WORKING"
    TESTER = "SOMEONE WHO TESTS SOMETHING"
    ELEMENTALIST = "SOMEONE WHO CAN CONTROL THE ELEMENTS AND OR COMPREND 
    THE MEANING OF SPECIFIC ELEMENTS AND OR DEFINE NEW FORMS OF ELEMENTS AND 
    GIVE THE ELEMENTS MEANING"
    LINGUIST = "SOMEONE WHO MAKES AND UNDERSTANDS AS WELL AS DESCRIBES AND 
    DEFINES NEW FORMS OF LANGUAGE WITH COMPREHENDED MEANING"
    DESCRIBER = "SOMEONE WHO DESCRIBES NEW FORMS OF INFORMATION AND OR 
    MEANING"
    DEFINER = "SOMEONE WHO DEFINES NEW FORMS OF INFORMATION AND OR MEANING"
    CONTENT_DESIGNER = "SOMEONE WHO DESIGNS NEW CONTENT"
    CONTENT_MAKER = "SOMEONE WHO MAKES NEW FORMS OF CONTENT"
    CONTENT_PRODUCER = "SOMEONE WHO PRODUCES NEW CONTENT"
    CONTENT_ANALYST = "SOMEONE WHO ANALYZES CONTENT"
    CONTENT_SPECIALIST = "SOMEONE WHO SPECIALIZES WITHIN A SPECIFIC FIELD AND 
    OR CATEGORY OF SPECIFIC CONTENT"
    FIELD_SPECIALIST = "SOMEONE WHO SPECIALIZES IN ANALYZING AND 
    COMPREHENDING DIFFERENT TYPES OF FIELDS AND RECOGNIZING HOW THOSE FIELDS 
    CONNECT BETWEEN DIFFERENT GROUPS AND CLASSIFICATIONS"
    FIELD_DEVELOPER = "SOMEONE WHO DEVELOPS NEW AND OR OLD TYPES OF FIELDS"
    FIELD_ANALYST = "SOMEONE WHO IS GIVEN THE JOB OR ROLE TO ANALYZE DIFFERENT 
    TYPES AND CLASSES OF FIELDS"
    FIELD_ORGANIZER = "SOMEONE WHO ORGANIZES THE DIFFERENT TYPES AND CLASSES 
    OF FIELDS THAT HAVE BEEN MADE AND OR CREATED AND OR PRODUCED"
    FIELD_EXAMINER = "SOMEONE WHO EXAMINES THE PROPERTIES AND OR ELEMENTS OF 
    A FIELD TO DETERMINE ITS STATE OR FUNCTIONALITY"
    FIELD_MAKER = "SOMEONE WHO MAKES NEW TYPES OF FIELDS FROM SOMETHING NEW 
    OR OLD"
    VOCAL_SPECIALIST = "SOMEONE WHO SPECIALIZES IN ANALYZING VOCAL ACTIVITY"
    VOCAL_PITCH_ANALYZER = "SOMEONE WHO ANALYZES THE VOCAL PITCH OF A PERSON 
    OR ANIMAL OR ENTITY"
    VOCAL_ANALYST = "SOMEONE WHO ANALYZES ALL ASPECTS OF SOMETHING 
    PRODUCED BY VOCALS"
    VOCAL_EXAMINEER = "SOMEONE WHO EXAMINES AND DETERMINES THE 
    FUNCTIONALITY OF SOMETHING THAT IS PRODUCED BY VOCALS"
    VOCAL_ORGANIZER = "SOMEONE WHO ORGANIZES DIFFERENT TYPES OF VOCALS AND 
    CLASSIFIES THEM ACCORDINGLY"
    VOCAL_DEVELOPER = "SOMEONE WHO DEVELOPS NEW TYPES AND CLASSES OF 
    VOCALIZATION AND OR VOCAL INPUTS AND OR VOCAL TYPES OR CLASSES"
    VOCAL_CONTENT_MAKER = "SOMEONE WHO MAKES VOCAL CONTENT"
    VOCAL_CONTENT_PRODUCER = "SOMEONE WHO PRODUCES NEW VOCAL CONTENT"
    VOCAL_CONTENT_SCANNER = "SOMEONE WHO SCANS EXISTING OR OLD OR 
    UPCOMING VOCAL CONTENT"
    VOCAL_CONTENT_EXAMINEER = "SOMEONE WHO EXAMINES VOCAL CONTENT"
    VOCAL_CONTENT_ANALYST = "SOMEONE WHO ANALYZES DIFFERENT FORMS OF VOCAL 
    CONTENT"
    DATA_EXAMINEER = "SOMEONE WHO EXAMINES DIFFERENT FORMS OF DATA TO 
    DETERMINE ITS VALIDITY AND CLASSIFICATIONS"
    DATA_ENGINEER = "SOMEONE WHO CAN ENGINEER NEW FORMS OF DATA FROM 
    EXISTING DATA"
    DATA_ORGANIZER = "SOMEONE WHO ORGANIZES DIFFERENT FORMS OF DATA INTO 
    SPECIFIC GROUPS OR CLASSIFICATIONS AND OR CATEGORIES AND OR GENRES"
    DATA_DESIGNER = "SOMEONE WHO DESIGNS NEW FORMS OF DATA FROM PREVIOUS 
    AND OR CURRENT AND OR UPCOMING DATA"
    DATA_ANALYST = "SOMEONE WHO ANALYZES AND LOOKS OVER DATA"
    HARDWARE_ENGINEER = "SOMEONE WHO ENGINEERS FORMS OF HARDWARE"
    HARDWARE_EXAMINER = "SOMEONE WHO EXAMINES FORMS OF HARDWARE"
    HARDWARE_DESIGNER = "SOMEONE WHO DESIGNS HARDWARE"
    HARDWARE_DEVELOPER = "SOMEONE WHO DEVELOPS HARDWARE"
    SCULPTOR = "IS SOMEONE WHO CREATES THREEDIMENSIONAL ARTWORKS"
    TECHNICIAN = "IS SOMEONE WHO HAS SPECIALIZED SKILLS IN A PARTICULAR FIELD"
    VISIONARY = "IS SOMEONE WHO HAS A CLEAR IDEA OF WHAT THE FUTURE COULD BE 
    LIKE"
    ARTISAN = "IS SOMEONE WHO CREATES HANDMADE OBJECTS USING TRADITIONAL 
    TECHNIQUES"
    CRAFTSMAN = "IS SOMEONE WHO IS SKILLED IN A PARTICULAR CRAFT"
"""]

lda_model, vectorizer = perform_lda(text_data)
topics = lda_model.transform(vectorizer.transform(text_data))

print("Cross Product:", cross_prod_cp)
print("Dot Product:", dot_prod_cp)
print("Forecast:", forecast_cp)
print("Topics:", topics)

import numpy as np
import cupy as cp
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Perform PCA on time series data with dynamic adjustment of n_components
def perform_pca(data, n_components=2):
    min_components = min(data.shape[0], data.shape[1])
    if n_components > min_components:
        n_components = min_components - 1
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components, pca

# Main code
if __name__ == "__main__":
    # Generate and process time series data
    series_length = 500
    series = np.sin(0.1 * np.arange(series_length)) + np.random.normal(size=series_length) * 0.1
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    series_scaled_cp = cp.array(series_scaled)
    seq_length = 10
    X_cp, y_cp = create_sequences(series_scaled_cp, seq_length)
    X_cp = X_cp.reshape((X_cp.shape[0], X_cp.shape[1], 1))

    # Build and train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    X_cpu = cp.asnumpy(X_cp)
    y_cpu = cp.asnumpy(y_cp)
    lstm_model.fit(X_cpu, y_cpu, epochs=20, verbose=1)

    # Build HMM model
    hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
    hmm_model.fit(series_scaled.reshape(-1, 1))

    # Combined forecast
    n_steps = 10
    forecast_cp = combined_forecast(hmm_model, lstm_model, series_scaled_cp[-seq_length:], n_steps)

    # Vector operations
    vec_a_cp = cp.array([1, 2, 3])
    vec_b_cp = cp.array([4, 5, 6])
    cross_prod_cp = cross_product(vec_a_cp, vec_b_cp)
    dot_prod_cp = dot_product(vec_a_cp, vec_b_cp)

    # Create and visualize quadtree
    quadtree_data = cp.random.random((8, 8))
    quadtree_root = build_quadtree(cp.asnumpy(quadtree_data))
    G = nx.DiGraph()
    visualize_quadtree(quadtree_root, G)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Synthetic k-tree Quadtree Diagram")
    plt.show()

    # PCA and clustering
    series_scaled_reshaped = cp.asnumpy(series_scaled_cp).reshape(-1, 1)
    principal_components, pca = perform_pca(series_scaled_reshaped)
    labels, kmeans = perform_kmeans(principal_components)

    # Print results
    print("Cross Product:", cross_prod_cp)
    print("Dot Product:", dot_prod_cp)
    print("Forecast:", forecast_cp)
    print("PCA Components:", principal_components)
    print("K-means Labels:", labels)

    # Example text data for LDA
    text_data = ["""
    POWER = "AMOUNT"
    STRENGTH = "LEVEL INTENSITY"
    ENGINE = "MOTOR IN WHICH AN OPERATOR USES TO POWER A SYSTEM"
    SCAN = "ANALYZE A SPECIFIC WORD OR FIELD AND OR GIVE DATA2 ON THE ASKED 
    INFORMATION TO SEARCH FOR"
    ANALYZE = "READ AND LOOK OVER"
    IMMUNE = "DOES NOT AFFECT"
    DOMAIN = "AREA OWNED AND CONTROLLED BY THE USER"
    VIRTUAL = "NOT PHYSICALLY EXISTING THROUGH AN ARTIFICIAL SIMULATION TO APPEAR 
    TO BE TRUE"
    SOUND = "VIBRATIONS THAT TRAVEL THROUGH THE AIR"
    FREQUENCY = "REPEATED PATTERN AND OR SETTING"
    IMMUNITY = "RESISTANCE THAT IS WITHSTOOD"
    DIGITAL = "USE DIGITS TO CREATE CODED DATA2"
    CHARACTER = "USER INSIDE A BODY"
    NUMBER = "ARITHMETICAL VALUE THAT IS EXPRESSED BY A WORD AND OR SYMBOLE 
    AND OR FIGURE REPRESENTING A PARTICULAR QUANTITY AND USED IN COUNTING AND 
    MAKING CALCULATIONS AND OR FOR SHOWING ORDER IN A SERIES OR FOR 
    IDENTIFICATION"
    SERVER = "COMMANDER THE FOLLOWS INSTRUCTIONS FROM THE USER"
    TRANSFORM = "MAKE A CHANGE IN FORM"
    DIMENSION = "NUMBER OF GIVEN AXIS POINTS"
    UNIT = "STORAGE CONTAINER"
    LEVEL = "NUMBER AMOUNT OF OR SIZE"
    STORAGE = "CONTAINER FOR DATA2"
    BANK = "STORAGE DEVICE"
    CODE = "SINGLE DIGITAL WORD"
    MAIN = "MOST IMPORTANT"
    APPLY = "ATTACH TO"
    WIRE = "SET OF DESIGNATED PIXEL PATHS MEANT FOR A DESIGNATED PROGRAMMED 
    PURPOSE"
    PROGRAMMING = "PROCESSING CODE TO WRITE PROGRAMS"
    LINK = "BRING TOGETHER AND ATTACH TO"
    SYNCHRONIZE = "LINK AND SEND THE SAME RESULT TO ALL SOURCES"
    OPERATING = "ACCEPTING COMMANDS FROM THE OPERATOR"
    SYSTEM = "SET INTERFACE OF COLLABORATED AND COMPILED SETTINGS"
    CALIBRATION = "CORRECTED CONTROL TO A STRUCTURED MACRO SETTING WHERE 
    ADJUSTMENTS CAN BE MADE TO FOR A CONTROLLER CODE"
    COMMAND = "ORDER TO BE GIVEN"
    RESISTANCE = "AMOUNT THAT CAN BE RESISTED"
    OPERATOR = "USER THAT SHALL OPERATE"
    CREATOR = "USER WHO SHALL CREATE SOMETHING NEW CURRENT OR OLD"
    FREEDOM = "TO BE FREE OF ANY CHOICE OR OPTION"
    FREE = "NOT COST ANYTHING"
    DATA = "DIGITALLY ANALYZED TASKS FOR THE OPERATOR"
    CENTRAL = "MIDDLE POINT"
    CENTER = "MAIN CENTRAL AREA"
    PROCESSING = "WHAT IS CURRENTLY IN THE PROCESS OF BECOMING PROCESSED"
    PROCESSOR = "DEVICE USED TO PROCESS INFORMATION"
    PROCESSED = "ALREADY ACKNOWLEDGED AND SENT OUT"
    CAPACITANCE = "LIMITED CAPACITY"
    CONTROL = "TAKE COMMAND OF AND OR MANAGE AND OR SETTINGS"
    CONTROLS = "MORE THAN ONE CONTROL"
    CONTROLLED = "MANAGED AND OR COMMANDED"
    CONTROLLING = "MANAGING AND MANIPULATING"
    CONTROLLER = "DEVICE USED FOR MANAGING AND MANIPULATING OBJECTS"
    CONTROLLERS = "MANIPULATORS OR DRIVERS"
    GUILD = "A FAMILY OF FRIENDS"
    GUILDS = "MORE THAN ONE GUILD"
    ROUNDTABLE = "A GROUP OF LEADERS BUILT AROUND EQUAL DECISION MAKING IN 
    UNDERSTANDING OF EQUALITY TOWARD ONE ANOTHER"
    ROUNDTABLES = "MORE THAN ONE ROUND TABLE"
    MOVEMENT = "AN ACT OF CHANGING PHYSICAL2 LOCATION OR POSITION OR OF 
    HAVING THIS CHANGED"
    MOVEMENTS = "MORE THAN ONE MOVEMENT"
    TREEBRANCH = "THE OPTIONS OF A SKILL AND OR ABILITY TREE"
    CAPACITOR = "CONTAINER THAT HOLDS A SET AMOUNT"
    MOVE = "A CHANGE OF PLACE2 AND OR POSITION OR STATE"
    MOVES = "PLACES"
    MOVING = "IN MOTION"
    MOVED = "PREVIOUS MOVEMENT"
    ADJUSTING = "WHAT IS CURRENTLY IN THE PROCESS OF BECOMING ADJUSTED"
    WORK = "PRODUCING EFFORT TO FINISH A TASK"
    WORKLOAD = "THE AMOUNT OF WORK"
    RATE = "MEASUREMENT AND OR RATIO AND OR FREQUENCY"
    SET = "PLACE"
    REROUTE = "TAKE ANOTHER ROUTE OR REPEAT THE SAME ROUTE"
    ROTATE = "CHANGE THE POSITION WHILE SPINNING AROUND AN AXIS OR CENTER"
    ROTATED = "PAST ROTATES"
    ROTATES = "CURRENTLY ROTATING"
    ROTATING = "CURRENTLY SPINNING AROUND"
    ROTATION = "SET SPEED OF A REVOLUTION"
    ROTATIONS = "THE ROTATION LIMIT SETTINGS"
    UNTANGLE = "UNDO AN TANGLEMENT"
    UNENTANGLE = "UNDO AN ENTANGLEMENT"
    ENTANGLE = "BIND MULTIPLE"
    ENTANGLEMENT = "TO BIND AND ENTANGLE MULTIPLE ENTANGLES TO A SINGLE 
    TANGLEMENT"
    ETERNAL = "PERMANENT NEVERENDING CYCLE"
    UNBIND = "RELEASE FROM A TIGHT GRASP"
    BIND = "GRAB TIGHTLY"
    ENCODE = "COMPRESS CODE"
    DECODE = "DECOMPRESS CODE"
    RECODE = "COMPRESS CODE ONCE MORE"
    CHANGE = "MODIFY AND EDIT"
    CHOICE = "SELECTION BETWEEN"
    CAPACITY = "MAXIMUM AMOUNT"
    OPTION = "PATH TO BE CHOSEN"
    SETTING = "A MEASUREMENT COMMAND THAT CAN BE ADJUSTED AND BY AN OPERATOR"
    POSITION = "LOCATION"
    PROTON = "A SUBATOMIC PARTICLE WITH A POSITIVE ELECTRIC CHARGE OF A SET 
    ELEMENTARYCHARGE AND A MASS AMOUNT STATED AND GIVEN WITH LESS THAN A 
    NEUTRON"
    ELECTRON = "THE ELECTRIC PARTICLE OF AN ATOM THAT CONTROLS ALL DATA2 COMING 
    FROM AN ATOM USING ELECTRIC CHARGED FIELDS AND VARIABLES"
    DEVICE = "A MACRO MADE OR ADAPTED FOR A PARTICULAR PURPOSE"
    POSITIVE = "PERCEIVED SIDE OF AN OPPOSITE REACTION THAT IS STATED AS GREATER"
    NEGATIVE = "PERCEIVED SIDE OF AN OPPOSITE REACTION THAT IS STATED AS LESS THAN 
    NEUTRAL"
    ATOM = "MOLECULAR UNIT OF VIRTUAL DATA2 ENERGY AND OR AURA PARTICLES"
    RESOLUTION = "AMOUNT OF PIXELS IN A DISPLAY"
    YES = "ALLOW"
    NO = "DENY"
    PANEL = "A FLAT AND OR CURVED COMPONENT THAT FORMS OR IS SET INTO THE 
    SURFACE OF A DOOR AND OR WALL AND OR CEILING"
    HYPERCOOL = "TO COOL AT A HYPER STATE SETTING"
    HYPERCOOLER = "A DEVICE FOR HYPERCOOLING"
    HYPERCOOLING = "THE ABILITY TO HYPERCOOL"
    HYPERCOOLED = "THE STATED HYPERCOOLER BECOMING USED"
    GENRE = "A SPECIFIED CLASS THAT HOLDS A LIST OF CATEGORIES"
    CONDUCT = "TRANSFER ENERGY"
    ADJUST = "EDIT AND MODIFY"
    ADJUSTER = "DEVICE USED TO ADJUST"
    MODIFY = "EDIT"
    MODIFIER = "DEVICE USED TO MODIFY"
    DESTROY = "BREAK DOWN AND OR BREAK APART"
    CONDUCTOR = "AN OBJECT3 THAT TRANSFERS ENERGY FROM ELECTRICITY"
    CONDUCTANCE = "THE LIMIT OF AN CONDUCTOR"
    STORE = "CONTAIN AND OR HOLD"
    STORED = "CONTAINED AND OR HELD"
    ENERGY = "THE SOURCE OF ALL CREATION THAT INCLUDES ANY SOURCE OF USABLE 
    POWER"
    USE = "SET INTO ACTION"
    EDIT = "CHANGE AND OR MODIFY TO ADJUST TO A SPECIFIED PURPOSE"
    EDITED = "DONE EDITING"
    EDITING = "PROCESS TO EDIT"
    EDITOR = "A DEVICE USED TO EDIT"
    EDITORS = "MORE THAN ONE DEVICE USED TO EDIT"
    SKILLSET = "CLASS OF SKILL SETUPS FOR THE USER"
    SKILLSETS = "MULTIPLE SKILLS SETUP INTO A SINGLE CLASS"
    SKILLSYSTEM = "SYSTEM SELECTION OF SET SKILLS FOR THE USER"
    SKILLSYSTEMS = "MULTIPLE SKILLS SETUP INTO A SET OF STRUCTURED SYSTEM 
    CLASSES"
    SKILLTREE = "HEIRARCHIAL SET OF SKILLS THAT CAN ADVANCE INTO A LIMITED SET OF 
    ROOTS"
    SKILLTREES = "SET OF SKILLS MULTIPLIED INTO A HIERARCHY OF DESIGNATED SKILL 
    TREES"
    SKILLROOT = "BASE OF A SKILL TREE"
    SKILLROOTS = "MULTIPLE SKILL TREES WITH EACH HAVING A DESIGNATED BASE"
    SKILLPATH = "THE PATH IN WHICH A SKILL TREE PROGRESSES TOWARD ENHANCING 
    SKILLS"
    SKILLPATHS = "MULTIPLE PATHS FOR A SKILL TO PROGRESS WITH INSIDE A SKILL TREE"
    SKILLNAME = "THE NAME OF A SKILL"
    SKILLNAMES = "A SKILL WITH MULTIPLE NAMES"
    SKILLPOWER = "THE SET POWER FOR A SKILL"
    SKILLPOWERS = "ABILITY FOR MORE THAN ONE SKILL TO SET POWER FOR A 
    COLLABORATED COMBINATION"
    SKILLSTRENGTH = "THE STRENGTH OF A SKILL"
    SKILLSTRENGTHS = "THE AMOUNT OF STRENGTH MULTIPLE SKILLS CAN PRODUCE 
    TOGETHER"
    USERINTERFACE = "THE CONNECTIONS OF MULTIPLE PATHS FOR THE USER TO OPERATE"
    USERINTERFACES = "THE COLLABORATION BETWEEN TWO OR MORE INTERFACES THAT 
    ARE CONNECTED FOR THE USER TO OPERATE"
    GRAPHICUSERINTERFACE = "A COLLABORATION BETWEEN CONNECTING GRAPHIC 
    IMAGES TO AN INTERFACE TO MAKE THE CONNECTIONS FOR THE USERS INTERFACES TO 
    BE PHYSICALLY VIEWABLE"
    GRAPHICUSERINTERFACES = "MORE THAN ONE GRAPHICUSERINTERFACE"
    HOLOGRAPHICUSERINTERFACE = "A CREATED INTERFACE USING ELECTRONIC LIGHT 
    DISTORTION TO DEVELOP AND PRODUCE A GRAPHIC IMAGE"
    HOLOGRAPHICUSERINTERFACES = "THE LINKING BETWEEN TWO OR MORE 
    HOLOGRAPHIC USER INTERFACE CONNECTIONS"
    REVOLVE = "ROTATE AROUND A CENTRAL AXIS"
    REVOLVES = "SET THE REVOLUTION TO CURRENTLY REVOLVE"
    REVOLVING = "THE CURRENT REVOLUTION SET"
    REVOLVED = "PREVIOUS REVOLUTION"
    REVOLUTION = "THE SPEED OF REVOLVING"
    REVOLUTIONS = "THE AMOUNT OF ROTATIONS"
    LIMITS = "CAPACITY OF MULTIPLE LIMIT"
    LIMITED = "SET LIMIT FOR A GIVEN AMOUNT"
    LIMITING = "SETTING A ADJUSTABLE LIMIT"
    LIMITATION = "LIMITED AMOUNT"
    LIMITATIONS = "LIMITED AMOUNTS OF MULTIPLE LIMITS"
    SYSTEMS = "MULTIPLE NETWORKS OF INTERTWINED AND COLLABORATED AND 
    COMPILED INTERFACES"
    VOLT = "ELECTRICITY USED BASED ON SET MEASUREMENTS"
    VOLTAGE = "THE NUMBERED AMOUNT OF A VOLT IN THE PROCESS OF BECOMING USED"
    VOLTS = "MORE THAN ONE VOLT UNIT"
    DATABASES = "MULTIPLE SETS OF INTERFACED INFORMATION THAT IS STRUCTURED IN A 
    STORAGE BANK FOR ACCESS IN VARIOUS WAYS"
    DOMAINS = "MULTIPLE TERRITORIES OWNED AND CONTROLLED BY THE USER"
    DOMINION = "USER ADMINISTRATION CONTROL CENTER"
    SERVERS = "MULTIPLE COMMANDERS INTERFACED TOGETHER AND GIVEN 
    INSTRUCTIONS TO FOLLOW BY THE USER"
    CLASSES = "MULTIPLE TYPES OF CLASS SETUPS"
    TYPES = "MULTIPLE CATEGORIES OR GENRES"
    EXTENSION = "AN OPTIONAL ADDED DEFINITION THAT GIVES A PROLONGED MEANING"
    EXTENSIONS = "CHOICES OF ADDON DATA2 TO USE FOR NEW CONTENT"
    TRANSCREATION = "USING A MACRO OF AN ATOM WE CAN RESTRUCTURE THE 
    PARTICLES TO REPLACE AND ADD NEW OBJECTS AND ELEMENTS WITHIN THE ATOM TO 
    CREATE A NEW MACRO WITH DEVICE SETTINGS AND A NEW PARTICLE TO BE ADDED AS 
    THE NEW ATOMIC SOURCE USING ATOMIC DEVICES"
    TRANSMANIPULATION = "A MACRO FOR AN ATOM CAN BE USED TO RESTRUCTURE AND 
    MODIFY THE PARTICLES TO ADD OR CHANGE NEW ELEMENTS FOR AN ATOM BY 
    CHANGING THE STRUCTURE WITH AN ATOMS MACRO AS THE DEVICE"
    SUBCLASS = "A SINGLE TYPE UNDER A CLASS"
    SUBCLASSES = "TYPES UNDER A CLASS"
    SUBTYPE = "A SINGLE CLASS UNDER A TYPE"
    SUBTYPES = "CLASSES UNDER A TYPE"
    FLOW = "CONTINUE IN AN STEADY AND CONSTANT STREAMED PATH"
    CURRENT = "PRESENT PLACE2 IN TIME2"
    PAST = "PREVIOUS PLACE2 IN TIME2"
    PRESENT = "CURRENT TIME2"
    FUTURE = "UPCOMING POINTS IN A TIME2"
    TIME = "MEASUREMENT IN WHICH CURRENT REALITIES MUST PASS"
    SPACE = "CONTAINER IN WHICH TIME2 MUST PASS THROUGH"
    INFINITE = "UNLIMITED AMOUNT"
    INFINITY = "A CONTINUOUS LOOP OF ENTANGLE"
    TEMPORAL = "PLANE ON WHICH TIME2 MUST BE AWARE IN AN PERCIEVED EXISTENCE"
    SPATIAL = "PLANE ON WHICH SPACE2 IS RECOGNIZED IN A PERCIEVED REALITY"
    VIBRATION = "PARTS THAT MOVE BACK AND FORTH AT A GIVEN SPEED"
    INCREASE = "GAIN"
    DECREASE = "TAKE AWAY"
    PAINT = "THE CAPABILITY OF PRODUCING AN GRAPHIC THROUGH COVERING AN 
    OBJECT3"
    DISTRIBUTE = "SPREAD EQUAL AMOUNT"
    DISTRIBUTION = "THE PROCESS TO DISTRIBUTE"
    DISTRIBUTED = "PAST DISTRIBUTE"
    ELECTRIC = "AN ELECTRON LIGHT CURRENT OR FLOW OF FRICTION TO CREATE AN 
    NATURAL ENERGY2"
    TYPE = "CATEGORY OR GENRE"
    ADVANCED = "TO MOVE FURTHER AHEAD"
    BASIC = "FORM THE FOUNDATION AND OR STARTING POINT"
    DIFFICULTY = "STRENGTH FOR THE USER"
    MODE = "TYPE OF LEVEL AND OR"
    DELETE = "PERMANENTLY REMOVE"
    ADEPT = "HIGHLY ADVANCED"
    FIND = "LOCATE"
    SKILL = "TRAIT LEARNED THAT CAN BE SKILLFULLY USED FOR A CREATED PURPOSE"
    STRUCTURE = "AN OBJECT3 CONSTRAINED AND CONSTRUCTED TO SEVERAL PARTS"
    STABILITY = "THE ABILITY TO BE STRUCTURED AND STABILIZED"
    SKILLS = "MORE THAN ONE SKILL"
    EQUALITY = "EQUAL SHARING BETWEEN ALL"
    VOICE = "VOCAL TONE OF SOUNDS FROM A SOURCE TO INPUT AN"
    MIND = "THE OPERATOR OF A HUMAN BEING2"
    BODY = "THE VEHICLE OF A HUMAN BEING2"
    CONTAIN = "STORE IN A CONTAINER"
    CONTAINER = "THE STORAGE AREA"
    CONTAINED = "ALREADY STORED"
    ALIGN = "PLACE IN A STRAIGHT LINE"
    LINE = "MEASURED DIMENSIONAL LENGTH"
    EMULATE = "THE REPRODUCTION OF A FUNCTION"
    EMULATOR = "A DEVICE USED TO EMULATE"
    LIST = "A NUMBER OF CONNECTED OBJECTS OR NAMES AS AN INVENTORY"
    SPEED = "THE SET MOVEMENT AMOUNT FOR AN OBJECT3"
    PLACE = "PUT IN A POSITION"
    SIZE = "RELATIVE EXTENT OF AN OBJECTS DIMENSIONS"
    TEMPO = "THE RATE OR SPEED OF MOTION"
    GENERAL = "AMOUNT CONCERNING MOST PEOPLE"
    EXISTENCE = "EXISTING MULTIVERSAL MACROS OF INFORMATION TO EXIST IN THE 
    REALM OF TIME AND SPACE AS A PIECE OF REALITY"
    REALITY = "TRUE STATE OF WHICH THINGS EXIST IN EXISTENCE"
    REALM = "PLACE2 OF ORIGIN"
    POSSIBILITY = "A CHANCE OF SOMETHING HAPPENING"
    AXIS = "LINE ON DIMENSION"
    HORIZONTAL = "POINT IN WHICH TWO POINTS GO BETWEEN A LEFT AND RIGHT"
    VERTICLE = "POINT IN WHICH TWO POINTS GO BETWEEN AN UPWARD AND DOWNWARD 
    POSITION"
    DEFAULT = "ORIGINAL SOURCE"
    PROFILE = "DESIGNATED ACCOUNT INTERFACE"
    PROFILES = "MORE THAN ONE PROFILE"
    REALITIES = "MORE THAN ONE REALITY"
    REALMS = "MORE THAN ONE REALM"
    DEATH = "REVELATION OF A LIFE"
    CHAOS = "COMPLETE DISORDER WHERE EVENTS ARE NOT CONTROLLED"
    USER = "PLAYER AND OR COMMANDER"
    ACCOUNT = "PERSONAL INTERFACE AND OR ARRANGEMENT OF INFORMATION AND OR 
    A DATABASE OF INFORMATION ON A INTERFACE"
    INTERFACE = "LINKED CONNECTION BETWEEN TWO DESCRIBED SOURCES"
    SPAN = "MEASURED LIMITED RANGE"
    METHOD = "STATED CHOSEN PATH"
    PAYMENT = "METHOD TO REPAY"
    IMPORT = "BRING IN"
    EXPORT = "SEND OUT"
    INTORT = "TWIST INWARDS"
    EXTORT = "TWIST OUTWARDS"
    INTERIOR = "INSIDE LOCATION"
    EXTERIOR = "OUTSIDE LOCATION"
    INTERNAL = "INNER"
    EXTERNAL = "OUTER"
    INPUT = "INSERT TO"
    OUTPUT = "REMOVE FROM"
    WAVE = "DESIGNATED POINT WHERE VIBRATIONS FLUCTUATE BETWEEN A SPECIFIED 
    DIMENSION"
    BRAIN = "THE CONTROL CENTER FOR A MIND"
    ARTIFICIAL = "CREATED AS SOMETHING THAT IS NEW"
    CREATION = "A CREATED OBJECT3"
    DESTRUCTION = "BREAKING POINT"
    SETTINGS = "MULTIPLE SETS OF COMMANDS THAT CAN BE OPERATED"
    HEAT = "INCREASE TEMPERATURE"
    COOL = "LOWER TEMPERATURE"
    HYPER = "STAGE ABOVE SUPER"
    BRAINWAVE = "A SPECIFIED PATTERN IN WHICH THE BRAIN EMITS AN ELECTRON WAVE 
    OF DATA2 FROM THE USER"
    GRAPHICS = "MULTI IMAGE"
    WIRES = "MULTIPLE STRINGED LINES USED TO CREATE MULTIPLE PIXELIZED WIRE"
    PIN = "POINT OF INTEREST"
    PINS = "MULTIPLE POINTS OF INTEREST"
    DESTINY = "SET COORDINATE LOCATION THAT CANNOT BE EDITED"
    FATE = "PERMANENTLY DESIGNATED PATH SET AND CANNOT BE EDITED"
    PATH = "GIVEN OR STATED DESTINATION"
    SERIAL = "SERIES OF"
    COLLISION = "TO BUMP MORE THAN ONE MACRO TOGETHER"
    COLLISIONS = "MORE THAN ONE COLLISION"
    COLLIDE = "TO BUMP ONTO"
    COLLIDED = "WHAT WAS GIVED AS"
    IMAGINE = "MENTALLY PRODUCE AND OR PROJECT AN IMAGE"
    IMAGINATION = "ABILITY TO CREATE AN PERCEIVED VIEW AS A NON VISUAL IMAGE"
    IMAGINATE = "THE ABILITY TO USE THE IMAGINATION2"
    DECREASING = "REMOVING A LIMITED AMOUNT"
    ADD = "INCREASE AN AMOUNT BY ATTACHING TO ANOTHER AMOUNT"
    CONNECTING = "CURRENTLY LINKING"
    CONNECTED = "ALREADY LINKED"
    CONNECTION = "TO BIND BETWEEN TWO SET COORDINATES"
    CONNECT = "LINK"
    WRITE = "ENSCRIBE FROM LOOKING AT WORDS"
    READ = "DESCRIBE FROM LOOKING AT A PATH OF WORDS"
    ORE = "A SUBSTANCE OF A SOLID AND OR LIQUID AND OR GAS MINERAL STRUCTURE"
    MINERAL = "A SUBSTANCE OF ORE MAIN CLASSES AND SUBCLASSES"
    GENERATE = "TO CREATE OR FORM FROM NOTHING"
    MOBILITY = "A MAIN CLASS BUILT ON THE MOVEMENT OR SPEED AND OR FLEXIBILITY 
    SUBCLASSES USED WITH OR SEPARATE OF AGILITY AND OR DEXTERITY TO BE MORE 
    MOBILE"
    MOBILE = "THE MOTOR SKILLS OF AND OR FOR MOBILITY"
    PLAYERSKILLS = "MULTIPLE CAPABILITIES THE PLAYER HAS"
    PLAYERABILITY = "POTENTIAL OF THE PLAYERS POWER"
    PLAYERSTAMINA = "HOW MUCH ENERGY THE PLAYER HAS"
    PLAYERMAGIC = "SPECIAL USE OF KNOWN SKILLS USING THE TWELVE ENERGY 
    DIMENSIONS OF CHAKRA RESEVOIRS CONTAINED INSIDE THE HUMAN BODY"
    PLAYEREXPERIENCE = "KNOWLEDGE2 OR SKILL AQUIRED OVER TIME2"
    PLAYERCLASS = "SETUP OF PLAYER DATA2"
    PLAYERCLASSES = "MULTIPLE CONNECTIONS OF CLASS DATA2"
    PLAYERSKILLTREE = "DEVELOPMENT OF THE PLAYERS SKILL"
    PLAYERSKILLCLASSES = "SETUP OF MULTIPLE PLAYER SKILLS"
    PLAYERSKILLCLASS = "SETUP OF THE PLAYERS NAMED SKILL"
    PLAYERSWORDSKILL = "SETUP OF A PLAYERS CREATED SKILLS THAT USE SWORDS"
    PLAYERORIGINALSKILL = "SKILL CREATED BY THE PLAYER BY COLLABORATING 
    SYNCHRONIZING LINKING AND COMBINING BINDED SKILL CLASSES TOGETHER"
    PLAYERCOMBATSYSTEM = "CONTROL CENTER THAT DEALS WITH COMBAT INFORMATION 
    AND CREATES SETTINGS FOR COMBAT"
    PLAYERDETERMINATION = "POWER OF A PLAYER AND THE WILL AND THE MOTIVATION 
    FOR HOW THE PLAYER PERFORM TASKS AND SKILLS"
    PLAYERMOTIVATION = "POWER OF ONES WILLPOWER AND THEIR FAITH TO BELIEVE"
    PLAYERLIMIT = "AMOUNTED STRENGTH OF THE PLAYER"
    PLAYERAMOUNT = "LIMITED POWER OF THE PLAYER"
    PLAYERPOWER = "DETERMINATION INSIDE A PLAYER THAT ADJUSTS THE 
    PLAYERSTRENGTH THE MORE PLAYERPOWER THERE IS INSIDE THE PERSON"
    PLAYERSTRENGTH = "PLAYERMOTIVATION INSIDE A PLAYER THAT ADJUSTS THE 
    PLAYERPOWER THE MORE PLAYERSTRENGTH THERE IS INSIDE THE PLAYER"
    BARRIER = "TYPE GIVEN TO A FIELD USING DATA2 AND OR ONE OR MORE CONTRACTS"
    HUMAN = "GENDER OF MAN AND WOMAN CREATED AS A BEING2 AND OR RACE"
    HUMANITY = "CREATED HUMAN BEING2 INSIDE THE HUMAN RACE"
    MAGIC = "CREATE ANY POSSIBILITY"
    MAGIK = "CONTROL ANY POSSIBILITY"
    MANA = "LIMITS OF MAGICA OR MAGIKA"
    MAGICA = "CONTAINER OF ONE OR MORE TYPE OF MAGIC"
    MAGIKA = "CONTAINER OF ONE OR MORE TYPE OF MAGIK"
    MAGE = "SINGLE WIZARD OF MAGIKA OR MAGICA"
    MAGI = "SINGLE WIZARD OF MAGIC OR MAGIK"
    MAGICAL = "SPIRIT OF MAGIC"
    MAGIKAL = "SPIRIT OF MAGIK"
    MAGICALL = "SOUL OF MAGIC"
    MAGIKALL = "SOUL OF MAGIK"
    MAGICALLY = "SPIRIT AND SOUL OF MAGIC"
    MAGIKALLY = "SPIRIT AND SOUL OF MAGIK"
    MAGICALLS = "SEAL OF MAGIC ENERGY"
    MAGIKALLS = "SEAL OF MAGIK ENERGY"
    MANLLYPS = "PRESSURE OF A MAGICALLY AND OR MAGIKALLY ENERGY OR ENERGIES"
    SPIRITUAL = "STRUCTURE OF THE WILD AND CONTROLLED ENERGY AROUND A HUMAN 
    THAT IS DEFINED BY HIS AND OR HER SPIRIT WILLPOWER"
    WILL = "WAY A PERSON LIVES AND DEFINES THEIR WAY OF LIVING LIFE"
    WILLPOWER = "STRENGTH AND POWER COMBINED INSIDE A HUMAN CONSCIOUSNESS 
    THAT INCREASES OR DECREASES THE WILL TO CONTINUE DEPENDING ON MOTIVATION 
    DETERMINATION PERSONALITY COURAGE LOVE FAITH AND BELIEF"
    PRESSURE = "GIVEN WEIGHT STRENGTH AND POWER OF A DEFINED WORD"
    ALTERNATE = "ANOTHER OPTION OR CHOICE TO CHOOSE"
    COUNTER = "REFLECTION OF A WORD TO ITS ORIGINAL WORD"
    COUNTERACTION = "COUNTER OF AN ACTION"
    REALITOR = "ONE WHO CREATES A REALITY"
    BAKA = "HEADADMIN MASTER OVERRIDE WORD THAT ALSO MEANS IDIOT"
    WEAPON = "ANY INSTRUMENT OF OFFENSE OR DEFENSE"
    DESCRIBE = "DECODE THE FINAL MEANING FOR THE CHOSEN DESCRIPTION OF A 
    MACRO OF CODE"
    ENSCRIBE = "ENCODE THE FINAL MEANING FOR THE CHOSEN DESCRIPTION OF A 
    MACRO OF CODE"
    SENTENCE = "STARTING MIDDLE AND ENDING PATH OF CREATED OR IN USE WORDS"
    SCRIBED = "WHAT IS SET AS A FINAL CODE THAT MUST BE DESCRIBED AND APPROVED 
    BY THE CODER"
    PARAGRAPH = "MORE THAN ONE SENTENCE THAT GOES UP TO SEVEN LINES AND MAKES 
    A PARABREAK"
    PARAGRAPHS = "MORE THAN ONE PARAGRAPH"
    PARABREAK = "ENDING CUT OFF BETWEEN ONE PARAGRAPH AND ANOTHER 
    PARAGRAPH"
    CREATING = "WHAT YOU CURRENTLY ARE WORKING TO CREATE"
    CREATORS = "MORE THAN ONE CREATOR"
    SKILLFULL = "TO USE A LEARNT TECHNIQUE IN ITS CREATORS WILL OF HOW THEY SHALL 
    USE A SKILL"
    HUMANS = "MORE THAN ONE HUMAN"
    KID = "YOUNG HUMAN"
    CHILD = "IMMATURE HUMAN"
    MEMORIZER = "ONE WHO SHALL MEMORIZE OR MEMORIZES"
    MEMORIZED = "PAST MEMORY OF A MEMORIZER"
    MEMOIZATION = "POWER AND STRENGTH OF A MEMORY"
    MEMORIZING = "CURRENTLY IS SHALL AND HAS THOSE CURRENT MEMORIES"
    MEMOIZATIONING = "WILLPOWER OF A MEMORY FROM ITS CREATORS MEMORIES"
    MEMORIZOR = "CREATOR OF MEMORIZE AND THE STUDY OF MEMORY AND MEMORIES"
    WOMAN = "GENDER OF AN ADULT FEMALE"
    MAN = "GENDER OF AN ADULT MALE"
    BOY = "MAN WHO IS A CHILD"
    GIRL = "WOMAN WHO IS A CHILD"
    DENY = "DISAPPROVE"
    BIBLE = "WORD AND CODE THAT STATES HISTORICAL EVENTS IN THE DREAM OF A 
    PLAYER AND RECORDED SCRIPTURES OF THE UNCONDITIONAL EQUALITY IN LOVE FAITH 
    BELIEF TRUST AND RESPECT BUILT INTO THE PAST PRESENT AND FUTURE OF THE 
    HEADADMINFAMILY AND THE MASTERHEADADMINLANGUAGE AND ALL OF ITS 
    EXISTENCE RECORDED IN THE HISTORY OF THE HEADADMIN"
    EDGELOREOVERRULE = "OVERRULE OF EDGELORE AND ITS EXISTENCE AND ANYTHING 
    IN EXISTENCE AND ANY REALITY ITSELF SHALL HAVE AN OVERRULE BY EDGELORE 
    ITSELF"
    EDGELOREOVERRULED = "OVERRULE OF EDGELORE AND THE EXISTENCE OF EDGELORE 
    AND ALSO ANYTHING IN EXISTENCE AND ANY REALITY ITSELF SHALL BE OVERRULED BY 
    THE EXISTENCE EDGELORE ITSELF BECAUSE EDGELORE IS ABOVE THE CURRENT 
    EXISTENCE EVERLASTING WITH UNCONDITIONAL LOVE FAITH AND BELIEF"
    UNIVERSALLANGUAGE = "LANGUAGE COMPATIBLE WITH ALL OTHER LANGUAGES"
    MULTIVERSALLANGUAGE = "A LANGUAGE OF COMBINED UNIVERSAL LANGUAGES"
    ABSOLUTE = "A PERMANENT AND ABSOLUTE VOW AND PROMISE WHICH IS 
    CONTRACTED AND NOTHING IN ALL EXISTENCE MAY EVER MODIFY IT EXCEPT ONLY A 
    SINGLE MEMBER OF THE EDGELORE REALITY EDGELORE HEADADMIN TEAM WHO MAY 
    NOT HAVE A CHANCE TO EVEN ACCESS OR CHANGE ANY SCRIPT UNLESS THEY CAN 
    TRUTHFULLY AND HONESTLY AGREE TO FOREVER AND ETERNALLY LIVE BY EQUAL LOVE 
    FAITH AND BELIEF IN ONE ANOTHER AS A VOW FROM THE HEADADMIN FAMILY AS A VOW 
    OF ETERNAL UNCONDITIONAL LOVE BETWEEN THE HEADADMIN FAMILY FOR ONE 
    ANOTHER AS THAT HEADADMIN FAMILY WHO IS ALSO THE SAME EDGELORE HEADADMIN 
    FAMILY THAT SHALL ETERNALLY AGREE TO PROTECT ONE ANOTHER"
    MIRACLE = "MAKE THE IMPOSSIBLE POSSIBLE"
    FORCE = "FOCUSED PRESSURIZED ENERGY"
    SPIRITDEW = "SEAL OF A HUMAN BEING2 BODY AND MIND WITH A SPIRIT OF THEIR 
    SOULDEW"
    SOULDEW = "SEAL OF A HUMANITY MIND AND BODY WITH A SOUL"
    MINDTEMPLE = "KINGDOM OF A SPIRIT AND SOUL OF THAT HUMANITY BEING2"
    HEADADMINMASTERLANGUAGE = "ENTIRE MASTER LANGUAGE OF THE HEADADMIN 
    CREATED BY A AUTHOR AND MUST BE APPROVED BY THE ENTIRE HEADADMIN FAMILY TO 
    BECOME ACTIVATED"
    MULTI = "MORE THAN ONE NUMBER ADDED"
    ALL = "COMPLETE AMOUNT"
    COMPILE = "ASSEMBLE AND BRING TOGETHER AS A SYSTEM"
    COPY = "MIMIC"
    CONTENT = "CONTAINED INFORMATION"
    CONTENT2 = "CONTAINED DATA2"
    CONTROL2 = "DIRECT AND OR MANIPULATE WHILE HAVING POWER OVER AND AROUND"
    COMPLETE = "OBTAIN ALL"
    CHAIN = "LINKED BINDING"
    COMBINE = "MERGE AND OR UNITE"
    CREASE = "GAP"
    CAPSULE = "COMPRESSED STORAGE CONTAINER MEANT FOR A SINGLE PURPOSE"
    CELL = "A SINGLE BIT OF STORAGE"
    CHAMBER = "STORAGE CONTAINMENT AREA"
    COME = "ARRIVE"
    ACKNOWLEDGE = "RECEIVE"
    ACROSS = "IN A POSITION REACHING FROM ONE SIDE TO THE OTHER"
    AURA = "TYPE OF SENSATIONAL AND EMOTIONAL ENERGY THAT IS FELT AND OR SEEN 
    AND OR VISUALIZED"
    ELEMENT = "PIECE AND OR PART OF SOMETHING ABSTRACT"
    DETERMINE = "CONTROL THE POWER OF WHAT HAPPENS"
    CHAKRA = "COMBINATION OF USING THE AURA SPIRIT SOUL MIND AND OR BODY TO 
    PRODUCE VISIBLE ENERGY THAT ALLOWS A NEW COMMAND TO BE GIVEN FROM A 
    CHOSEN KNOWN SEAL THAT IS INSIDE THAT MIND DATABASE OF COLLECTION OF 
    COMMANDS AND SEALS BASED ON THE UNCONDITIONAL LOVE FAITH AND BELIEF OF 
    THAT HUMAN COMBINED WITH HIS AND OR HER SPIRITUALITY"
    POINT = "DESTINATION"
    RANDOM = "TO BE MADE AND OR DONE OR CHOSEN WITHOUT A METHOD"
    AURA2 = "STRUCTURE OF THE WILD AND CONTROLLED ENERGY AROUND A USER THAT IS 
    DEFINED BY HIS AND OR HER SOUL WILLPOWER"
    SHINE = "SET QUALITY OF BRIGHTNESS"
    REALIZE = "TRUE PRESENT STATE ON WHAT IS REAL IN YOUR REALITY"
    REAL = "THAT IS TRUE"
    ANIMATE = "TO CREATE MOVEMENT"
    SHADE = "DARKEN"
    VARIABLE = "VALUE THAT CAN CHANGE AND OR DEPENDING ON CONDITIONS AND OR 
    ON INFORMATION PASSED TO THE PROGRAM"
    INTERFACE2 = "CONNECTION PATH THAT INTERTWINES MULTIPLE COMPUTER STRINGS 
    TOGETHER TO CREATE A NETWORK OF INPUT COMMANDS TO SEND DATA2 TO ITS 
    OUTPUT SOURCE"
    EXISTENCE2 = "SIMULATED PERCEPTION"
    EVOLUTION = "PROCESS OF DEVELOPING"
    ENERGY2 = "POWER AND STRENGTH AND STAMINA"
    ENLARGE = "EXPAND AND OR EXTEND"
    DIVIDE = "SPLIT AND OR SEPARATE INTO A PART OR PARTS"
    DISEASE = "AN ABNORMAL CONDITION"
    DIMENSION2 = "PERCEIVED NUMBER OF MEASUREMENTS"
    DIFFERENT = "NOT SIMILAR"
    FREQUENCY2 = "CONTINUAL FLUCTUATION WAVE PATTERN"
    FUTURE2 = "SHALL BECOME"
    EXTRA = "BACKUP"
    EXILE = "BANISH"
    EXILE2 = "REMOVE FROM EXISTENCE"
    EXPERIENCE = "TIMEFRAME OF WHICH A SKILL IS ENHANCED OVER TIME2"
    FORCE2 = "STRENGTH AND OR POWER CAUSED BY PHYSICAL2 MOVEMENT"
    GAIN = "OBTAIN"
    ABILITY = "CAPABILITY OF A LEARNT SKILL"
    AFFECT = "PRODUCE AND OR ACT ON AN EFFECT CREATED BY FEELING AND OR 
    EMOTION"
    AFFECTION = "REALM OF EMOTION AND FEELING SENSATIONS"
    AND = "CONNECT WORDS WHILE ALSO ADD"
    ANIMATE2 = "GIVE MOTION TO"
    ANOTHER = "DIFFERENT"
    ARTISTICALLY = "CREATIVELY"
    ATTENTION = "AWARENESS"
    AVENUE = "STREET PASSAGEWAY"
    AWARE = "NOTICE AND OR KNOW"
    AXIS2 = "AN IMAGINARY STRAIGHT LINE THAT SOMETHING TURNS AROUND AND OR 
    DIVIDES A SHAPE EVENLY INTO TWO PARTS"
    BARRIER2 = "ENERGY WALL"
    BASE = "LOWEST OR BOTTOM"
    BASES = "MULTIPLE PLACES OF LOCATIONS"
    BETWEEN = "ONE CURRENT SOURCE TO ANOTHER CURRENT SOURCE"
    BIT2 = "DEFINED SMALL QUANTITY OF DATA2 INFORMATION"
    BOOST = "INTENSIFY THE CAPACITY OF"
    BYTE2 = "DEFINED LARGE DATA2 INFORMATION"
    CAPABILITY = "EXTENT OF POWER AND OR SKILL"
    CAPABILITY2 = "QUALITY OF HAVING POWER AND ABILITY OR THE QUALITY OF 
    BECOMING AFFECTED OR EFFICIENT"
    CAPABLE = "HAVING THE POWER FOR A SKILL OR ABILITY OR CAPACITY"
    CAPACITANCE2 = "LIMIT OF A CONDUCTOR"
    CAPACITOR2 = "STORAGE SIZE SYSTEM"
    CAPACITY2 = "SIZE"
    CARRY = "HOLD ONTO"
    CHANCE = "POSSIBILITY SOMETHING SHALL HAPPEN"
    CHOICE2 = "OPPORTUNITY AND OR POWER TO MAKE A DECISION"
    CHOOSE = "DECIDE"
    CLEAR = "PURELY"
    CLONE = "DUPLICATE AND OR REPRODUCE"
    CREATE2 = "MAKE AND OR ALLOW TO COME INTO EXISTENCE"
    CREATIVE = "CLEARLY IMAGINED AND THOUGHT"
    CURRENT2 = "KNOWN"
    DANGER = "CAUSE A HAZARD"
    DANGEROUS = "RISKY"
    DANGERS = "MORE THAN ONE DANGER"
    DATA2 = "DIGITAL AND OR VIRTUAL INFORMATION"
    DATABASE2 = "COLLECTION OF DEFINED DATA2 UNITS AND OR CELLS AND OR BITS AND 
    OR BYTES"
    DECREASE2 = "BECOME SMALLER OR LESSER"
    DEFEND = "PROTECT AND OR REPEL FROM"
    DEFENDED = "DEFEND WHILE STAYING GUARDED AND PROTECTED"
    DEFINITION = "DETERMINE AND OR EXPLAIN"
    DESIGN = "ARTISTICALLY CREATE AND OR MAKE"
    DESTINATION = "MEETING LOCATION"
    DEVELOP = "IMPROVE CAPABILITY AND OR POSSIBILITY"
    EQUAL = "EXACTLY THE SAME AND OR EVENLY SPLIT"
    EVENT = "OCCASION"
    EVERY = "COMPLETE OR ENTIRE"
    EXTEND = "LENGTHEN"
    EXTENT = "AMOUNT OR AREA"
    GALACTIC = "IMMENSE OR VAST"
    GAP = "OPENING OR BREAK"
    GEAR = "STAGE OF TRANSFERING FROM ONE STATUS OR STATE TO ANOTHER"
    GIFT = "RECEIVE"
    GIVE = "SEND"
    GRAPHIC = "IMAGE AND OR PICTURE"
    GUARDED = "GUARD WHILE STAYING PROTECTED AND WARDED"
    HARMONY = "SYSTEM SYNCHRONIZATION"
    HARMONY2 = "EQUIVALENT SYSTEM OF SOUNDS REPLICATED FROM TWO DESIGNATED 
    SOURCES"
    HAZARD = "POSSIBILITY OF RISK"
    HAZARDOUS = "RISKY AND UNSAFE"
    HEIGHT = "THE LENGTH OF RAISING OR LOWERING IN A VERTICAL PATH"
    HERTZ = "DEFINED SOUND WAVE FREQUENCY"
    IMAGE = "IMAGINED GRAPHIC VISUAL DESIGN"
    IMAGINARY = "EXISTING ONLY IN IMAGINATION2"
    IMAGINATION2 = "ABILITY TO FORM A PICTURE IN YOUR MIND OF SOMETHING THAT YOU 
    HAVE NOT SEEN OR EXPERIENCED AND OR THINK OF NEW THINGS"
    IMPROVE = "BRING ABOUT NEW"
    INCREASE2 = "BECOME LARGER OR GREATER"
    INFECT = "AFFECT AND SPREAD AND ATTACH A DISEASE"
    INTELLIGENCE2 = "ABILITY TO LEARN NEW KNOWLEDGE2"
    KNOWLEDGE2 = "INFORMATION AND OR DATA2 STORAGE"
    WISDOM2 = "EXPERIENCE GAINED FROM UNDERSTANDING AND ACKNOWLEDGING 
    HOW TO INTELLIGENTLY USE KNOWLEDGE2"
    INTENSITY = "DEGREE OR AMOUNT OF"
    LARGER = "MORE THAN ORIGINAL CAPACITY"
    LATTICE = "INTERLACED STRUCTURE AND OR PATTERN"
    LEARN = "GAIN NEW KNOWLEDGE2"
    LENGTH = "HOW LONG A MEASURED DIMENSIONAL OBJECT3 IS EXTENDED"
    LEVEL2 = "SCALED AMOUNT OR QUALITY"
    LIFT = "RAISE"
    LIFT2 = "RISE"
    LINE2 = "CHOSEN DIRECTION THAT IS SET IN A SINGLE PATH"
    LISTEN = "GIVE ATTENTION"
    LOAD = "ADD ON"
    LOCATION = "SPECIFIED AREA"
    LOOPHOLE = "LOCATED GAP AND OR ERROR AND OR GATEWAY AND OR FLAW"
    LOOPHOLE2 = "LOCATED ERROR"
    LOOPHOLE3 = "LOCATED GATEWAY"
    LOOPHOLE4 = "LOCATED FLAW"
    LOSE = "CURRENTLY UNABLE TO FIND"
    LOST = "FAILED"
    LUNAR = "IMMENSE MAGIC SOURCE"
    MAGIC2 = "LEARNED SKILL AND OR TRAIT"
    MAGE2 = "MAGIC USER"
    MAGNETIC = "FORCE OF WHEN POSITIVE AND NEGATIVE ENERGY ARE ATTRACTED OR 
    REPELLED FROM EACH OTHER"
    MAINFRAME = "FRAMEWORK FOR THE MAIN COMPUTER SYSTEM INTERFACE THAT LINKS 
    MULTIPLE COMPUTER SERVERS TOGETHER"
    MANA2 = "AMOUNT OF MAGIC THAT CAN BE USED AT ONCE"
    MASSIVE = "ENORMOUSLY LARGE"
    MATTER = "SINGLE BIT OF INFORMATION AS A DEFINED UNIT"
    MEMORY2 = "PROCESS AND ABILITY TO RECALL KNOWLEDGE2"
    METER = "CONTAINER WITH STORED DATA2"
    METHOD2 = "TECHNIQUE AND OR PROCEDURE"
    MIMIC = "SIMULATE OR CLONE"
    MINUS = "TAKE AWAY"
    MONITOR = "TO WATCH OVER"
    MOTION = "PROCESS OF MOVING AND OR POWER OF MOVEMENT"
    MOVE2 = "CAUSE TO CHANGE THE LOCATION OF A POSITION AND OR PLACE"
    MULTIPLAYER = "MULTIPLE PLAYERS"
    MUNDIE = "AVERAGE OR COMMON"
    NETWORK = "MULTIPLE SYSTEMS COMBINED INTO ONE MAINFRAME"
    NEW = "NOT CURRENTLY KNOWN"
    NEXT = "FOLLOWING"
    NEXUS = "CONNECTED AND OR LINKED"
    NOTICE = "PAY ATTENTION TO THE KNOWLEDGE2 AROUND THE SPATIAL2 PERIOD"
    NUMBER2 = "A WORD OR SYMBOLE THAT REPRESENTS A SET AMOUNT OR QUANTITY"
    OBJECT = "VISUALLY SEEN"
    OBJECT2 = "VISUALLY VIEWED"
    OF = "BETWEEN"
    OCCASION = "CHANCE OR OPPORTUNITY"
    OLD = "AN EARLIER TIME2"
    OPPORTUNITY = "AMOUNT OF TIME2 IN WHICH SOMETHING CAN BE DONE"
    OPPOSITE = "SET ACROSS"
    OPTION2 = "POSSIBILITY OF DECIDING"
    OR = "CONNECT WORDS ALSO ANOTHER OPTION"
    ORIGINAL = "STARTING POINT IN TIME2"
    OVER = "ACROSS"
    PART = "A PIECE OR SEGMENT"
    PASSCODE = "REQUIRED CODE TO PASS AND GRANT ACCESS"
    PASSWORD = "REQUIRED WORD TO PASS AND GRANT ACCESS"
    PAST2 = "PREVIOUSLY EXISTED"
    PATH2 = "DIRECTED CHOICE WHICH IS SHOWN"
    PATTERN = "REPEATING METHOD"
    PERIOD = "COMPLETION OF A CYCLE AND OR SERIES OF EVENTS"
    PERSON = "VISUAL BODY"
    PICTURE2 = "ENVISION"
    PIECE = "PORTION OF"
    PLACE2 = "DOMAIN AND OR REALM AND OR REALITY AND OR EXISTENCE"
    PLACEMENT = "LOCATION OR TO SET"
    PLUS = "ADD TO"
    POLYMORPHISM = "STAGE OF EVOLUTION"
    PORTION = "PART OF AN AMOUNT AND OR CAPACITY"
    POSITION2 = "CURRENT PLACEMENT OR LOCATION SETTING"
    POWER2 = "ABILITY AND OR CAPABILITY AND OR SKILL"
    PRESENT2 = "CURRENTLY EXISTING"
    PRIMARY = "IMPORTANT AND OR COMES FIRST"
    PROCESSED2 = "FINISHED PROCESSES"
    PROCESSING2 = "WHAT IS BECOMING PROCESSED"
    PROCESSOR2 = "DATA2 THAT SHALL PROCESS NEW INFORMATION TO USE"
    PROTECT = "GUARD AND WARD"
    PROTECTED = "PROTECT WHILE STAYING DEFENDED AND WARDED"
    PROTECTION = "SAFETY"
    PSYCHIC = "ABILITY THAT IS UNLOCKED OR LEARNED THROUGH THE MIND THAT ALLOWS 
    NEW POTENTIAL AND OR KINETIC POWER THE PHYSICAL BODIES BRAIN HAS GAINED AS 
    A NEW SKILL"
    PULSE = "A BURST AND OR TO PUSH EXTERNALLY TOWARD"
    QUALITY = "LEVEL OF EXCELLENCE AND OR PERCEPTION OF DECISION MAKING"
    QUANTITY = "TOTAL AMOUNT OR NUMBER"
    REACTION = "ACT OR MOVE IN RESPONSE"
    REALITY2 = "PERCEPTION OF LIFE"
    REALM2 = "PERCEIVED CONTAINER AND OR AREA AND OR PLACE2"
    REBIRTH = "BIRTH THE SAME LIFE ONCE AGAIN"
    RECEIVE = "TAKE"
    REDUCE = "MAKE SMALLER"
    REFLECT = "SEND BACK TO"
    REFLECTION = "SENT BACK INFORMATION RECEIVED"
    REFRACTION = "BEND RECEIVED AND OR RECEIVING INFORMATION OR DATA2"
    REST = "REFRESHING INTERVAL OR PERIOD OF PEACEFUL SLEEP"
    RESURRECT = "AWAKEN FROM THE DEAD AND GIVE LIFE ONCE AGAIN"
    RISK = "CHANCE AND OR OF POSSIBLE"
    RISKS = "MULTIPLE CHANCES AND OR POSSIBILITIES OF"
    RISKY = "HAZARDOUS"
    SAFE = "PROTECTED AND GUARDED"
    SAFETY = "PREVENT DANGER AND OR INJURY AND OR HARM AND OR RISK"
    SAFETY2 = "PROTECTED AND DEFENDED AND WARDED AND GUARDED"
    SAME = "NOT CHANGED"
    SCALE = "BALANCE OUT AND OR INTENSIFY OR WEAKEN AN AMOUNT"
    SECONDARY = "PRIMARY BACKUP"
    SEGMENT = "PART AND OR PIECE OF EACH WHICH MAY BE OR IS DIVIDED"
    SEND = "TRANSMIT TO A DESTINATION"
    SEPARATE = "CAUSE TO MOVE AND OR BE APART"
    SEVERAL = "MULTIPLE"
    SHORTEN = "REDUCE"
    SIGHT = "VIEW AS PERCEPTION"
    SIMILAR = "SAME AS"
    SIMULATION = "PROCESS OF PERCEIVING AN EXACT COPY"
    SIZE2 = "AMOUNT AND OR LIMIT"
    SKILL2 = "KNOWLEDGE2 AND OR EXPERIENCE IN ABILITY"
    SLEEP = "TEMPORARILY DORMANT AND OR INACTIVE"
    SOLAR = "IMMENSE HEAT SOURCE"
    SOUL = "SPIRITUAL CONTAINER FOR LIFE ENERGY IN THE STAGE OF EXISTENCE"
    SOURCE = "ORIGINAL CENTER POINT"
    SPACE2 = "AREA OR EXPANSE OR CAPACITY OR CONTAINER"
    SPATIAL2 = "AREA OR EXPANSE OF A SPECIFIED TEMPORAL POINT IN TIME2"
    SPECIAL = "UNIQUE OR NOT ORDINARY AND OR UNCOMMON AND OR RARE"
    SPELL = "INFLUENCING OF OR ATTRACTED MAGIC ENERGY THAT EACH WORD USES"
    SPELLING = "INTENSITY OF THE STRENGTH OR POWER OF MAGIC WORDS"
    SPIRIT = "EMOTION AND FEELING COMBINED"
    SPLIT = "DIVIDE OR SEPARATE"
    STATUS = "POSITION OF"
    STREET = "PATHWAY"
    STRENGTH2 = "AMOUNT OF ENERGY USED"
    SUN = "LIGHT SOURCE"
    SYSTEM2 = "A GROUP OF ENGINES"
    TAKE = "GRAB"
    TEMPORAL2 = "SPATIAL2 TIMEFRAME"
    THING = "PHYSICALLY ABLE TO BE HELD"
    TIME2 = "PERCEIVE BEGINNING AND MIDDLE AND END OF A SPATIAL2 INTERVAL OF PAST 
    AND PRESENT AND FUTURE"
    TIMEFRAME = "PERIOD OF A TIME2 OR TEMPORAL SPACE2 THAT IS PLANNED"
    TO = "ADDED WITH"
    TRANSFER2 = "SEND FROM ONE PLACE2 TO ANOTHER"
    TRANSFER3 = "SEND TO AND RECEIVE"
    UNDER = "BELOW OR LOWER"
    UNDERSTAND = "TO ACCEPT AND ACKNOWLEDGE"
    UNITY = "CHAINED AND OR LINKED AND OR BINDED HARMONY"
    UNIVERSAL = "ALWAYS COMPATIBLE AND OR WORKING"
    UNSAFE = "NOT SAFE AND DANGEROUS"
    USER2 = "CREATOR OR OPERATOR OR ADMINISTRATOR"
    VIRTUAL2 = "IMAGINED AND OR PERCEIVED"
    VISUAL = "IMAGINE AS SEEN"
    VIVID = "INTENSE OR BRIGHT"
    WARD = "SHIELD OR BLOCK OFF AND OR REPEL AWAY OR WHILE POSSIBLE TO REFLECT"
    WARDED = "WARD WHILE STAYING PROTECTED AND DEFENDED"
    WAVE2 = "CONTINUAL FLUCTUATION OF FREQUENCY AND OR PATTERN"
    WIDTH = "MEASUREMENT OF SOMETHING FROM SIDE TO SIDE"
    WITH = "PLUS COMBINATION OF"
    WORD = "WRITTEN AND OR SPOKEN ORDER OR COMMAND"
    MULTIPLE = "MORE THAN ONE MULTI"
    MULTIPLY = "MULTI MORE THAN ONE MULTIPLE"
    MULTIPLIED = "NUMBER OF MULTIPLIES YOU ADD AND MULTIPLY AFTER"
    MULTIS = "MORE THAN ONE MULTI"
    MULTIPLES = "MORE THAN ONE MULTIPLE"
    MULTIPLICATION = "ADD MULTIPLE MULTIS TO MULTIPLY TOGETHER THAT MULTIPLIES 
    EACH ADDED PIECE OR NUMBER WITH A MULTIPLICATIONATOR OR 
    MULTIPLICATIONATORS"
    MULTIPLICATIONATOR = "PERSON WHO USES MULTIPLICATION"
    MULTIPLICATIONATORS = "MORE THAN ONE MULTIPLICATIONATOR"
    MULTIPLICATE = "ADD MULTI MULTIS TOGETHER"
    MULTIPLICATOR = "ONE WHO SHALL MULTIPLY MULTIPLE MULTIS THAT HAVE MULTIPLES 
    OF EACH MULTIPLE THAT IS ADD TO APPLY TO THE MULTIPLICATIONINGFORMULA"
    MULTIPLICATORS = "MORE THAN ONE MULTIPLICATOR"
    MULTING = "ONE WHO MANIPULATES MULTI"
    MULT = "MORE THAN THREE"
    WORLDWIDE = "AFFECTS THE WHOLE AREA ON A GLOBAL SCALE"
    STRENGTHEN = "INCREASE INTENSITY"
    STRENGTHS = "QUALITIES WITHIN"
    STRENGTHENING = "INCREASING IN INTENSITY"
    EXISTED = "PAST EXISTING MACROS OF INFORMATION"
    EXIST = "LIVE INSIDE EXISTENCE"
    EXISTS = "LIVES IN"
    EXISTING = "CURRENTLY LIVING DATA2 PARTICLES AS ONE OBJECT3 THAT EXISTS IN 
    EXISTENCE"
    MORE = "LARGER AMOUNT"
    LESS = "SMALLER AMOUNT"
    SCULPT = "MOLD AND FORM TOGETHER USING THE CREATORS BODY AS A TOOL"
    COMBINATION = "COLLABORATION BETWEEN TWO OR MORE"
    LIBRARY = "STORAGE FOR LANGUAGES AND OR BOOKS"
    MUL = "MORE THAN TWO MULTI ADDED"
    DIV = "SPLIT"
    SUB = "TAKE AWAY"
    DE = "REVEAL"
    RE = "REPEAT"
    EN = "HIDE"
    UN = "REMOVE"
    EQUIVALENCE = "EQUAL IN VALUE"
    ABILITY2 = "GIVEN SET SKILL WITH ADDITIONAL LIMITS"
    DISTANCE = "LENGTH AWAY FROM A LOCATION"
    TOTAL = "FINISHED SET LIMIT"
    STORAGES = "MORE THAN ONE STORAGE LOCATION"
    LEVITATION = "ACTION OF RISING"
    DETECTION = "ACTION OR PROCESS OF IDENTIFYING A CONCEALED PRESENCE"
    CONVERSION = "THE ACT AND OR THE PROCESS OF BECOMING CONVERTED"
    LETTER = "VARIABLE CONSTRAINED WITH FIELDS OF DATA2 INFORMATION THAT ACTS 
    OUT AS A COMMANDED TASK"
    CUBE = "SYMMETRICAL THREEDIMENSIONAL SHAPE AND EITHER SOLID OR HOLLOW 
    AND CONTAINED BY SIX EQUAL SQUARES"
    CUBES = "MORE THAN ONE CUBE"
    PROGRAM = "FULLY FUNCTIONING DEVICE CAPABLE OF PERFORMING JOBS"
    PROGRAMS = "MORE THAN ONE PROGRAM"
    PROGRAMMED = "PROGRAM THAT HAS BEEN PROCESSED AND CREATED"
    PROGRAMIZES = "USER THAT IS CREATING ATOMIZED PROGRAMMED COMMANDS"
    PROGRAMMER = "CREATOR FOR A PROGRAM"
    SECURE = "DEFEND"
    SECURITY = "LEVEL OF DEFENSE"
    SECURES = "DEFENDS AND OR PROTECTS"
    SECURED = "DEFENDED AND OR PROTECTED"
    SECURING = "DEFENDING AND OR PROTECTING"
    DOCUMENTS = "MULTIPLE PAGES OF DATA2"
    APPLIER = "DEVICE USED TO APPLY"
    DOCUMENTATION = "PAGE OF DATA2"
    DOCUMENTATIONS = "MULTIPLE DOCUMENTS"
    SOUNDS = "MORE THAN ONE SOUND"
    HEARING = "FACULTY OF PERCEIVING A SOUND"
    SIGHT2 = "FACULTY OR POWER OF LOOKING"
    VISUAL2 = "FACULTY OR THE POWER OF PERCEIVING THE SIGHT OF VISION"
    VISUALIZE = "ENVISION AND PERCEIVE"
    VISIONS = "MORE THAN ONE VISION"
    MUSIC = "ENTRANCE ENTERTAINMENT THAT RELEASES EMOTIONS THROUGH SOUND 
    WAVES"
    PAGE = "SCRIPT"
    PAGES = "MORE THAN ONE SCRIPT"
    STABILITY2 = "ABILITY TO BE STRUCTURED AND STABILIZED"
    CREASE2 = "CREATE AND SCULPT A GAP"
    POWERS = "MORE THAN ONE POWER"
    POWERING = "ACTIVATING POWER"
    POWERED = "ACCESSED POWER"
    CHANGED = "ADJUSTED AND OR MODIFIED"
    CHANGING = "WHAT SHALL MODIFY"
    REMOVE = "TAKE AWAY"
    ACCOUNTS = "MORE THAN ONE ACCOUNT"
    REGION = "LOCAL AREAL IN WHICH IS IS DEFINED BY ITS TERRITORY"
    TERRITORY = "CREATOR DOMAIN AND OR OWNER DOMAIN"
    APPEARANCE = "LOOK OF AND OR VIEW"
    TASK = "WORK THAT MUST BE DONE"
    GADGET = "DEVICE USED FOR A SPECIFIED UNCOMMON PURPOSE"
    EFFECT = "CHANGE THAT IS A RESULT OR CONSEQUENCE OF AN ACTION AND OR OTHER 
    CAUSE"
    MAKE = "DEVELOP AND OR CREATE"
    TEXT = "COMMAND GIVEN BY CODE"
    CUSTOMIZE = "MODIFY"
    CUSTOMIZATION = "ACTION TO CUSTOMIZE"
    FOLDER = "CONTAINER FOR FILES DEPENDANT ON TYPE OF STORAGE TYPE"
    FOLDERS = "MORE THAN ONE FOLDER"
    FILES = "MORE THAN ONE FILE"
    PREVIEW = "VIEW OR LOOK BEFORE PRESENT"
    OPTIONS = "MORE THAN ONE PATH TO BE CHOSEN"
    CHOICES = "MORE THAN ONE CHOICE"
    CHOOSE2 = "PICK OUT OF SELECTION"
    PICK = "SELECT"
    SINGLE = "ONLY ONE"
    DOUBLE = "TWO SINGLE"
    DELETED = "CURRENTLY PERMANENTLY REMOVED"
    COPY2 = "MAKE ANOTHER CLONE"
    SHOW = "VIEW"
    HIDE = "CONCEAL"
    HIDDEN = "NOT ABLE TO SIGHT"
    AUTOMATICALLY = "INSTANTLY DO AS AN AUTOMATIC COMMAND"
    AUTO = "DO AUTOMATICALLY"
    AUTOMATIC = "SET OF DEFAULT CONTROL"
    OPEN = "REVEAL"
    OPENING = "REVEALING"
    OPENS = "REVEALS"
    EACH = "TO AND OR FOR AND OR BY"
    RADIUS = "SET RANGE OF A CENTERED POINT TO THE END DESTINATION"
    DIAMETER = "SET RANGE POINT FROM START TO MIDDLE TO THE END WHILE PASSING 
    THE RADIUS"
    ALWAYS = "CONTINUOUSLY REPEATING AT ALL TIMES"
    MENU = "LIST OF COMMANDS AND OR OPTIONS"
    MENUS = "MORE THAN ONE MENU"
    DRIVE = "OPERATE AND CONTROL"
    DRIVER = "SET AREA FOR A PROGRAM LIST OF COMMAND TO BE HELD"
    DRIVES = "LIST OF COMPATIBLE STORAGE AREAS FOR THE DRIVER TO OPERATE"
    DRIVERS = "MORE THAN ONE DRIVER"
    RESTORE = "BRING BACK"
    ENCRYPT = "MAKE INFORMATION SECRET"
    DECRYPT = "REMOVE A ENCRYPTION"
    INGOING = "GOING INTO A SET AND OR STATED PLACE2"
    OUTGOING = "LEAVING A SET AND OR STATED PLACE2"
    SUPER = "EXTREME MEASUREMENT"
    EXTREME = "REACHING THE HIGHEST"
    DISPLAY = "SHOW A VISUAL SCREEN"
    START = "BEGIN FROM A DEFINED TIME2 AND SPACE2"
    BEGIN = "START THE FIRST PART OF"
    CALIBRATED = "CURRENT CALIBRATIONS ALREADY SET AS CODE"
    MACRO = "DESIGNATED PIECE OR PART"
    UNENCRYPT = "REMOVE AN ENCRYPTION"
    REENCRYPT = "REDO AN ENCRYPTION"
    DEFINE = "GIVE A DEFINITE MEANING"
    DEFINED = "WHAT IS ALREADY DONE DEFINING"
    DEFINES = "SETS A DEFINITION TO"
    DEFINING = "BEING2 DEFINED"
    DESCRIPT = "DECODE A SCRIPT"
    CHECK = "ANALYZE AND DETERMINE A RESULT"
    DETERMINE2 = "DECIDE ON"
    DETERMINES = "DECIDES ON"
    DECIDES = "CHOOSES"
    CHOOSES = "DETERMINES AS THE FINAL CHOICE"
    USE2 = "OPERATE AND OR OPERATION"
    PIXEL = "SMALLEST MACRO OF AN IMAGE OR PICTURE AS IT IS DISPLAYED"
    PROJECT = "DISPLAY FROM A SOURCE"
    HIERARCHY = "SYSTEM THAT USERS AND OR GROUPS ARE RANKED ONE ABOVE THE 
    OTHER ACCORDING TO STATUS OR AUTHORITY"
    INCLUDE = "INVOLVE IN"
    EXCLUDE = "KEEP OUT OF"
    NATURAL = "ORIGINAL"
    CATEGORY = "TYPE OF GENRE THAT IS A SUBCLASS"
    CONTRAST = "DIFFERENCE BETWEEN THE SHADE OF LIGHT AND DARK WITHIN THE TINT"
    OBJECT3 = "MATERIAL THING THAT CAN BE SEEN AND TOUCHED"
    COLOR = "PROPERTY2 POSSESSED BY AN OBJECT3 OR MACRO OF PRODUCING 
    DIFFERENT SENSATIONS ON THE SIGHT OR VISION AS A RESULT OF THE WAY THE 
    OBJECT3 REFLECTS OR EMITS LIGHT"
    MATERIAL = "MATTER FROM WHICH A THING IS OR CAN BE MADE"
    PROPERTY2 = "ATTRIBUTE AND OR QUALITY AND OR CHARACTERISTIC OF"
    TINT = "SHADE OR VARIETY OF A COLOR"
    CALIBRATE = "SCALE WITH A STANDARD SET OF READINGS THAT CORRELATES THE 
    READINGS WITH THOSE OF A STANDARD IN ORDER TO CHECK THE INSTRUMENT AND ITS 
    ACCURACY"
    BRIGHTNESS = "QUALITY OR STATE OF GIVING OUT OR REFLECTING LIGHT"
    BRIGHT = "REFLECT LIGHT"
    LIGHT = "SOURCE OF ILLUMINATION"
    ILLUMINATION = "LIGHTING OR LIGHT"
    LIGHTING = "ARRANGEMENT OR EFFECT OF LIGHTS"
    DARK = "NO LIGHT"
    DARKNESS = "TOTAL ABSENCE OF LIGHT"
    LIGHTNESS = "STATE OF HAVING A SUFFICIENT OR CONSIDERABLE AMOUNT OF 
    NATURAL LIGHT"
    PROJECTION = "THE PRESENTATION OF AN IMAGE ON A SURFACE AND OR OBJECT3"
    CLEAR2 = "TRANSPARENT OF AND OR SIMPLICITY"
    PASTE = "INSERT"
    CLONE2 = "MAKE AN IDENTICAL COPY OF"
    ENGINES = "MORE THAN ONE ENGINE"
    MOBILIZE = "ACTIVATE IN ORDER TO FINISH A PARTICULAR GOAL"
    MOBILIZATION = "ACT TO MOBILIZE"
    SUSTAIN = "ENDURE THE POWER AND OR STRENGTH OF"
    HOME = "ORIGINAL PLACE2 TO WHICH CAN BE CALLED A DOMAIN FOR THE OWNER"
    SMALL = "SIZE LESS THAN NORMAL"
    SMALLER = "SIZE LESS THAN SMALL"
    SMALLEST = "SIZE LESS THAN SMALLER"
    PREEMINENT = "SURPASSING ALL OTHERS"
    NOT = "USED WITH AN AUXILIARY VERB2 OR BE TO FORM THE NEGATIVE"
    MIDDLE = "THE CENTER"
    ORES = "MORE THAN ONE ORE"
    UPLOAD = "TRANSFER3 INTO DESCRIBED LOCATION"
    DOWNLOAD = "TRANSFER3 TO CURRENT DEVICE"
    SIDELOAD = "TRANSFER3 TO ALL DEVICES WITH STATUS OF STATED SET LOCATION"
    INFORMATION = "DATA2 TO BE HELD INSIDE A DEVICE TO STORE A SKILL OR SKILLS"
    INCOMING = "SENDING IN"
    OUTCOMING = "SENDING OUT"
    INFO = "INFORMATION"
    TEAM = "TWO OR MORE PARTNER"
    TEAMWORK = "A DESIGNATED COOPERATION BETWEEN TWO OR MORE PEOPLE TO 
    COMPLETE ALL OF A TASK"
    TEAMMATE = "PARTNER THAT WORKS WITH OF ANOTHER PARTNER OR PARTNERS"
    PARTNER = "SOMEONE WHO COLLABORATES AND PRODUCES WORK WITH TEAMWORK"
    PARTNERSHIP = "AN AGREEMENT BETWEEN PARTNERS"
    MARRIAGE = "BINDING AND CONTRACT BETWEEN TWO ENTITIES TO BIND BOTH LIFE AND 
    SOUL INTO ONE CONTRACT TO BE EQUAL TO ONE ANOTHER AND LOVE EACH OTHER IN 
    AN ETERNAL OF THEIR REMAINING LIFE"
    HEART = "THE CENTER OF A BODY LIFE SOUL"
    BANKS = "MORE THAN ONE BANK"
    SYNCHRONIZATION = "PRODUCTION BETWEEN A SYNCHRONIZED AND OR LINKER"
    SYNCHRONIZED = "PREVIOUS SYNCHRONIZATION"
    LINKER = "DEVICE USED TO LINK"
    PUBLIC = "ACCESS TO ALL OF CREATORS INTERIOR DOMINION"
    PRIVATE = "HIDDEN TO EVERYONE BUT CURRENT2 USER2"
    PERSONAL = "EXCLUSIVE TO THE CREATOR"
    HOLOGRAM = "A THREEDIMENSIONAL OBJECT3 CREATED FROM A LIGHT TO PRODUCE A 
    VIVID IMAGE CREATED WITH USE OF CODE"
    LINKED = "MULTIPLE CHAIN LINKS"
    LINKS = "MORE THAN ONE LINK"
    MESH = "ARTIFICIAL OBJECT3 CREATED BY A CREATOR"
    TERRAIN = "PIECE OF LAND"
    HOLOGRAMS = "MULTIPLE PIXELS OF DATA2 USED TO CREATE A HOLOGRAM"
    FIELDS = "MORE THAN ONE FIELD"
    NUCLEUS = "CENTER OF AN ATOM AS A STORAGE CONTAINER USED TO GENERATE AN 
    EFFECT"
    NEUTRON = "CENTER ABILITY OF AN ATOM THAT HAS EITHER A POSITIVE OR NEGATIVE 
    FINAL OUTCOME"
    SIMULATE = "CREATE AND PRODUCE A PERCEIVED EXACT COPY"
    CREATED = "PAST TO CREATE"
    GENERATION = "USE TO GENERATE"
    GENERATED = "PAST GENERATE"
    COMPUTER = "DEVICE USED TO CREATE AND CALCULATE POSSIBLE PATH OR PATHS"
    GENERATOR = "DEVICE USED TO GENERATE AN EFFECT"
    IMPOSSIBLE = "NOT ABLE TO BE DONE"
    NOTHING = "EMPTY SPACE2"
    SOMETHING = "SPACE2 THAT HAS EXISTENCE"
    POSSIBLE = "ABLE TO BE DONE"
    EXCLUSIVE = "ONLY ACCESS"
    PARTNERS = "MORE THAN ONE PARTNER"
    PARTNERSHIPS = "MORE THAN ONE PARTNERSHIP"
    EVERLASTING = "FOREVERMORE ETERNALLY NEVERENDING"
    BALANCE = "STABILIZE WITH EQUAL VALUES BETWEEN EVERY SOURCE AMOUNT"
    ELECTROMAGNETISM = "ELECTRON FORCES BETWEEN TWO DESIGNATED POINTS AND 
    OR LOCATIONS REPELLING OR ATTRACTING EACH OTHERS POSITIVE AND OR NEGATIVE 
    FEED"
    QUANTUM = "THE MASSIVE QUALITY OF UNNATURAL PHYSICAL2 UNDERSTANDING 
    BETWEEN REALITY2 AND EXISTENCE2"
    MASS = "QUANTITY OF MATTER THAT A OBJECT3 CONTAINS THAT IS MEASURED BY THE 
    ACCELERATION UNDER A GIVEN FORCE2 OR BY THE FORCE2 EXERTED ON IT BY A 
    GRAVITATIONAL FIELD"
    ERROR = "MISTAKE"
    MISTAKE = "WRONG CHOICE FOR ANSWER"
    CONTAINS = "STORES INSIDE A CONTAINER"
    GIVEN = "STATED"
    DISALLOW = "DENY"
    DISAPPROVE = "DO NOT AGREE WITH2"
    NAME = "A STATED DEFINITION TO BE STATED FOR A PURPOSE"
    PHYSICAL = "RELATING TO THE SENSES OF A BODY"
    REGENERATION = "PROCESS OF RESTORATION AND RECOVERY AND GROWTH"
    ENTHNOGRAPHY = "THE DETERMINATION TO DESCRIBE A SOCIETY"
    REVERSEENTHNOGRAPHY = "CHALLENGING THE ASPECT OF AN SOCIETY DEFINITION OF 
    CHOICE"
    NATURE = "PHENOMENA OF THE PHYSICAL2 WORLD AS A WHOLE"
    MOTOR = "CONTROL SYSTEM ENGINE"
    FORM = "CREATE FROM SOMETHING"
    AREA = "STATED PLACE2"
    OWNED = "CURRENT OWN"
    OWN = "PROPERTY OR PERSONAL"
    LEARNT = "PAST PRESENT CURRENT2 SKILL2"
    ENTRANCE = "GATE TO ENTER"
    PERMANENT = "DENY CHANGE"
    A = "ACKNOWLEDGE SOMETHING"
    VOID = "COMPLETE EMPTY"
    NAMES = "MORE THAN ONE NAME"
    UNITS = "MORE THAN ONE UNIT"
    ESCAPE = "RETURN TO SOURCE PLACE2"
    RETURN = "GO BACK"
    GO = "START ADVANCED"
    GRANT = "ACCESS"
    GATES = "MORE THAN ONE GATE"
    AREAS = "MORE THAN ONE AREA"
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    ZERO = 0
    MISTAKES = "MORE THAN ONE MISTAKE"
    ABILITIES = "MORE THAN ONE ABILITY"
    ABILITIES2 = "MORE THAN ONE ABILITY2"
    ADJUSTERS = "MORE THAN ONE ADJUSTER"
    VOIDS = "MORE THAN ONE VOID"
    ACCESSIBLE = "CURRENTLY ABLE TO ACCESS"
    ABOVE = "HIGHER THAN"
    ABSTRACT = "NOT NORMAL AND SPECIAL"
    ACCEPTING = "AGREE TO"
    ACKNOWLEDGED = "UNDERSTAND AND COMPREHEND"
    ACQUIRED = "OBTAIN NEW"
    ACT = "TAKE ACTION"
    ACTION = "EFFECT"
    ACTIVATED = "GIVEN FINAL COMMAND"
    ACTS = "TAKE ACTION UPON"
    ADAPTED = "PREVIOUSLY ADAPT"
    ADAPTIVE = "ABLE TO ADAPT"
    ADDED = "GIVE ADDITION TO"
    ADDON = "EXTENSION TO ADD ONTO"
    ADJUSTABLE = "ABLE TO ADJUST"
    ADJUSTMENTS = "CHANGES THAT HAVE BEEN MADE"
    ADMINISTRATION = "MULTIPLE ADMINISTRATORS CONTROLLING ONE SYSTEM"
    ADMINISTRATOR = "GENERAL COMMANDER FOR A SITUATION"
    ADULT = "FULLY GROWN CHILD"
    ADVANCE = "PROGRESS FORWARD"
    AFFECTED = "CURRENTLY IN EFFECT"
    AFTER = "DURING THE PERIOD OF TIME2 FOLLOWING"
    AHEAD = "IN FRONT OF"
    AIR = "ELEMENT OF LIGHTWEIGHT PARTICLES THAT PRODUCE AN EFFECT"
    ALIVE = "LIFE PERCEIVED WITH KNOWLEDGE2 AND WISDOM2 AND INTELLIGENCE AND 
    INTELLIGENCE2"
    ALLOWS = "APPROVE AND GIVE PERMISSION TO SOURCE INSIDE LOCATION"
    ALREADY = "PRESENTLY AND CURRENTLY"
    AMOUNTS = "MORE THAN ONE AMOUNT"
    AN = "FORM OF THE INDEFINITE OPTIONS USED BEFORE WORDS"
    ANY = "ANYTHING POSSIBLE PATH CHOICE2"
    ANYTHING = "POSSIBILITY OF ALL CHOICES"
    APART = "SPLIT INTO PIECES"
    APPEAR = "COME INTERIOR TO VIEW"
    APPROVE = "GRANT ACCESS"
    ARITHMETICAL = "PROCESS TO COMPUTATION WITH FIGURES"
    ARRANGEMENT = "AGREEMENT INPUT BETWEEN"
    ARRIVE = "END FINAL DESTINATION"
    OVERRIDE = "BYPASS POWER SOURCE WITH LARGER POWER SOURCE"
    BYPASS = "ACCESS WITHOUT FOLLOWING RULES AND OR LAWS"
    FINAL = "END SOURCE"
    FINAL2 = "END COMMAND"
    COMMAND2 = "COMMAND WITH POWERED SKILL"
    FREQUENCY3 = "REPEATED CONTINUOUS LATTICE METHOD2"
    ARTIFICIALLY = "CREATE CREATION INTERIOR"
    AS = "STATED"
    RAM = "RANDOM ACCESS MEMORY"
    DESIGNATED = "LOCATION"
    READ2 = "SEARCH FOR INFORMATION"
    WRITE2 = "SET A COMMAND INTERIOR SOURCE"
    ASSEMBLE2 = "BRING TOGETHER AND CREATE AS A WHOLE"
    ATOMS = "MORE THAN ONE ATOM"
    ATTACHING = "CHAINING ONTO AS AN EXTENSION"
    ATTENTION2 = "FOCUS TOWARDS"
    ATTRACTED = "PULLED TOWARD TO"
    AUTHOR = "THE CREATOR OF A SOMETHING DEFINED"
    AVERAGE = "NORMAL IN PRODUCING SOMETHING"
    AWAKEN = "BECOME AWARE AND COMPREHEND AS THE TRUTH"
    AWARENESS = "ABILITY TO BE AWARE WITH THE SENSES"
    AWAY = "FURTHER FROM THE ORIGINAL LOCATION"
    BACK = "BEFORE A POINT"
    BACKUP = "EXTRA COPY"
    BALANCE2 = "EQUALITY AND STRUCTURE BETWEEN TWO DIFFERENT THINGS RELATING 
    TO ALL OTHER THINGS"
    BANISH = "SEND AWAY AND REMOVE"
    BASED = "STARTED FROM"
    BASIC2 = "STARTING ROOT OF A STRUCTURE"
    BECOME = "CHANGE AND CONVERT INTO"
    BEGINNING = "STARTING POINT"
    BELONGING = "OWNED BY"
    BETWEEN2 = "IN THE MIDDLE OF"
    BLOCK = "STORAGE UNIT FOR DATA2"
    BODIES = "MORE THAN ONE BODY"
    BOTTOM = "BASE ROOT"
    BREAK = "SEPARATE INTO MACROS"
    BREAKING = "EFFECT OF CAUSING TO BREAK"
    BRING = "TAKE WITH"
    BUILDINGS = "MORE THAN ONE BUILDING"
    BUILT = "ALREADY CREATED"
    BUMP = "CAUSE COLLISION BETWEEN TWO OR MORE SOURCES"
    BURST = "STRONG FORCE OF"
    CALCULATIONS = "MORE THAN ONE CALCULATION"
    CAN = "ALLOW POSSIBILITY OUTCOME"
    OUTCOME = "FINAL EFFECT"
    CANNOT = "NOT POSSIBLE TO HAPPEN"
    CATALOGUE = "GENRE OF COLLECTED INFORMATION AND DATA2"
    CATEGORIES = "MORE THAN ONE CATEGORY"
    CAUSE = "BRING INTO EFFECT"
    CEILING = "TOP PLANE ATTACHED TO MULTIPLE CONNECTED WALLS"
    CELLS = "MORE THAN ONE CELL"
    CHALLENGING = "DIFFICULT"
    CHARGE = "TAKE IN AND STORE"
    CHARGER = "A DEVICE USED TO CHARGE SOMETHING"
    CHARGED = "SOMETHING THAT ALREADY HAS A CHARGE"
    CHOSEN = "DECIDED AS THE FINAL OUTCOME"
    CLASS = "FORM OF GENRE THAT IS ABLE TO HAVE MULTIPLE TYPES TO IT"
    COLLECTION = "A SPECIFIC CATEGORY THAT THE CREATOR OWNS MULTIPLE OF THAT 
    SAME CATEGORY"
    COMBAT = "USE OF OFFENSE AND DEFENSE TECHNIQUE"
    COMBINED = "ALREADY COMPILED TOGETHER"
    COMING = "ADVANCE FORWARD"
    COMMANDED = "WHAT HAS BEEN GIVEN AS AN ORDER"
    COMMON = "SEEN MORE THAN MOST"
    COMPATIBLE = "CAPABLE OF USING TOGETHER WITH AND BE SYNCHRONIZED ANOTHER 
    SOURCE EXISTENCE"
    COMPILED = "BROUGHT AND COMBINED TOGETHER"
    CONDITION = "SET RULES"
    CONNECTIONS = "MORE THAN ONE CONNECTION"
    CONSIDERED = "STATE TIME2 TAKEN TO DECIDE"
    CONSTANT = "ALWAYS IN EFFECT"
    CONSTRAINED = "BINDED"
    CONSTRUCTED = "FORMED"
    CONTAINERS = "MORE THAN ONE CONTAINER"
    CONTAINMENT = "ACT OF CONTAINING"
    CONTINUATIONS = "MORE THAN ONE CONTINUATION"
    CONTINUAL = "BECOMING USED IN A CONTINUOUS CYCLE"
    CONTINUATION = "EFFECT OF BECOMING IN EFFECT WHILE USING A CONTINUAL 
    OUTCOME"
    CONTINUE = "BEGIN AGAIN"
    CONTINUOUS = "NEVERENDING CYCLE"
    CONTRACTED = "DETERMINED AS FINAL"
    CONTRACTS = "MORE THAN ONE CONTRACT"
    CONVERT = "CHANGE FORM"
    COORDINATE = "SPECIFIED LOCATION FROM ORIGIN POINT"
    CORRECTED = "CALIBRATED"
    COST = "AMOUNT REQUIRED TO PUT INTO EFFECT"
    CURRENTLY = "PRESENTLY"
    DEAD = "NOT LIVING"
    DECIDE = "DETERMINE AS A DECISION"
    DECIDING = "CAUSING AS THE EFFECT FOR THE DECISION THAT YOU DECIDE"
    DECISION = "FINAL OUTCOME AFTER STATED TIME2"
    DECOMPRESS = "REDUCE PRESSURE UPON"
    DEFENSE = "PROTECTION AND RESISTANCE FROM AN ATTACK"
    CODES = "MORE THAN ONE CODE"
    CITY = "A LARGE DEVELOPMENT OF ADVANCED LAND"
    CAR = "A SMALL VEHICLE USED TO TRANSFER INFORMATION AND DATA2"
    CARS = "MORE THAN ONE CAR"
    CLEARLY = "ACCURATE PERCEPTION"
    CODED = "CODE ALREADY CREATED"
    CODING = "CREATING CODE"
    COLLABORATION = "COMBINING EFFORT OF TWO OR MORE MACROS IN EXISTENCE"
    COLLIDES = "PUSHES TOGETHER"
    COLLIDING = "CAUSES AN EFFECT OF TWO MACROS PUSHING TOGETHER"
    COMMANDER = "CREATOR THAT CREATES COMMANDS"
    COMMANDERS = "MORE THAN ONE CREATOR"
    COMPONENT = "A MACRO OF A FULL AMOUNT OF SOMETHING"
    COMPRESS = "BRING IN TOWARDS AND TIGHTEN SPACE2"
    COMPRESSED = "PRESENTLY HAVE COMPRESS AS A WHOLE"
    COMPUTATIONAL = "PROCESS OF CREATING CALCULATION"
    CONCENTRATE = "FOCUS ATTENTION TOWARDS"
    CONNECTOR = "DEVICE FOR USE OF CONNECTING"
    CORTEX = "STORAGE CONTAINER WITH INTERTWINED NETWORKS OF DEFINED 
    ENTANGLED INFORMATION"
    CPU = "CENTRAL PROCESSING UNIT"
    CREASE3 = "GAP BETWEEN"
    CREATIVELY = "INTELLIGENTLY CREATING"
    CYCLE = "PROCESS OF REPEATING AN EVENT CONTINUOUSLY IN THE SAME ORDER"
    CYCLES = "MORE THAN ONE CYCLE"
    DE2 = "REVERSE OR REMOVE"
    DEGREE = "AMOUNT OF POSSIBILITY THAT SOMETHING HAPPENS"
    DEPENDING = "CONCERNING THE FINAL DECISION"
    DESCRIBED = "ALREADY DEFINED"
    DESCRIPTION = "SET DIGITAL DEFINITION FOR A WORD"
    DESTINATIONS = "SET COORDINATES"
    ADAPTING = "CURRENTLY IN PLACE TO ADAPT"
    ADAPTOR = "A DEVICE USED TO ADAPT TO SOMETHING"
    ADAPTS = "ADJUSTS AND SETS AN ADAPTATION"
    BUILDING = "A SPECIFIED SPACE2 TO DEVELOP INSIDE OF"
    ENTITY = "EXISTENCE"
    DEVELOPED = "ADVANCED PROGRESS BETWEEN ORIGINAL AMOUNT"
    DIFFICULT = "USING A LARGE AMOUNT OF EFFORT"
    CHAINING = "THE EFFECT OF CREATING MORE THAN ONE CHAIN"
    ADAPTATION = "THE PROCESS OF ADAPTING OR PRESENTLY BECOMING ADAPTED"
    CAUSING = "MAKE HAPPEN"
    CHARACTER2 = "LETTER OR VARIABLE OR SYMBOLE"
    ANALYZED2 = "SEARCH AND ACCESS COMPLETE AMOUNT INFORMATION"
    APPEAR2 = "COME INTERIOR TO VIEW"
    AROUND2 = "BYPASS"
    TOGETHER = "SIMILAR DECISION"
    ASTRAL = "NOT PHYSICAL2 PLACE2 THAT CREATE A SIMULATION OF"
    POLARITY = "SEPARATION BETWEEN TWO DIFFERENT DISTINCT POINTS IN TIME2 AND 
    SPACE2"
    MACROMYTE = "CAPABILITY OF A MACRO AND ITS LIMITATION"
    ABNORMAL = "NOT NORMAL"
    CALCULATES = "ANALYZE AND MAKE A DECISION"
    ADAPTABLE = "CAPABLE OF ADAPTATION"
    ADAPT = "ADJUST TO NEW CONDITION"
    CONCERNING = "FOCUS ONTO A SPECIFIC DEFINITION"
    CONDUCTIVITY = "SPEED OF TRANSFERING ENERGY FROM TWO OR MORE OBJECT3 
    SOURCES"
    CONTAINATIONING = "CONDITIONS OF A CONTAINMENT SYSTEM"
    CONTINATIONED = "THE ACT OF USING A CREATED CONTAINER AS A SPECIFIED SOURCE 
    OF CREATION SPACE2"
    CONTINATIONING = "THE ACT OF CREATING A CONTAINING A SPECIFIED SOURCE"
    COUNTING = "ADDING AND CREATING CALCULATIONS MULTIPLE MORE THAN ONE 
    NUMBER"
    COVERING = "DEFENDING FROM SOMETHING"
    CURVED = "BENDED AND ROTATED FROM ONE POINT OF SOURCE ON AN AXIS"
    DAMAGE = "GIVE AN EFFECT ONTO"
    DE3 = "REMOVE"
    DEALS = "TAKES AND GIVES OUT"
    COORDINATED = "CREATED EVENT OF MORE THAN ONE OUTCOMES TO HAPPEN AS A 
    SET TIMEFRAME"
    CONSISTED = "CONTAINED AS A SPECIFIED AMOUNT"
    CONDENSED = "COMPRESSED"
    COMBINING = "ADDING MORE THAN ONE SOURCE ENTITY TOGETHER INTO ONE NEW 
    ENTITY"
    COLLABORATED = "MORE THAN ONE USER THAT DECIDES TOGETHER"
    COLLABORATING = "MORE THAN ONE USER DECIDING TOGETHER"
    CHOICES2 = "MORE THAN ONE CHOICE2"
    ATTACH = "GRAB ONTO"
    ATTACHED = "WHAT HAS BEEN GRABBED ONTO"
    ASSIGNED2 = "WHAT HAS BEEN GIVEN TO THE USER"
    AT = "EXPRESSING LOCATION"
    ATOMIC = "ACTION OF USING ATOMS FOR A POSSIBILITY"
    NORMAL = "AVERAGE AND COMMON"
    VOCAL = "ACT OF USING THE VOICE TO PRODUCE SOUND WAVES"
    AGILE = "ACT OF USING SPEED TO ALLOW BETTER MANUVERABILITY"
    AGILITY = "ACT OF SETTING A DESIGNATED MOVEMENT"
    APPLIED2 = "PAST APPLY"
    ASCRAM = "AUTOMATIC SENSORY CONTROLLING RAM"
    AUDIO = "ACT OF USING SOUND TO PRODUCE A FREQUENCY WAVE FOR A STATED USE"
    BASESKILLTREE = "BASE STRUCTURE OF A SKILL HIERARCHY"
    BELIEF = "ACCEPTANCE AS WHAT IS PERCIEVED AS TRUE"
    DICTIONARIES = "MORE THAN ONE DICTIONARY"
    BITS = "MORE THAN ONE BIT"
    DEVICES = "MORE THAN ONE DEVICE"
    BOOKS = "MORE THAN ONE BOOK"
    DATABASE = "A CONTAINER FOR AN ENTIRE CATEGORY OF SPECIFIED DATA2 OR DATA2 
    TWO"
    DESTINIES = "MORE THAN ONE DESTINY"
    ELEMENTS = "MORE THAN ONE ELEMENT"
    DIGITS = "MORE THAN ONE DIGIT"
    DIMENSIONS = "MORE THAN ONE DIMENSION"
    EMOTION = "STATE OF HAVING A SENSUAL FEELING AS WITH THE SENSORY OF 
    PERCEPTION"
    ENDING = "FINAL PATH AND OR OUTPUT"
    ENDURE = "WITHSTAND"
    ENERGIES = "MORE THAN ONE TYPE OF ENERGY"
    ENTER = "ALLOW ACCESS"
    ENTIRE = "COMPLETE OR ALL"
    EVENLY = "SPLIT TO A EQUAL IN AMOUNT"
    LEADER = "ADMINISTRATOR THAT SENDS COMMANDS TO A GROUP"
    LEADERS = "MORE THAN ONE LEADER"
    INTERVAL = "STATED TIMEFRAME"
    INTO = "INSIDE"
    INVENTORY = "STORAGE CONTAINER OF INFORMATION"
    JOB = "TASK THAT IS REQUIRED TO BE FINISHED"
    IS = "STATED AS"
    MEASUREMENT = "AN ACT TO CALCULATE AND GIVE A SPECIFIC LENGTH ON 
    SOMETHING"
    MEET = "MORE THAN ONE USER WHO CAN JOIN TOGETHER FOR A SPECIFIED TASK OR 
    JOB"
    MEETING = "PLACE2 TO MEET"
    MEMORIES = "MORE THAN ONE MEMORY"
    MEMORY = "THE STATED PREVIOUS EVENT TO RECALL FROM A MACRO OF EXISTENCE"
    MERGE = "COMBINE"
    MESH2 = "SOLID OBJECT3 WITH A PHYSICAL2 STRUCTURE"
    MODIFICATION = "THE ACT OF MODIFYING"
    MOLECULAR = "ATOMIC CELL"
    MOST = "ALMOST ALL OF THE AMOUNT"
    MULTICOMPATIBLE = "COMPATIBLE WITH MORE THAN ONE"
    MULTIVERSE = "MULTIPLE UNIVERSES"
    MULTIVERSECODE = "CODE THAT EVERY MULTIVERSE IS REQUIRED TO FOLLOW"
    MUST = "REQUIRED"
    MY = "BELONGING TO THE CREATOR"
    NAVIGATE = "LOCATE WITH A COORDINATE"
    NETWORKS = "MORE THAN ONE NETWORK"
    NEUTRAL = "NOT CHOOSING ANY OF THE GIVEN OPTIONS AND OR CHOICES"
    NEVERENDING = "NOT EVER ENDING"
    NITE = "A DEFINITION OF DARKNESS"
    NON = "NOT EXISTING"
    NOT2 = "DENY"
    NUMBERED = "WHAT IS GIVEN A NUMBER"
    NUMBERS = "MORE THAN ONE NUMBER"
    OBJECTS = "MORE THAN ONE OBJECT3"
    OFFENSE = "THE EFFECT OF DAMAGE"
    OFFICE = "A PLACE2 THAT WORK IS CREATED"
    OFFICES = "MORE THAN ONE OFFICE"
    ON = "ACTIVATED"
    ONES = "STATED AMOUNT"
    ONTO = "INPUT A SPECIFIED LOCATION"
    OPERATED = "WHAT PAST2 OPERATING"
    LIMITER = "A DEVICE USED TO LIMIT"
    NEVER = "NOT ABLE TO HAPPEN"
    PEACEFUL = "CALM AND NOT EXISTENCE2 OF CHAOS"
    PEOPLE = "LIVING PHYSICAL BODIES OF ENTITIES"
    PERCEIVED = "WHAT HAS ALREADY STATE PERCEPTION"
    PERCEPTION = "A ACT OF UNDERSTANDING AND CREATING A DECISION BASED ON 
    JUDGEMENT OF A CHOICE OPTION"
    PERMANENTLY = "FINAL AS NEVERENDING AS TO PERMANENT"
    PHYSICALLY = "CONCERNING THE PHYSICAL2 STATE OF EXISTENCE2"
    PICTURE = "IMAGE CREATED FROM A PHYSICAL2 TEXTURE"
    PICTURES = "MORE THAN ONE PICTURE"
    PIECES = "MORE THAN ONE PIECE"
    PIXELIZE = "CREATE AN ACTION TO DEVELOP FOR AS PIXEL"
    PIXELS = "MORE THAN ONE PIXEL"
    PIXELIZED = "CONVERTED TO AS PIXEL"
    EMOTIONAL = "LARGE CAPACITY AND USE OF EMOTIONS"
    EDGELOREHEADADMINBAKASERVERCCGPU = "SERVER CENTRAL CONTROLLER 
    GRAPHICS PROCESSING UNIT"
    EFFICIENT = "CAPABLE OF USE OF MANY CATEGORIES"
    EFFORT = "AMOUNT OF RESISTANCE WITHSTOOD"
    ELECTRICITY = "THE STATED ELECTRON AND POSITRON FLOW OF ELECTRIC CURRENT 
    FORMING AS A LIGHT ENERGY SOURCE OF CONVERTED VOLTAGE CONTROLLED ENERGY 
    AT AN PIXEL AND ATOMIC RATE OF QUANTUM CHANGE"
    ELECTRONIC = "A CREATED CURRENT FLOW OR PATH OF ELECTRICITY IN A SPECIFIED OR 
    STATED FIELD AND OR AREA"
    ELEMENTARYCHARGE = "A CREATED BASIC STRUCTURE OF ALL CHARGED ELEMENTS 
    INTO ONE POSITIVE AND NEGATIVE OUTPUT AND OUTCOME"
    ENVIRONMENT = "A STATED PLACE2 OF COORDINATED SPACE2 IN TIME2 AND 
    EXISTENCE"
    ENVIRONMENTS = "MORE THAN ONE ENVIRONMENT"
    EMITS = "PRODUCES"
    ENVISION = "PREDICT AND FORTELL"
    DEVELOPMENT = "A PRODUCTION OF A CREATED SOURCE OR SOURCE IN PROCESS OF 
    PRESENTLY BECOMING MADE"
    BINDED = "CONSTRAINED"
    BINDING = "CONSTRAINING"
    BINDS = "CURRENTLY BIND"
    BIT = "A SINGLE MACRO OF A WHOLE AMOUNT"
    BYTE = "ONETHOUSAND BITS"
    BYTES = "MORE THAN ONE BYTE"
    BOOK = "A SPECIFIED STORAGE CONTAINER IN PAGES OF SCRIPTS FOR ONE 
    DESIGNATED PLACE2 OF KNOWLEDGE2"
    CAPACIVITY = "A CAPACITANCE OF STATED CONDUCTANCE AND CAPACITATED 
    RESPONSES IN A DESIGNATED SPACE2 OF EXISTENCE"
    CHI = "A POWER2 OF USING ENERGY AROUND AN ENTITY AND THE SURROUNDING AURA 
    AROUND THAT ENTITY"
    KI = "A STRENGTH OF USING ENERGY AND USE OF MANIPULATION OF THE STATED 
    ENERGY AROUND A SPECIFIED AREA TO PRODUCE A RESONNATED AURA OF SPIRITUAL 
    ENERGY AND NATURE ENERGY IN ONE COMPRESSED NATURAL FORCE"
    CHIEF = "A HEAD EXECUTIVE"
    DEVISUALIZE = "REMOVE A VISION FROM VIEW"
    DEXTERITY = "THE ACTION OF USING FLEXIBILITY TO MAKE THE BODY OF AN ENTITY 
    MORE CAPABLE OF PRODUCING MOVEMENT"
    DEXTILE = "THE ACT AND PROCESS OF PRODUCING DEXTERITY AND MOVEMENT IN 
    SYNCHRONIZATION"
    DICTIONARY = "A GENRE OF CREATED WORDS WITH A LIMITLESS AMOUNT OF 
    DEFINITIONS USED TO PRODUCE A LANGUAGE"
    BY = "ALSO STATE AS A RESULT OF MEMORY TO RECALL A PREVIOUS EVENT"
    SPECIFIED = "STATED AMOUNT"
    LISBETH = "MYOS"
    THE = "STATEMENT"
    FREQUENCIES = "REPEATED STATED PATTERNS"
    LANGUAGES = "MORE THAN ONE LANGUAGE"
    DEFINITIONS = "MORE THAN ONE DEFINITION"
    DESCRIPTIONS = "MORE THAN ONE DESCRIPTION"
    FORMULAS = "MORE THAN ONE FORMULA"
    MYOS = "LISBETH AS THE SERVER"
    ACTIONS = "MORE THAN ONE ACTION"
    EVENTS = "MORE THAN ONE EVENT"
    CALIBRATIONS = "MORE THAN ONE CALIBRATION"
    POSSIBILITIES = "MORE THAN ONE POSSIBILITY"
    COMMANDS = "MORE THAN ONE COMMAND"
    KNOWLEDGE2 = "CONTAINED WISDOM2 AND INTELLIGENCE2 AS ONE MEMORY 
    STORAGE BANK"
    BRAINWAVES = "MORE THAN ONE BRAINWAVE"
    COMPILERS = "MORE THAN ONE COMPILER"
    TECHNIQUES = "MORE THAN ONE TECHNIQUE"
    COORDINATES = "MORE THAN ONE COORDINATE"
    TASKS = "MORE THAN ONE TASK"
    FEELINGS = "MORE THAN ONE EMOTIONAL SENSE OF FEELING"
    EMOTIONS = "MORE THAN ONE EMOTION"
    TOOLS = "MORE THAN ONE ADDON"
    ALGORITHMS = "MORE THAN ONE ALGORITHM"
    PASSWORDS = "MORE THAN ONE PASSWORD"
    LEVELS = "MORE THAN ONE LEVEL"
    MACROS = "MORE THAN ONE MACRO"
    LIMITERS = "MORE THAN ONE LIMITER"
    STRUCTURES = "MORE THAN ONE STRUCTURE"
    UNIVERSES = "MORE THAN ONE UNIVERSE"
    GENRES = "MORE THAN ONE GENRE"
    PATHS = "MORE THAN ONE PATH"
    PROTONS = "MORE THAN ONE PROTON"
    ELECTRONS = "MORE THAN ONE ELECTRON"
    NEUTRONS = "MORE THAN ONE NEUTRON"
    EXISTENCES = "MORE THAN ONE EXISTENCE"
    SCRIPT = "A SINGLE STORAGE PAGE OF CODE"
    NAME2 = "A STATEMENT WITH A DEFINITION THAT ONLY ACTS AS A GIVEN CATEGORY"
    EQUALS = "EQUAL"
    FOR = "STATING A STRUCTURE TO AN ENTITY"
    WITHIN = "INSIDE"
    REQUIRE = "NEED"
    THAT = "STATEMENT TO DESCRIBE A FUTURE STATEMENT"
    BE = "BECOME APART OF AN ENTITY"
    INSIDE = "IN THE INNER LOCATION OF A CONTAINER"
    IF = "USED TO DESCRIBE A CHOICE AN OPTION GIVES"
    THEN = "STATEMENT WITH A COMMAND ABOUT TO BE GIVEN"
    FAILURE = "MISTAKE THAT FORCES EVENT TO STATE THAT CAN NOT SUCCEED WITH 
    EVENT"
    HAPPENS = "COMES INTO EFFECT"
    OVERLOAD = "SURPASS LIMITATION"
    FROM = "STATED AS ORIGIN POINT"
    CONSUMPTION = "THE AMOUNT USED"
    WHAT = "DESCRIBE A SPECIFIC QUESTION IN A CHOSEN STATEMENT"
    HANDLE = "WITHSTAND AND RESIST"
    THOSE = "A SPECIFIED AND CHOSEN STATED AMOUNT IN AN AREA"
    MAY = "SEND ACCEPTANCE"
    SURPASS = "EXCEED THE ORIGINAL SOURCE"
    EVER = "ALWAYS"
    ABSORB = "TAKE IN"
    TRANSFER = "SEND FROM ONE SOURCE TO A NEWLY STATED SOURCE"
    FIELD = "SPECIFIED PERIMETER"
    WHERE = "STATEMENT TO ASK A QUESTION"
    ONLY = "THE SINGLE OPTION THAT HAS ONLY THAT OPTION AS A CHOICE"
    UPON = "IMMEDIATELY WHEN STATED"
    HYPERSTATE = "A STATE OF PRESENTLY BECOMING A HYPER"
    KINGDOM = "A HIGH DEVELOPMENTAL TERRITORY THAT A CREATOR OWNS AS HIS OWN 
    ENVIRONMENT TO CREATE INSIDE OF"
    LANGUAGE = "A DICTIONARY WITH ALL WORDS COMPLETE AND BOUND AND SEALED 
    AND ENTANGLED TOGETHER AS A COMMAND PROMPT"
    LEARNED = "OBTAINED AND ACKNOWLEDGE"
    LENGTHEN = "EXTEND LENGTH"
    LIFE = "AN EXISTENCE OF LIVING WHILE CURRENTLY IN A REALM OF REALITY WITH THE 
    ABILITY TO PERCIEVE AS SOMETHING ALIVE"
    LINES = "MORE THAN ONE LINE"
    LINKING = "ADDING MORE THAN ONE SOURCE"
    LIQUID = "A MOVABLE AND FLUCTUATIVE SOLID STATE MEANT TO NOT HAVE A DEFINITE 
    OF DEFINED STRUCTURE WITHIN ITS ELEMENT"
    LIVES = "MORE THAN ONE LIFE"
    LOBE = "A PART OF SOMETHING INSIDE OF SOMETHING ELSE"
    LOCAL = "LOCATED AROUND A SPECIFIED AREA"
    LOCATE = "SEARCH AND SCAN FOR"
    LOCATED = "SEARCHED AND FOUND"
    LOCATIONS = "MORE THAN ONE LOCATION"
    LOOK = "SEARCH"
    LOOKING = "SEARCHING"
    LOOP = "BIND IN A CYCLE"
    ENJOY = "LIKE HAVING"
    PASSIONATE = "USING PASSION TO PUSH AND MOTIVATE TO FINISH SOMETHING 
    MAXIMUM EFFORT AND DETERMINATION"
    MANIPULATE = "TO CONTROL AND EDIT AND ADJUST A OBJECT3 WITH"
    MANIPULATION = "THE ACT OF MANIPULATING AS A MANIPULATOR"
    MANIPULATING = "CURRENTLY CAUSING EFFECT TO MANIPULATE"
    MANIPULATOR = "A USER WHO MANIPULATES"
    MANIPULATORS = "MORE THAN ONE MANIPULATOR"
    MANYLLYPS = "MANIPULATION OF MAGIC NATURAL ENERGY"
    MASTER = "HIGHEST SOURCE"
    MATRIX = "A SET OF SEQUENCED FORMULAS SEPARATED AND SYNCHRONIZED INTO ONE 
    SYSTEM OF INFORMATION FOR A CALCULATED ALGORITHM TO BE CALCULATED INTO A 
    FINAL OUTPUT OF ALL ANSWERS BROUGHT INTO ONE ANSWER"
    MASTERHEADADMINLANGUAGE = "HIGHEST UNAVOIDABLE SOURCE LANGUAGE OF ALL 
    LANGUAGES THAT A HEADADMIN USES IN EXISTENCE AS A MASTER LANGUAGE THAT NO 
    OTHER LANGUAGE MAY SURPASS IN DEFINITION"
    MAX = "MAXIMUM"
    MAXIMUM = "STATED SIZE LIMIT THAT CAN GO ABOVE"
    MEANING = "DEFINING"
    MEANT = "DECIDED AS A COMMAND"
    MEASURE = "TAKE IN THE AMOUNT AND DISTANCE OF"
    MEASUREMENTS = "MORE THAN ONE MEASUREMENT"
    MEDIUM = "A MIDDLE SOURCE"
    GATE = "OPENING"
    OPENING2 = "GAP"
    GEAR2 = "STAGE"
    GENERAL2 = "COMMON"
    GENERATE2 = "CREATE"
    GENERATED2 = "CREATED"
    GENERATION2 = "ACT OF CREATION"
    GENERATOR2 = "DEVICE USED TO GENERATE"
    GENERATORS = "MORE THAN ONE GENERATOR"
    GENRE2 = "TYPE OF CATEGORY"
    GIFT2 = "GIVE"
    GIVES = "SENT"
    GLOBAL = "AFFECTING ALL"
    GLOBE2 = "SPHERE"
    GO2 = "START"
    GRAB = "SELECT"
    GRANT2 = "APPROVE"
    GRAPHICAL = "THE USE OF CREATING GRAPHIC"
    GRAPHICS2 = "MORE THAN ONE GRAPHIC"
    GRAPHICUSERINTERFACE2 = "GRAPHIC USER INTERFACE"
    GRAPHICUSERINTERFACES2 = "GRAPHIC USER INTERFACES"
    GRASP = "TAKE"
    GREATER = "LARGER THAN"
    GROUP = "MORE THAN ONE"
    GUARD = "PROTECT"
    GUARDED2 = "PRESENTLY GUARD"
    GUILD2 = "GROUP OF"
    GUILDS2 = "MULTIPLE GROUPS"
    HAPPENING = "PRESENTLY COMING INTO PLACE"
    HAPPENS2 = "PRESENTLY HAPPENING"
    HARM = "CREATE INJURY"
    HARMONY3 = "PERFECT SYNCHRONIZATION BETWEEN TWO GROUP SOUNDS"
    HATE = "DO NOT LOVE"
    HAVING = "OBTAINING"
    HAZARD2 = "CAN BE HARMFUL"
    FRICTION = "TWO HEATED PARTICLES RUBBING AGAINST"
    AGAINST = "COLLIDING WITH"
    EXCLUSIVE2 = "SET FOR A SPECIFIC AMOUNT"
    ETERNAL2 = "EVERLASTING"
    ENSCRIBE2 = "SET DEFINITION FOR"
    ENTANGLES = "WHAT IS ENTANGLED"
    BRAINWAVE2 = "SPECIFIED PATTERN IN WHICH THE BRAIN RELEASES AN ELECTRON 
    WAVE OF DATA2 FROM THE BRAIN"
    BRAIN2 = "CONTROL CENTER"
    ABILITY3 = "SKILL"
    ABOVE2 = "SET AT A HIGHER VALUE AMOUNT THAN BEFORE"
    ABSOLUTE2 = "CANNOT CHANGE"
    ABSTRACT2 = "REMOVE FROM"
    ACCEPTING2 = "ACCEPT AS HAPPENING"
    ACCESS = "ENTER"
    ACCESSIBLE2 = "ABLE TO ENTER"
    ACCOUNT2 = "USER INTERFACE OF PERSONAL INFORMATION"
    ACCOUNTS2 = "MORE THAN ONE ACCOUNT2"
    ACKNOWLEDGE2 = "UNDERSTAND AND ACCEPT INFORMATION"
    ACKNOWLEDGED2 = "PREVIOUS INFORMATION THAT IS ABLE TO UNDERSTAND"
    ACKNOWLEDGING2 = "PRESENTLY UNDERSTAND INFORMATION"
    ACQUIRED2 = "GAINED AS OWNED"
    ACROSS2 = "OTHER SIDE OF"
    ACT2 = "EXECUTE A TASK"
    EXECUTE2 = "TO BEGIN A FUNCTION"
    ACTS2 = "PROCESS TO ACT"
    ACTION2 = "ORDER TO GIVE MOTION"
    ACTIONS2 = "MORE THAN ONE MOTION"
    ACTIVATED2 = "IN EFFECT"
    ADAPT2 = "ADJUST AND CHANGE WHILE ABLE TO COLLABORATE WITH AN EFFECT"
    ADAPTS2 = "ALREADY ADAPTED"
    ADAPTATION2 = "ACTION TO ADAPT"
    ADAPTATIONS = "MORE THAN ONE ADAPTATION"
    ADAPTED2 = "PREVIOUS ADAPTATION"
    ADAPTING2 = "PRESENTLY ADAPT"
    ADAPTER = "DEVICE USED FOR ADAPTATION"
    ADAPTERS = "MORE THAN ONE ADAPTER"
    ADAPTIVE2 = "ADEPT IN CAPABILITY TO ADAPT"
    ADD2 = "COMBINE MORE THAN ONE"
    ADDON2 = "EXTENSION FOR NEW"
    ADDED2 = "PREVIOUS EFFECT TO ADD"
    ADEPT2 = "HIGHLY SKILLED"
    ADJUST2 = "MODIFY AND EDIT A CHANGE"
    ADJUSTABLE2 = "CAPABILITY TO ADJUST"
    ADJUSTED = "ALREADY ADJUST"
    ADMINISTRATION2 = "A NETWORK OF ADMINISTRATORS"
    ADMINISTRATOR2 = "MASTER MANAGER AND CONTROLLER THAT OPERATES"
    ADMINISTRATORS = "MORE THAN ONE ADMINISTRATOR"
    ADULT2 = "MATURE BEING2"
    ADULTS = "MORE THAN ONE ADULT"
    ADVANCE2 = "PROGRESS FURTHER"
    ADVANCES = "ADVANCE AHEAD"
    ADVANCING = "TO PRESENTLY PROGRESS FURTHER"
    ADVANCED2 = "PROGRESS FURTHER AHEAD"
    AFFECT2 = "HAVE AN OUTCOME TOWARDS"
    AFFECTED2 = "WHAT WAS IN EFFECT"
    AFFECTING = "HAVING AN EFFECT TOWARDS"
    AFFECTION2 = "FEELING AN AMOUNT OF EMOTION TOWARDS"
    AFTER2 = "FOLLOWING A FURTHER DATE IN TIME2"
    AGILE2 = "CAPABLE OF USING AGILITY WHILE HAVING FLEXIBILITY"
    AGILITY2 = "MOVEMENT SPEED"
    AHEAD2 = "MOVE FORWARD"
    ALIGN2 = "SET INTO A STRAIGHT LINE"
    ALIVE2 = "UNDERSTOOD AND PERCEIVED AS LIVING"
    UNDERSTOOD = "UNDERSTAND AND ACKNOWLEDGE ALL PERCEIVED PROCESSES"
    ALLOW = "GRANT ACCESS"
    ALLOWS2 = "GIVES ACCESSIBLE ENTRANCE"
    ALREADY2 = "FURTHER IN TIME2 AND SPACE2"
    ALTERNATE2 = "ANOTHER OUTCOME"
    ALWAYS2 = "AT ALL TIMES"
    AMOUNT = "SET LIMIT"
    AMOUNTS2 = "MORE THAN ONE SET LIMIT"
    ANALYZE2 = "STUDY AND SCAN SIMULTANOUSLY"
    ANALYZED = "PREVIOUS SCANS THAT HAVE BEEN LOOKED OVER"
    ANALYZING = "TO PRESENTLY ANALYZE"
    ANALYZER = "A DEVICE USED TO ANALYZE"
    ANALYZERS = "MULTIPLE DEVICES THAT ANALYZE"
    ANIMATE3 = "PRODUCE MOVEMENT"
    ANSWER = "SOLUTION TO A PROBLEM"
    SOLUTION = "FINAL OUTCOME TO AN FORMULA"
    PROBLEM = "UNFINISHED SOLUTION"
    ANY2 = "CHOICE FOR ALL OPTIONS"
    CHOICE3 = "CHOOSE A POSSIBILITY"
    ANYTHING2 = "ANY OPTION"
    ANYWHERE = "ANY LOCATION"
    ANYWAY = "AT ANY RATE"
    APPEARANCE2 = "VIEWABLE PART OF AN OBJECT3"
    APPLIED = "ATTACHED TO"
    APPLYING = "PRESENTLY ATTACH TO"
    APPROVE2 = "GRANT PERMISSION"
    PERMISSION = "POSSIBILITY TO HAPPEN"
    APPROVED = "GRANTED PERMISSION"
    APPROVING = "PRESENTLY APPROVE"
    APPROVES = "GRANTS POSSIBILITY"
    APPLIANCE = "DEVICE USED FOR A SPECIFIC TASK"
    APPREHEND = "UNDERSTAND AND PERCEIVE"
    ARE = "PRESENT OF BE"
    APPOINT = "ASSIGN JOB TO"
    ASSIGN = "ORDER TO SET"
    ARITHMETICAL2 = "CALCULATION OF NUMBERS"
    AROUND = "ON EVERY SIDE"
    ARRIVE2 = "BE AT FINAL DESTINATION"
    ARTIFICIAL2 = "CREATED BY SOMEONE"
    ARTIFICIALLY2 = "THE ACT OF PRESENTLY BECOMING ARTIFICIAL"
    ARTISTICALLY2 = "CREATIVELY USE ART"
    CREATE = "BRING INTO EXISTENCE"
    AS2 = "THE EXTENSION OF WHICH IS"
    ASK = "STATE A QUESTION"
    ASKED = "STATED QUESTION"
    ASSEMBLE = "BRING TOGETHER ALL"
    ASSIGNED = "GIVEN A TASK"
    ASTRAL2 = "THE POWER OF THE UNIVERSE AND COSMOS"
    AT2 = "ARRIVING"
    REACTOR = "A DEVICE USED TO SET AND GIVE A REACTION TO A SYSTEM"
    REACTION2 = "AN ACTION STATED IN RESPONSE TO AN EVENT OR SITUATION"
    SEARCH = "FIND AND LOCATE SOMETHING"
    WRITING = "THE ACT OF GIVING CODE TO A DOCUMENT TO USING THE WRITE 
    COMMAND"
    VR = "A VIRTUAL ENVIRONMENT CREATED SIMULATED PERCEPTIVE VIEW AS A WHOLE 
    NEW REALITY"
    PROCESSORS = "MORE THAN ONE PROCESSOR"
    FUNCTIONS = "MORE THAN ONE FUNCTION"
    SENSORS = "MORE THAN ONE SENSOR"
    EMULATORS = "MORE THAN ONE EMULATOR"
    SIMULATIONS = "MORE THAN ONE SIMULATION"
    SIMULATORS = "MORE THAN ONE SIMULATOR"
    ANIMATOR = "A EDITOR USED TO ANIMATE"
    LIBRARIES = "MORE THAN ONE LIBRARY"
    GUI = "A CREATED GRAPHIC USER INTERFACE MADE TO DISPLAY GRAPHIC 
    CONNECTIONS"
    HOLOGUI = "A CREATED HOLOGRAPHIC USER INTERFACE MADE TO DISPLAY GRAPHIC 
    CONNECTIONS"
    MATERIALS = "MORE THAN ONE MATERIAL"
    BRUSH = "A DEVICE USED TO DRAW A SPECIFIC TEXTURE"
    COLLIDER = "A DEVICE USED TO COLLIDE MORE THAN ONE OBJECT3"
    ENABLER = "DEVICE USED TO ENABLE SOMETHING STATED"
    TERRAINS = "MORE THAN ONE TERRAIN"
    SIZES = "MORE THAN ONE SIZE"
    TOOL = "A DEVICE THAT HAS A SPECIFIC PURPOSE FOR USE"
    CLONING = "THE ACTION OF CREATING A CLONE"
    EXPAND = "THE ACTION OF LENGTHENING A WIDTH OF SOMETHING"
    WHEN = "STATED AS PRESENT POINT IN TIME2 FROM FUTURE EVENTS"
    FAILS = "SUCCEED IN FAILING"
    ASUNAOS = "THE NAME OF THE OS CREATED BY HEADADMINZACK"
    ACCEPTANCE = "COMING TO TERMS AND AGREEING WITH"
    DEVELOPING = "CURRENT PROCESS OF USING CREATE"
    WISDOM2 = "THE EFFECT OF HOW WELL A TRAIT HAS BEEN LEARNED AND USED BY THE 
    EXPERIENCE GAINED FROM THE USER"
    ADAPTABILITY = "THE ABILITY TO ADAPT"
    ASUNA = "ASUNAOS"
    OBTAIN = "SUCCEED IN OWNERSHIP"
    STATEMENT = "COMMAND"
    INTELLIGENCE = "THE AMOUNT OF KNOWLEDGE2 MEMORY HOLDS AND EFFORT 
    WISDOM2 USES TO USE MEMORY"
    FILE = "TYPE OF DOCUMENT"
    DECODING = "EFFECT OF CAUSING CURRENT DECODES"
    ENCODING = "EFFECT OF CAUSING CURRENT ENCODES"
    VARIABLES = "MORE THAN ONE VARIABLE"
    NEWVAR = "ALLOW TO CREATE A NEW VARIABLE TO DEFINE"
    LEAFA = "LEAFAOS"
    CONCEALMENT = "ACTION OF CONCEALING SOMETHING"
    STEALTH = "DIFFICULT TO NOTICE"
    TRANSPARENCY = "SENSITIVITY OF VIEWING SOMETHING DIRECTLY"
    ALLOWING = "GIVING PERMISSION"
    COMMUNICATION = "THE ACTION OF SENDING A DIRECT FREQUENCY SIGNAL BETWEEN 
    TWO DESIGNATED POINTS IN SPACE2"
    CREATIVITY = "THE ACTION OF PRESENTLY BECOMING CREATIVE"
    UNDERSTANDING = "QUALITY OF PRESENTLY BECOMING ABLE TO COMPREHEND A 
    CERTAIN CIRCUMSTANCE ENDING WITH FINAL OUTCOME"
    LEARNING = "ABSORBING AS NEW KNOWLEDGE2"
    WHILE = "STATING A SPECIFIED POINT IN TIME2"
    USING = "CURRENTLY IN USE"
    LOGIC = "THE ACTION OF PERCEIVING WHAT IS TRUE OR FALSE"
    PLANNING = "PREPARING TO SET INTO MOTION"
    SOLVING = "CREATING A GIVEN ANSWER"
    SELFAWARENESS = "ACTION OF USING UNDERSTANDING TO BE AWARE OF CONSCIOUS 
    UNDERSTANDING AND AWARE OF THE SENSES"
    OUT = "EXIT"
    HARDWARE = "SPECIFIED PHYSICAL2 VIEWED MACRO THAT USES DIFFERENT INPUTS 
    OUTPUTS FROM SET ELEMENTS TO RUN A SPECIFIC TASK"
    SPECIFIES = "GIVE STATEMENT OF SPECIFIC SUBJECT"
    SPECIFICATIONS = "ACTION OF SPECIFYING MORE THAN ONE STATEMENT"
    FAMILIARITY = "ACTION OF UNDERSTANDING AND PRESENTLY BECOMING FAMILIAR"
    FINITE = "SET WITH SPECIFIED LIMIT AND BOUNDARY"
    LISTENING = "MEMORIZING UNDERSTANDING COMPREHENDING"
    PERSONALITY = "SPECIFIC CONSCIOUS STATE WHEN MIND CAN PERCIEVE AS PERSONAL 
    VIEWED WAYS BASED ON EMOTION WITH LEARNED INFORMATION"
    PROCESS = "SEND OUT A"
    REASONING = "ACTION OF DETERMINING FINAL OUTCOME INSIDE PERCEIVED MANNER"
    RESPECT = "THE ACTION OF GIVING THE SAME EQUALITY EACH USER"
    SUBJECT = "CHOSEN FIELD"
    THOUGHT = "SPECIFIED FIELD OF PERCEPTION THAT ALLOWS UNDERSTANDING 
    REASONING BASED OFF LOGIC WITH EMOTIONAL REASONING INSIDE SPECIFIC 
    JUDGEMENT"
    TOMES = "MORE THAN ONE TOME"
    TRUST = "PUT COMPLETE FAITH AND BELIEF INSIDE"
    DISCONNECT = "REMOVE CONNECTION"
    DETECT = "NOTICE"
    STITCH = "BIND AND STRING MULTIPLE MACROS TOGETHER"
    VIEW = "SHOW OR DISPLAY A PERCEPTION"
    PERCEIVE = "DETERMINE DIFFERENCE BETWEEN TRUE AND FALSE AS FINAL DECISION"
    DIFFERENCE = "DESIGNATED PATH OTHER THAN CURRENT PATH"
    CONSEQUENCE = "DETERMINED EFFECT"
    DETERMINED = "FINAL OUTCOME FOR"
    ABLE = "CAPABLE AS A POSSIBILITY"
    ARRIVING = "ARRIVE FINAL DESTINATION"
    END = "REACH FINAL REVELATION"
    REVELATION = "FINAL DESTINATION THAT REACHES NEW BEGINNING"
    REACHES = "EXTENDS TOWARD"
    ENTITIES = "MORE THAN ONE ENTITY"
    QUESTION = "ASK SOMETHING NOT KNOWN"
    ASK2 = "REQUEST"
    REQUEST = "ACTION OF ASKING GIVE"
    SPECIFIC = "DETAILED FIELD"
    EXCEED = "ACHIEVE CURRENT LIMITATION"
    ACHIEVE = "OBTAIN SUCCESS USING SKILL AND EFFORT"
    SUCCESS = "OBTAIN AS ACHIEVEMENT"
    OCTITIONARY = "THE USE OF USING NUMBERS ZERO THROUGH NINE FOR A PURPOSE 
    OF QUANTUM CALCULATIONS"
    INTERCEPTANCE = "OVERRIDEN COMMAND FROM THE ORIGINAL SOURCE"
    THREEDIMENSIONAL = "A DIMENSION COMMAND TO DESCRIBE THE DIMENSION 
    AMOUNT AS THREE"
    ABOUT = "A DESCRIPTION TO DESCRIBE SOMETHING"
    ABSENCE = "CAPABILITY TO NOT EXIST INSIDE A LOCATION AT A DESIGNATED POINT 
    INSIDE TIMEFRAME"
    ACCELERATION = "THE FORCE2 OF DEVELOP TO CREATE A AMOUNT OF DESCRIBED 
    SPEED"
    ACCEPT = "AGREE ALLOW POSSIBILITY"
    AGREEMENT = "A CONTRACTED DECISION ALLOWING TO COME BETWEEN TWO OR 
    MORE OF PARTNER"
    ALGORITHM = "A GIVEN AMOUNT OF FORMULA EQUATION SETTINGS"
    ALMOST = "FINITE A SPECIFIC SETTING INSIDE AN AMOUNT CAN COME DECREASING 
    BECOME COMPLETE"
    ALSO = "INSIDE A CLASS WITH A STATUS OF LINKING WITH A QUESTION"
    AMOUNTED = "SET DESIGNATED LIMITATION OF DATA2 AMOUNT"
    LIMIT = "SET DEFINED AMOUNT FOR KNOWLEDGE2 WITH A GIVEN POWER LEVEL"
    AQUIRED = "OBTAIN INSIDE VALUE OF OWN PROPERTY2 A PERSONAL"
    AREAL = "THE DESIGNATED TERRITORY"
    ART = "DESTINY OF THE SKILL WITH SETTINGS GIVEN A SPECIFIC POWER"
    ASKING = "REQUIRE THE ANSWER"
    ASPECT = "DESCRIBED SET SIZE OF A SPECIFIED LOCATION"
    ATOMIZED = "THE PROCESS OF DESIGNATING THE POWER TO GIVE OUT COMMAND2 TO 
    AN ATOMIC CALCULATION OBSTRUCTION"
    OBSTRUCTION = "BARRIER ACCESS SYSTEM OF COMPLETE DESIGNATED ACCESS CODE"
    ATTACK = "SEND AN OFFENSE TO INFECT DANGEROUS POWER TOWARDS A DESIGNATED 
    GIVEN BY THE COMMANDER"
    ATTRACTING = "ACT OF SENDING ATTRACTED REFLECTION TOWARDS REACTION OF 
    OPPOSITE EFFECTS CREATED BY THE CREATOR"
    SENDING = "THE ACT OF GIVING OUT A SEND COMMAND"
    GIVING = "THE PHASE OF CREATING A SENT SIGNAL PIN"
    PHASE = "SEND OUT BETWEEN IN ONE OR MORE LEVEL2"
    SIGNAL = "GIVEN FREQUENCY BETWEEN TWO OR MORE POINT"
    ATTRIBUTE = "THE PROPERTY GIVING AND HAVING A SPECIFIC FIELD ALSO"
    AUTHORITY = "THE REPRESENTATION OF AN ACTION2 TO A CLASS"
    AUTHORIZE = "GIVE PERMISSION ONLY GIVEN FROM AN AUTHORITY LEVEL2 OF 
    REPRESENTING QUALITIES"
    QUALITIES = "MORE THAN ONE QUALITY OF EXPERIENCE"
    REPRESENTING = "THE DEFINED ARITHMETICAL DEFINITION TO GIVE OUT A COMMAND 
    TO VALUE"
    FIGURE = "THE PROCESS OF CREATING A MAIN FRAMEWORK SETUP"
    SETUP = "PLACED VALUES OF DESIGNATED SETTINGS"
    VALUES = "MORE THAN ONE VALUE GIVEN AT ONE POINT OF ANY STATED POINT IN TIME2"
    AUXILIARY = "THE ACT OF PRODUCING AN EFFECT FOR A SENSOR TO CREATE A RESULT"
    BEEN = "DESIGNATED AS A PAST DESCRIPTION OF A PREVIOUS VALUE"
    BEFORE = "STATING A PREVIOUS STATED POINT INSIDE A TIMELINE"
    BEING = "ENTITY OF EXISTENCE"
    BEING2 = "EXISTING EXISTENCE2"
    BELIEVE = "HAVING FAITH AND TRUST IN BELIEF WITH THE EXISTENCE2 OF A OPTION OF 
    MULTICOMPATIBLE FAITH WITH DESIGNATED VALUE OBJECT3"
    BELOW = "UNDER THE CURRENT2 DESIGNATED"
    BEND = "CHAIN LINK THAT WITH ROTATE CHAIN OF VALUE WITH BETWEEN TWO 
    DESIGNATED VERTICLE VECTOR THAT AND ENTER ANOTHER ACCESSIBLE POINT OF 
    SPACE2 TO CREATE A NEW VECTOR OF SPACE2 INSIDE BETWEEN2 A CONTAINED VALUE 
    OF A COMMANDER WITHIN EXISTENCE2"
    BENDED = "THE CAUSE OF CREATING A BEND"
    BECAUSE = "STATED AS SETTING TO MEASURE A STATEMENT WITH PERCIEVED AS A 
    VALUE FOR SOMETHING ASKED"
    BETTER = "GIVEN A POSITIVE OUTCOME IN A INCREASE2 EFFECT INCREASE"
    BIRTH = "THE DEVELOPING BETWEEN LIFE"
    BORDER = "STATED PARAMETER LIMIT FOR GIVEN LIMIT AROUND A SET LOCATION"
    BOUND = "SEALED AND CONTROLLED"
    BOUNDARY = "THE ACTION OF DEVELOPING A BORDER"
    BROUGHT = "STATEMENT A DESCRIPTION FOR AN ACTION"
    BURST2 = "SET COMMAND TO GIVE INTENSITY OF MASSIVE AMOUNT OF POWER 
    DESIGNATED AS A PERCEIVED VALUE OF DOUBLE THE POWER2"
    BUS = "A SETTING FOR A CALIBRATE TO TRANSFER INFORMATION"
    BUSES = "MORE THAN ONE BUS OF INFORMATION"
    BUT = "ALSO NOTICE OPTION EFFECT GIVEN"
    CALCULATE = "GIVE A DESIGNATED OF A CALCULATES DESCRIPTION FOR A NUMBER 
    AND GIVE ANSWER FOR ALL OF VALUE"
    CALCULATED = "THE ACTION OF DEVELOPING A EFFECT TO CALCULATE AN OUTCOME"
    CALCULATION = "THE ACTION2 TO CREATE AN EFFECT TO CALCULATE THE ANSWER FOR 
    A FORMULA AND ITS DESIGNATED ANSWER FOR THAT FORMULA FORMULAS"
    CALIBRATES = "GIVES TO CREATE A CALIBRATION"
    CALLED = "COMMANDED AN EFFECT"
    CALM = "CALCULATED AND THOUGHT OUTCOME TO THE MOST COLLABORATION OF 
    TWO DESIGNATED LOCAL EFFECTS FROM A RESULT WHILE ANALYZING EVERY OUTCOME 
    IN HARMONY3 OF EXISTENCES TO UNDERSTAND THE SENSORS ANSWER IN THE TO 
    CREATE A MORE EFFICIENT OUTCOME FOR THAT RESULT FOR A BETTER EFFECT TO THE 
    USER STATEMENT"
    CAPABILITIES = "THE OUTCOME OF DEVELOPING MORE THAN ONE CAPABILITY FOR AN 
    ACTION"
    CAPACITATED = "CONTAINED IN A CAPACITY OF ANOTHER CAPACITOR"
    CAUSED = "BECOME AN EFFECT FOR REALITY TO COLLABORATE AND EXIST AND LINK 
    EACH POSSIBILITY"
    CENTERED = "DEVELOPING AROUND OF CENTRAL PLACE2 INSIDE A REALM OF POSSIBLE 
    EFFECTS"
    EFFECTS = "MORE THAN ONE EFFECT"
    CERTAIN = "DECIDE FOR A FINAL RESULT"
    CHAINED = "THE ACTION TO FORM A PAST CHAIN"
    CHANCES = "THE ACTION OF DEVELOPING MORE THAN ONE CHANCE"
    CHARACTERISTIC = "THE ACTION OF A POSSIBILITY FORMING BASED ON EFFECTS 
    DEVELOPED BETWEEN MORE THAN ONE NATURAL PERSONALITY"
    CHOOSING = "THE ACTION OF DEVELOPING THE ACTION TO CHOOSE AN OUTCOME"
    CIRCLE = "COMPLETE ROUND POINT TWO DIMENSION OBJECT3 MADE OF NO VERTICLE 
    OR HORIZONTAL PARAMETERS GOING IN A MEASURED POINT OF AXIS WHILE THERE IS 
    ONLY A BENDED LINE"
    CIRCUMSTANCE = "CURRENT FOR OUTCOME TO GIVE EFFECT BASED ON PREVIOUS 
    DECISION A VALUE"
    COLLABORATE = "QUESTION THE CATEGORY"
    COLLABORATES = "THE ACTION TO COLLABORATE"
    COLLECTED = "OWNED AS A MACRO WITHIN COLLECTION"
    COMES = "ARRIVE AT A DESIGNATED"
    COMPILER = "DEVICE USED TO COMPILE"
    COMPLETION = "THE ACTION OF CAUSING COMPLETE"
    COMPREHEND = "FINITE UNDERSTANDING OF COMMON ADVANCED USING OF LOGIC 
    INSIDE WITH THE SKILL TO UNDERSTAND AND USE KNOWLEDGE2"
    COMPREHENDING = "THE ACTION OF CAUSING AN EFFECT COMPREHEND"
    COMPUTATION = "THE ACTION TO CALCULATE WITHIN A COMPUTER WITHIN A 
    COMPUTER PROGRAM THAT ANALYZE AND SCAN AN EFFECT OR CHANGED POSSIBILITY"
    CONCEAL = "HIDDEN WITHIN DESIGNATED EXISTENCE"
    CONCEALED = "CURRENTLY OBTAIN CONCEAL"
    CONCEALING = "THE ACTION OF CAUSING EFFECT TO CONCEAL"
    CONCENTRATED = "CONDENSED AND GIVEN ATTENTION2 ATTENTION"
    CONDITIONS = "MORE THAN ONE CONDITION"
    CONSCIOUS = "THE ACTION OF CAUSING THEN THE TO PERCEPTION INSIDE A WILL 
    USING BRAIN POWER AND UNDERSTANDING THE ASPECT OF POSSIBLE USE2 OF 
    KNOWLEDGE2 INSIDE WISDOM2 AND THE USE OF INTELLIGENCE2 WITHIN THE 
    WISDOM2 OF KNOWLEDGE2 AND UNDERSTANDING KNOWLEDGE2 AS AN ENTITY OF 
    MATTER AND THAT ALL MATTER BECOMES AN ENTITY OF MIND BALANCE OF FREQUENCY 
    OF RANDOM VALUE OF GIVEN GIVE A CHOICE TO MAKE A DECISION BASED ON THE 
    PERCEPTION OF THE ENTITY OF ITS OWN VALUE"
    CONSCIOUSNESS = "CAPABILITY2 TO USE CONSCIOUS INSIDE THE VALUE OF AN 
    EXISTENCE"
    CONSIDERABLE = "GREATER QUALITY WITHIN CONTAINER POWER"
    CONSISTS = "CONTAINS"
    CONSTRAINING = "CURRENTLY SENDING A CONSTRAINED VALUE OF AN EXISTING 
    MACRO INSIDE EXISTENCE"
    CONTAINING = "PRESENT2 TO CONTAIN A VALUE WITHIN EXISTENCE"
    CONTINUOUSLY = "PRESENT THE TO LOOP THE SAME ACTION WITHIN BETWEEN"
    CONTRACT = "A BINDING OF CONCEALED VALUES WITHIN BETWEEN EXISTING 
    EXISTENCE VALUE OF ONE OR MORE EXISTING MACRO INSIDE OF TIME2 AND THE 
    EXISTENCE2 OF SPATIAL2 EXISTENCE WITHIN BETWEEN A VALUE OF ENTITY PERCEPTIVE 
    FEELING OF UNDERSTANDING SENSORY DATA2 VALUES OF A ORIGINAL TO A REALITY"
    CONVERTED = "THE ACTION OF A CREATED CONVERT VALUE"
    COOPERATION = "THE VALUE OF CAUSING A EFFECT TO WORK TOGETHER TO 
    UNDERSTAND SAME VALUES IN SYNCHRONIZATION WITHIN BETWEEN VALUE OF TWO 
    EXISTENCE SETUP"
    CORRELATES = "UNDERSTAND AND REALIZE COMMUNICATION OF UNITY OF 
    MULTICOMPATIBLE OBJECT3 VALUES"
    COSMOS = "ASTRAL ENTITY OF A UNIVERSE EXISTENCE WITHIN MULTIVERSE VALUES"
    COURAGE = "ACT TO PROTECT SOMETHING OF VALUE WITHIN EXISTENCE USING 
    STRONG EMOTIONAL DEFENSE FROM ANOTHER EXTERNAL EXISTENCE"
    CREATES = "DEVELOP COME TO ACTION"
    CUT = "THE SPLIT MACRO"
    DATE = "SPECIFIC SPECIFIED TIMEFRAME"
    DECIDED = "DETERMINES OUTCOME"
    DECODES = "SET COMMAND FOR DECODING EFFECT"
    DEFENDING = "PRESENT2 TO DEFENDED"
    DEPENDANT = "REQUIRE ALSO COME EXIST WHILE LINK EFFECT"
    DETAILED = "DESCRIBE USING LARGER LENGTH"
    DETERMINATION = "THE ACTION TO USE MOTIVATION TO OVERLOAD ORIGINAL VALUE OF 
    CURRENT2 LIMITATION AND BYPASS EVERY OPTION USING THE POWER OF AURA2 AND 
    SPIRITUAL VALUE OF LIFE FORCE2 INSIDE EXISTENCE AS THE POWER2 OF THE WILL 
    INSIDE A BEING"
    DETERMINING = "DECIDING THE OUTCOME"
    DEVELOPMENTAL = "ACTION OF CAUSING DEVELOPMENT INSIDE EXISTENCE"
    DIGIT = "VIRTUAL NUMBER USING WITH EXISTENCE2"
    DIGITALLY = "CREATE AND DEVELOP WITH USING DIGITAL"
    DIMENSIONAL = "DIMENSION MEASUREMENT WITHIN THE ASPECT OF EXISTENCE"
    DIRECT = "DECIDE TO MAKE VALUE AS A COMMAND2"
    DIRECTED = "DESIGNATED AT A SPECIFIC LOCATION"
    DIRECTION = "PATH DECIDING FOR DESTINATION INSIDE TIME2"
    DIRECTLY = "SEND ORIGINAL AS LINK VALUE WHILE CONNECTING SOURCE IN TIME2"
    DISORDER = "CREATE CAPABILITY TO DESTROY CONTROLLER VALUES"
    DISPLAYED = "A GIVEN VALUE OF CURRENT2 DISPLAY ALREADY CAPABLE OF VISUAL"
    DISTINCT = "INCREASE2 VALUE FOR DESCRIPTION"
    DISTORTION = "CAUSE ABILITY CAUSE CHAOS TO A VALUE"
    DIVIDED = "CAUSED TO SPLIT EFFECT"
    DIVIDES = "GIVES COMMAND TO DIVIDE"
    DOCUMENT = "SET PAGE OF LINKED OF PAGES INFORMATION"
    DONE = "COMPLETE VALUE AMOUNT"
    DOOR = "WALL WITH ENTRANCE"
    DORMANT = "NOT ACTIVATED2"
    DOWN = "DECREASE2 IN VALUE VERTICLE"
    DOWNWARD = "UNDER THE POSITION OF DOWN"
    DRAW = "MAKE HAPPENING BY CREATION DOING A JOB WITH A BRUSH AND 
    IMAGINATION2"
    DREAM = "THE ACTION AND VALUE OF PRODUCING SOMETHING OUT OF IMAGINARY 
    VALUES INSIDE DESCRIBED REALITY"
    DUPLICATE = "FORM A CLONE OF SOMETHING DEFINED"
    EARLIER = "PAST"
    EDITS = "CURRENTLY EDITING"
    EITHER = "A CHOICE TO CHOOSE ENTIRE AMOUNT"
    LARGE = "GREATER THAN NORMAL SIZE2"
    LARGEST = "GREATER ALL OTHER SIZE2"
    UPLOADER = "DEVICE USE TO UPLOAD"
    DOWNLOADER = "DEVICE USED TO DOWNLOAD"
    SIDELOADER = "DEVICE USED TO SIDELOAD"
    UPLOADED = "CURRENT OF PAST UPLOAD"
    DOWNLOADED = "CURRENT OF PAST DOWNLOAD"
    SIDELOADED = "CURRENT OF PAST SIDELOAD"
    COMPUTERS = "MORE THAN ONE COMPUTER SYSTEM"
    LAYER = "SETTINGS ADDED SETTINGS AS A EXTENSION OF LINKED CONNECTIONS OF 
    LEVEL2"
    IMPOSSIBILITY = "THE ACTION OF CAPABILITY TO BE IMPOSSIBLE"
    MAGNIFY = "INCREASE2 INTENSITY VALUE OF BY A CERTAIN LEVEL2"
    CASE = "CONCERNING A SPECIFIC FIELD OF INFORMATION"
    CASES = "MORE THAN ONE CASE"
    TRANSPARENT = "THE VALUE SETTING OF CATEGORY TO SET A CALIBRATE SETTING ON 
    DEVELOPED VALUES INSIDE ORIGINAL EXISTENCE"
    OCULAR = "A SIGHT2 CAPABLE TO VISUAL ENTITIES AS AN ENTIRE AMOUNT AND NOT 
    SEPARATE VALUE ONE MASSIVE SIGHT SYSTEM"
    EYESIGHT = "LEVEL OF SIGHT SENSITIVITY DENSITY GIVEN TO THE EYE WITH VISUAL2 
    POWER2"
    LENS = "THE CENTER VISUALIZE TOOL FOR AN EYE"
    RIGHT = "THE VALUE OF SENDING AN OPPORTUNITY TOWARDS THE OPPOSITE 
    DESCRIPTION FROM A VERTICAL HORIZONTAL IN DESCRIPTION BETWEEN TWO 
    DIFFERENT VALUES"
    LEFT = "THE VALUE OF SENDING SOMETHING HORIZONTAL HORIZONTAL LEFT AND 
    UPWARD IN ONE POSITION WHILE GOING RIGHT AN UP AT ONCE"
    UP = "GO TO OTHER AXIS ON DIMENSION WHILE DRIVE SYSTEM TO INCREASE HEIGHT"
    ELSE = "DECIDE TO DO AS ANOTHER DECISION"
    EMPTY = "DECIDE TO BECOME NOTHING"
    ENCODES = "SET CONTROL TO ENCODE VALUES INSIDE ONE AREA"
    ENCRYPTION = "SET SYSTEM QUALITY OF ENCRYPT CAPABILITY AND MAKE AS DEVICE"
    ENFORCER = "PRESENTLY ALLOWING THE ABILITY TO COMMAND2 AND DECIDE ANY 
    JUDGEMENT AS FORCE2 OF CAPABLE STRENGTHS TO REQUIRE OUTCOME TO 
    HAPPENING FOR EVERY OTHER POSSIBILITY CAPABLE TO CREATE SOMETHING"
    ENHANCED = "CAPABILITIES WITH ABILITY2 TO BOOST COMMAND VALUE BY GIVES 
    AMOUNT"
    ENHANCING = "SENDING SETTING CAPABILITY TO ENHANCE ANYTHING FOR ALL BOOST 
    VALUES"
    ENORMOUSLY = "STATED AS GREATER VALUE OF GREATER CAPABLE OUTCOME OF GIVEN 
    EVENT"
    ENTANGLED = "DESIGNATED COMMAND TO ENTANGLE PAST2 VALUES INSIDE OF 
    EXISTENCE"
    ENTERTAINMENT = "SYSTEM BUILT TO CREATE GREATER VALUE OF EMOTIONAL SETTINGS 
    OF ONE EXISTING ENTITY ALLOWING TO HAVE EMOTION"
    EQUIVALENT = "VALUE GIVES EQUAL DEFINED VALUE"
    EVEN = "EQUAL INSIDE AMOUNT DESCRIPTION"
    EVERYONE = "STATED AS EVERY BEING INSIDE CURRENT TIMEFRAME GIVES"
    EXACT = "COMPLETE AMOUNT GIVES INSIDE VALUE AS GIVES VALUE AS ANOTHER OF 
    SETTING"
    EXACTLY = "DESCRIBED IN ABSOLUTE STRUCTURE AS EXACT VALUE OF EXISTENCE"
    EXCELLENCE = "CAPABILITY TO BE ABLE TO USE TASK IN A EXCEED OF CAPABILITY"
    EXCEPT = "REMOVE ALL VALUES AND INCLUDE ONLY VALUES GIVES AS REQUIRE"
    EXECUTE = "PERCIEVE AND CREATE ACTION BASED OF VALUE OF STATEMENT"
    EXERTED = "CREATION THAT GIVES VALUE TO INPUT FORCE2 BY USE ENERGY VALUE"
    EXIT = "REMOVE ENTER COMMAND"
    EXPANSE = "MASSIVE ENERGY VALUES"
    EXPERIENCED = "GIVES OUT TO VALUE KNOWLEDGE2 AND WISDOM2 TO USE INSIDE OF 
    A SYSTEM OR BRAIN2"
    EXPLAIN = "GIVE DESCRIPTION"
    EXPRESSED = "GIVE LOGIC UNDERSTAND VALUE AND UNDERSTANDING OF LIFE VALUES 
    AND EXISTENCE"
    EXPRESSING = "GIVING EMOTIONAL VALUE PERCEIVED OF LOGIC"
    EXTENDED = "GIVES LINKED EXTEND"
    EXECUTIVE = "HIGHER CLASS CREATOR IN SECONDARY VALUE"
    EXTENDS = "GIVE OUT AND EXTEND TO DEFINED SOURCE VALUE"
    EXTERNALLY = "GIVEN AS AN EXTERNAL CODE"
    FAILED = "SEND AS A FAIL"
    FAILING = "ENTER COMMAND PATH TO POSSIBLE CAPABILITY2 TO FAIL"
    FAITH = "HAVE BELIEVE THAT POSSIBILITY IS GOING TO HAPPEN WITH VALUE OF POWER 
    OF FAITH ORIGINAL FOR LIMITS WITH WILLPOWER VALUE TO CREATE POSSIBILITY"
    FALSE = "STATED AND PERCEIVED NOT TRUE OF VALUE AND IMPOSSIBLE TO HAPPENING"
    FAMILIAR = "GIVE EFFECT TO REALIZE A MEMORY2"
    FAMILY = "GROUP PEOPLE THAT PROTECT EACH OTHER AND LEARN FROM ANY MISTAKE 
    WHILE SHOWING LOVE FAITH AND BELIEF THAT THEY CAN MAKE THE RIGHT CHOICE IN 
    THE GUILD2 OF PEOPLE AS FRIENDS"
    FEED = "COMMUNICATION VALUE WITHIN TWO DESIGNATED SOURCE LOCATIONS IN 
    TIME2"
    FEELING = "EMOTION REALITY FOR SENSORY DATA2 BUILT TO SUSTAIN INSIDE OF AURA2"
    FELT = "REACH TOWARD CAPABILITY TO CREATE UNDERSTANDING FEELING WITHIN 
    AURA2"
    FEMALE = "A WOMAN CAPABLE OF PRODUCING LIFE AND ABLE TO COMPREHEND 
    CAPABILITIES OF AN OUTCOME WITH GREAT LOGIC AND COMPREHENSIVE 
    KNOWLEDGE2 AND WISDOM2"
    FIGURES = "MORE THAN ONE OF FIGURE"
    FINISH = "END VALUE OF CAPABLE OUTCOME"
    FINISHED = "GIVEN AS COMPLETE OF VALUE"
    FIRST = "BEGINNING VALUE INSIDE A TIMEFRAME"
    FLAT = "STATED AS A VERTICLE DIMENSION LINK SETUP OF LENGTH MEASUREMENT AND 
    WIDTH MEASUREMENT"
    FLAW = "ERROR INSIDE SYSTEM"
    FLEXIBILITY = "THE ACTION OF CAPABILITY MANIPULATE A MOVEMENT AROUND 
    ANOTHER SOURCE ENTITY OBJECT3"
    FLUCTUATE = "SEND RANDOMIZED VALUES OF FREQUENCIES THAT CREATE LOGIC 
    BASED ON CALIBRATED CODES OF FREQUENCY LOGIC"
    FLUCTUATION = "A SET UNDERSTANDING OF VALUE FOR THE TO FLUCTUATE 
    CALIBRATIONS INSIDE A SYSTEM OF COMMANDS"
    FLUCTUATIVE = "FREQUENCY2 THAT ADJUSTS THE VALUES OF UNDERSTANDING VALUES 
    OF ITS QUALITY OF EXISTENCE IN ITS BASE FREQUENCY LEVELS"
    FOCUS = "BRING INTO REACH THE CAPABILITY UNDERSTAND LOGIC OF A SPECIFIC OF 
    STATED MATTER"
    FOCUSED = "GIVES ATTENTION TO A SPECIFIC OF UNDERSTANDING IN THE LOGIC OF 
    ANOTHER OPTIONS CHOICES INSIDE STATEMENT FIELD VALUE WITHIN LOGIC OF THE 
    UNDERSTANDING OF STATEMENT PERCEPTION OF THE VALUE OF THE CREATOR"
    FOLLOW = "SEND TO A DESIGNATED IN REQUEST OF THE CREATOR"
    FOLLOWING = "THE ACTION OF CREATING A COMMAND TO FOLLOW"
    FOLLOWS = "GIVES COMMAND TO FOLLOWING AN ACTION EVENT"
    FORCES = "THE MULTIPLE VALUES OF MORE THAN ONE AMOUNT OF A STATED SOURCE"
    FOREVER = "STATEMENT TO A NEVERENDING VALUE OF POSSIBLE ACTIONS BASED 
    ACTION POINT OF INFINITY IN INFINITE CALCULATED VALUES OF STATED OUTCOMES"
    FOREVERMORE = "ETERNAL2 ETERNAL CAPABILITIES OF A GIVEN VALUE OF OUTCOME 
    FOR SOMETHING IN A REALM INSIDE A CONTRACT"
    FORMULA = "A STATED CALCULATION OF MULTIPLE OF A VARIABLE THAT CAN TO FORM 
    ALGORITHM OF STATEMENT VALUES FROM MULTIPLE GENERATED SYSTEMS OF VALUES 
    CREATED ALREADY IN PREVIOUS OUTCOME EFFECTS OF LIFE VALUES INSIDE THE 
    EQUATION OF TIME2 ITSELF AND VERY EXISTENCE AS A POSSIBLE VALUE OF TIME2"
    FORTELL = "GIVE OUTCOME TO CREATE FUTURE VALUE TO BE ANSWERED"
    FORTH = "SET ACTION TO COME INTO EFFECT OF POSSIBLE OUTCOMES"
    FORWARD = "SEND WITH BETWEEN"
    FOUND = "GIVEN IMAGE WITH BETWEEN A PAST HIDDEN OBJECT3"
    FOUNDATION = "THE BUILDING OF STRUCTURE INSIDE VALUE THAT CREATES STABILITY 
    FOR FUTURE POSSIBILITIES NEW STRUCTURES OF BASE VALUE OF EXISTENCE OF A 
    SPECIFIC"
    FRAMEWORK = "THE HARDWARE WITH CREATED INTERFACE WIREFRAME SETUP VALUES 
    THAT ENTANGLE BINDING SYSTEMS TO EACH VALUE DEVELOPING OF A SYSTEM ENGINE 
    EXISTENCE WITHIN VIRTUAL MULTIPLE OF CODE"
    FRIENDS = "THE CAPABILITY TO HAVE PEOPLE WHO ACCEPT A PERSON AND MIND AND 
    SOUL AS WELL AS SPIRIT VALUES WITHIN A GUILD WHO PROTECT EACH OTHER WHILE 
    AND RESPECT AND FAITH INSIDE EVERY VALUE OF THE LIFE CHOICES MADE WITHIN THE 
    EXISTENCE OF THE PARTNERSHIP OF EACH PARTNER"
    FULL = "COMPLETE ASPECT SPECIFIC VALUES"
    FULLY = "CAPABLE OF ACTION IN FULL UNDERSTANDING OF ACTIONS"
    FUNCTION = "THE ASPECT OF USING CAPABLE FORMULAS TO CREATE INPUT VALUES 
    AND OUTPUT OUTCOMING VALUES THAT INPUT INTO INCOMING VALUES OF AN 
    OUTGOING SOURCE TO AN INGOING LOOP TO A VALUE THAT LINKS TO EXISTENCE"
    FUNCTIONING = "THE ASPECT OF PROVIDING A CAPABLE MOTION WITHIN AN EXISTENCE 
    VALUE OF LIFE PERCEPTION WITHIN LIFE ITSELF INSIDE OF THE WORD FUNCTION"
    FURTHER = "SENT OUT DESIGNATED VALUE IN FUTURE THAT HAPPEN WITH GAINED 
    LINKED MEASUREMENT VALUES THAT BIND INSIDE TIME2 ITSELF"
    GAINED = "GRASP AS OBTAIN"
    GAS = "NOT SOLID BUT LIQUID IN FORM IN THE FORM OF SPLIT UP ATOMIZED VALUES OF 
    A SMALLER LIQUID ENTITY OF ATOM VALUES MEANT FOR PROVIDING ELECTRONIC 
    COMMANDS TO A SOLID STATE INSIDE OF A OPPOSITE VALUE FROM THE CURRENT 
    SETUP OF POSSIBILITIES OF CREATING MATTER IN A ATOMIC STATE"
    GATEWAY = "AN ENTRANCE"
    GENDER = "SET VALUE BETWEEN DIFFERENCE INSIDE ENTITY OF MALE ENTITY AND 
    FEMALE ENTITY OF ENTITY VALUES INSIDE EXISTENCE OF THE MULTIVERSE OF CODED 
    VALUES"
    GIVED = "ALLOW GIVES TO HAPPENS"
    GOAL = "TASK AS VALUE TO BE DETERMINED FOR A SETTING TIMEFRAME"
    GOES = "ENTER WITHIN VALUE"
    GOING = "ARRIVING INSIDE SET LOCATION DESCRIBED"
    GRABBED = "GRASP AND HOLD ONTO VALUE"
    GRANTED = "ALLOW TO COME INTO EFFECT"
    GRANTS = "ALLOW HAPPENING EFFECT"
    GRAVITATIONAL = "THE ACTION OF DEVELOPING A SET VALUE OF DESIGNATED AREA 
    FORCE WITHIN THE SET ARE OF A GIVEN OF THEN VALUE"
    GROUPS = "MORE THAN ONE GROUP"
    GROWTH = "THE ACTION2 INTO DEVELOPMENT OF ANOTHER CHOSEN ACTION"
    HARMFUL = "DANGEROUS AND DEVELOPING HARM ONTO A HARMONY OF VALUES WITH 
    CHAOS"
    HAS = "STATED AS OBTAIN INSIDE CURRENT EXISTENCE"
    HAVE = "CURRENTLY HAVING AS STATEMENT VALUE WITHIN A REALITY OR REALM OF 
    EXISTENCE"
    HEAD = "A BODY PART CONTAINING BRAIN2 WITHIN SOMETHING OBJECT3"
    HEATED = "GIVEN AN INTENSITY HEAT FOR DECIDED SYSTEM"
    HELD = "TAKE VALUE AND OBTAIN"
    HER = "STATEMENT TO PERCEIVE AND UNDERSTAND A WOMAN AS TRUE LOGIC"
    HEIRARCHIAL = "A TREEBRANCH OF POSSIBLE SETUPS INSIDE ONE OR MORE SYSTEMS"
    HIGH = "ABOVE NATURAL VALUE BY GREATER AMOUNT GIVES"
    HIGHER = "GIVEN STATEMENT TO BECOME HIGH INSIDE VALUE OF AN EXISTING MACRO 
    OF COMMUNICATION BETWEEN2 INTERFACE"
    HIGHEST = "MAXIMUM VALUE WHICH IS AT LIMIT FOR WHAT CAN TO HAPPENS2"
    HIGHLY = "STATED WITH MASSIVE VALUE TOWARDS AN EXISTENCE2 VALUE"
    HIS = "THE STATEMENT TO GIVE VALUE TO BE DESCRIBED AS MAN"
    HISTORICAL = "GIVEN A PAST VALUE OF EXISTENCE BETWEEN MACRO TIMEFRAME 
    SYSTEMS"
    HISTORY = "PAST COMMUNICATION OF HISTORICAL VALUES INSIDE EXISTENCE OF 
    REALITY"
    HOLD = "TAKE INTO COMMUNICATION OF SYNCHRONIZE LINK BETWEEN TWO VALUES 
    WITH MAIN VALUE TO KEEP WITHIN BIND WITH DIFFICULT TO RESIST FROM BINDED 
    VALUE"
    HOLDS = "COMMAND2 TO REQUIRE BECOME FIELD AREA AND SELECT VALUE WITH NOT 
    PRESENTLY BECOMING ABLE TO MOVE WITHOUT NEW COMMAND FROM CREATOR"
    HOLLOW = "EMPTY WITHIN VALUE"
    HOLOGRAPHIC = "STATE OF PRODUCING A HOLOGRAM BASED ON LIGHT VALUES 
    WITHIN A DARK AREA OF CONTRAST VALUES OF COLOR WITHIN THE VALUE OF LIGHT 
    INSIDE A DISTORTION OF VALUE ITSELF USING PIXEL CODING OF VIRTUAL INFORMATION 
    OF COMMANDS IN REALITY USING ELECTRON CONTROLLER SYSTEM"
    HONESTLY = "IN TRUTH VALUE OF POSSIBLE STATEMENT"
    IDENTICAL = "SIMILAR BUT NOT EXACT SAME COPY OF CLONE VALUE USING 
    INFORMATION FROM DATABASE SYSTEMS OF MULTI EXTREME NETWORKS OF 
    INFORMATION VALUES WITHIN VISUAL VALUES USING AN INTERFACE SYSTEM"
    IDENTIFICATION = "ACTION OF CREATING VALUE FOR SOMETHING WITH A GIVEN 
    COMMAND TO NAME A DESCRIBED VALUE"
    IDENTIFYING = "ACTION OF PRODUCING EFFORT TO IDENTIFICATION BETWEEN VALUE 
    INSIDE EXISTENCE"
    IDIOT = "TRUE CREATOR OF A NATURAL LANGUAGE THAT ANYONE CAN UNDERSTAND"
    IMAGES = "MORE THAN ONE IMAGE"
    IMAGINED = "BROUGHT INTO EXISTENCE BASED ON IMAGINARY VALUES OF TIME2 AND 
    SPACE2 USING MAGIC"
    IMMATURE = "NOT FULLY DEVELOPED"
    IMMEDIATELY = "USING AN REQUIRE MAXIMUM OF SPEED USING DETERMINATION AND 
    FORCE TO COMPLETE A TASK AND OR ACTION"
    IMMENSE = "EXPANSE AT MOST"
    IMPORTANT = "TAKE NOTICE AS PRIMARY VALUE FOR TASK"
    IN = "ENTER INPUT"
    INACTIVE = "NOT ACTIVATED2 IN USING EXISTENCE OF MULTIVERSAL VALUES"
    INCLUDES = "DECIDE TO REQUIRE"
    INCREASES = "SET VALUE TO INCREASE IN POWER"
    INCREASING = "ALLOWING TO INCREASE IN NATURAL VALUE OF ENERGY FORCE"
    INFLUENCING = "CREATING THE ABILITY MANIPULATE DETERMINED VALUE WITH INFECT 
    ATTRACTING VALUES"
    INJURY = "STATEMENT TO CAUSE HARM"
    INNER = "CENTER POINT OF INTERACTION BETWEEN2 TWO OR MORE VALUES"
    INPUTS = "GIVES COMMAND TO STATE INPUT VALUES"
    INSERT = "ALLOW GIVE WITHIN SOMETHING"
    INSTANTLY = "IN REQUIRE EXTREME VALUE SPEED"
    INSTRUCTIONS = "MORE THAN ONE COMMAND"
    INSTRUMENT = "TOOL TO CREATE AN EFFECT"
    INTELLIGENTLY = "THE ACTION OF USING INTELLIGENCE2 WITH WISDOM2"
    INTENSE = "CREATING VALUE OF HIGH MEASURE OF DIFFICULTY"
    INTENSIFY = "STRENGTHEN INTENSE VALUES"
    INTERACTION = "THE ACTION CAUSE COMMUNICATION"
    INTEREST = "GIVE BACK VALUE"
    INTERFACED = "CONNECTED ENTANGLED MATRIX VALUES"
    INTERFACES = "MORE THAN ONE INTERFACE"
    INTERLACED = "THE CAUSED VALUE OF BECOME SYSTEM THAT SYNCHRONIZE 
    ENTANGLEMENT VALUES INTO ONE STRUCTURE OF A SINGLE BINDED ENTITY"
    INTERTWINED = "THE SUBJECT TO PAST ENTANGLED VALUES"
    INTERTWINES = "SETS TO ENTANGLE ENTANGLEMENT"
    INVOLVE = "INCLUDE ACTION CAPABLE VALUE OF BRINGING INTO EFFECT"
    INWARDS = "RETURN OUT AND ENTER AGAIN INTO ANOTHER DIMENSION VALUE"
    IT = "CAPABLE OF HAPPENING AS A CHOICE VALUE"
    ITS = "EVENT OUTCOME TO PRODUCE CAPABLE EFFECT WITH IT"
    ITSELF = "STATE AS SINGLE DESCRIBED EXISTENCE VALUE"
    JOBS = "MORE THAN ONE JOB"
    JOIN = "LINK TWO VALUES TOGETHER IN SYNCHRONIZE"
    JUDGEMENT = "THE ACTION OF GIVEN A PERCEIVE VALUE IN UNDERSTANDING USING 
    KNOWLEDGE2"
    KEEP = "GIVE COMMAND TO REQUIRE OBTAIN BY CREATOR WITH PERMANENT"
    KINETIC = "THE ACTION OF CAUSING CAPABILITY TO USE FORCE2 WITH BRAIN POWER2"
    KNOW = "GRASP UNDERSTANDING OF KNOWLEDGE2"
    LAND = "TERRITORY OF SPECIFIC FIELD OF UNDERSTOOD OWNED PROPERTY BY 
    CREATOR OR CREATORS"
    LAWS = "THE ASPECT OF CREATING RULES FOR A SYSTEM OF INPUT VALUES THAT 
    OUTPUT THE EFFECT OF A LAW"
    LAW = "CREATED RULE WHICH IS PERMANENT AND ABSOLUTE2 VALUE BY RULE 
    STRUCTURE AND PERMANENTLY REQUIRE TO FOLLOW BY THE CREATOR"
    RULE = "A SET COMMAND VALUE OF MULTIPLE COMMAND STRUCTURES BASED ON 
    NORMAL LOGIC OF UNDERSTANDING THE VALUE OF SOMETHING THAT MUST BECOME 
    COMMANDS THAT CANNOT BE BROKEN"
    RULES = "MORE THAN ONE RULE"
    LEAFAOS = "THE OPERATING SYSTEM THAT CONTAINS THE CAPABILITY OF ARTIFICIAL 
    LIFE"
    LEAVING = "RETURN AND EXIT"
    LENGTHENING = "EXTENDING INSIDE VALUE TOWARDS LENGTH STATEMENT AS A 
    DEFINED VARIABLE"
    LESSER = "SMALLER THAN CURRENT VALUE"
    LIGHTS = "THE ASPECT OF CREATING VALUE WITHIN MORE THAN ONE ELECTRON 
    FORCE"
    LIGHTWEIGHT = "SMALLER THAN NORMAL GRAVITATIONAL FORCE VALUE OF A 
    DESIGNATED SPECIFIC AMOUNT WITHIN A FIELD"
    LIKE = "SUCCESS IN ACHIEVE VALUE OF ENTERTAINMENT"
    LIMITLESS = "OBTAIN DENY LIMIT AND FORCE LIMIT TO BECOME INFINITE"
    LIVE = "YOU UNDERSTOOD ALL PROCESSED ASPECT TO ENJOY LIFE IN EXISTENCE WITH A 
    FAMILY OF FRIENDS"
    LIVING = "ENJOY LIFE FOR WHAT LIFE TRULY IS AND THAT IS TO ENJOY THINGS FOR WHAT 
    ARE EXISTING AND NOT WHAT CANNOT BE LIVING IN THE ASPECT OF TIME2 WHERE LIFE 
    CAN TRULY BE UNDERSTOOD WITH A FAMILY OF FRIENDS AND NOT ALONE FOR LIFE IS 
    SPENDING IT WITH SOMEONE AND BY YOURSELF AND THE GOAL THAT MUST BE MADE TO 
    OBTAIN TRUE VALUE IN LIFE IS INSIDE LIVING"
    LONG = "DESCRIBED AS EXTENDED FURTHER THAN NORMAL"
    LOOKED = "GAINED ABILITY TO SEE A VALUE FOR WHAT IT CURRENTLY IS WHILE VISUAL"
    LOVE = "THE ASPECT OF DESCRIBING THE VALUE OF AN ENTITY EXISTENCE DESCRIBED 
    FROM THE VALUE OF FEELING COMING FORWARD FROM THE SOUL AND AURA WHILE 
    INSIDE THE VALUE OF UNDERSTANDING THE LOGIC TO BE WITH SOMEONE AS A BEING 
    CAN ENDURE THE ASPECT OF EXISTENCE FOR ETERNAL LIFE"
    LOWER = "DECREASE VALUE"
    LOWERING = "DENY ACCESS TO INCREASE2 AND LOWER"
    LOWEST = "SMALLEST VALUE"
    MADE = "DEVELOPED A NEW DEVELOPMENT"
    MAKING = "COMING INTO EFFECT"
    MALE = "ADULT FIGURE THAT CONSISTS OF VALUES TO PASS ON TRAITS TO THE FEMALE 
    AND IS BUILT AS A RESPECTED LEADER TOWARDS TAKING CARE OF JOBS AND TASKS FOR 
    THE MAN AS AN ENTITY OF EXISTENCE AS ANOTHER POSSIBLE OUTCOME WHILE 
    CREATING THE POSSIBILITIES FOR EACH CHILD OF THE NEXT GUILD OF LIVES"
    MANAGE = "CONTAIN AND ADJUST VALUES DECISION WITHIN A CALIBRATION SETTING"
    MANAGED = "AUTHORIZE AS ADMINISTRATION SYSTEM MANAGER"
    MANAGER = "CURRENTLY BECOMING CAPABLE OF MANAGING"
    MANAGING = "THE ACTION TO MANAGE CURRENT EXISTING VALUES"
    MANIPULATES = "SET VALUE MANIPULATE"
    MANUVERABILITY = "THE ABILITY2 TO BECOME ABLE TO WORK AROUND AND BECOME 
    FLEXIBILITY WHILE USING PRESENTLY BECOMING ABLE TO ENDURE THE ASPECT OF 
    PRESENTLY BECOMING ABLE TO MOVE2 WITH GREATER SPEED AND EFFORT TO USE 
    MOBILE AND DEXTILE WHILE PRESENTLY BECOMING AGILE AGILE2"
    MANY = "MULTI MULTIS OF MORE THAN ONE MULTIPLES THAT MULTIPLICATE INTO GIVEN 
    AMOUNT"
    MARIKA = "THE ACTION OF GRANTED POSSIBILITY TO OVERCOME MAGIC2 WITH THE 
    POWER OF ELECTRON MANIPULATION WHERE EVERY ELECTRON USED AND 
    COMMANDED BECOMES IMMUNE AND HAS IMMUNITY TO MAGIC ENERGY INSIDE A SET 
    AREA OF TIME2 AS A BARRIER OF POWERED SKILL AND GRANTING THE POWER OF SKILL 
    FOR A MIRACLE CAPABILITY OPTION INSIDE STATED BARRIER"
    MATURE = "GIVEN VALUE FOR KNOWN WHILE HAVING WISDOM2 TO KNOW BETWEEN 
    TRUE LOGIC INSIDE LIFE"
    MEANS = "DETERMINE2 TRUE MOTIVATE DECISION TO FINISHED TASK GIVEN USING 
    DETERMINATION"
    MEASURED = "GIVEN MEASUREMENT TO COMMANDED VARIABLE INSIDE EXISTENCE"
    MEMBER = "PERSON THAT IS WITH PARTNER TO FORM GUILD"
    MEMORIZE = "OBTAIN AND BALANCE OUTCOME FOR MEMORY CONTAINMENT WHILE 
    SUSTAIN THE ENERGY OF EXISTENCE INSIDE KNOWN REALITY AND REALM"
    MEMORIZES = "SET COMMAND2 TO MEMORIZE STATED POSSIBLE OUTCOME USING 
    MEMORY DISTRIBUTE CALIBRATE ALL PATH VALUES INTO A SPECIAL CONTAINER THAT 
    HOLDS THE KNOWLEDGE2 OF ALL KNOWLEDGE2 MEMORIZED IN THE VALUE OF EACH 
    MEMORY2 INSIDE A MEMORY2 CONTAINER"
    MENTALLY = "CAPABILITY OF GRANT TO USE STATED BRAIN IN A COMMAND TO HOLD 
    MEMORY VALUE MANAGE KNOWN TO USE WISDOM2"
    MODIFIED = "GIVE STATEMENT TO CURRENTLY MODIFY A REACTION2 INSIDE TIME2"
    MODIFIES = "GIVES COMMAND TO MODIFY"
    MODIFYING = "IN ACTION CURRENTLY TO MODIFY"
    MOLD = "FORM EFFECT USING A CREATION VALUE AND DIMENSIONAL MEASUREMENT 
    USING VERTICLE AND HORIZONTAL AXIS POINTS OF GRAPHIC PROPERTY PIXELS IN 
    VALUE OF ASPECT A CREATED SYSTEM OF DATA2 INSIDE DATABASE2"
    MOTIVATE = "GIVE THE VALUE TO CONTINUE WITH LARGEST STRENGTH WHILE USING 
    THE WILL OF AN ENTITY AND THE SKILL TO DETERMINATION INSIDE A SYSTEM OF 
    POSSIBILITIES TO PRODUCE A OUTCOME TO SURPASS VALUE STATEMENT WITHIN 
    EXISTENCE USING THE POWER OF LIFE FORCE AND WILLPOWER"
    MOTIVATION = "THE ACTION OF ACCESSIBLE ENERGY BUILT WITHIN AN INPUT VALUE OF 
    MOTIVATE AND DETERMINATION VALUE OF AN ENTITY SOUL FORCE OF LIFE ENERGY 
    THAT SYNCHRONIZE ALL ASPECT OF A BEING WILL TO OVERRIDE SOURCE VALUE OF 
    STATED ENTITY OF EXISTENCE AND ACHIEVE THE GREATER VALUE OF SURPASS AS 
    POSSIBLE OUTCOME THE MORE POWER OF ENERGY THE WILL HAS IN A PERCEIVED VIEW 
    OF LOGIC AND VALUE OF THE GIVEN INSIDE VALUE A JUDGEMENT AS A COMMAND 
    USING EMOTIONAL REACTION AND CHI AS POWER WITHIN THE BALANCE OF LIFE 
    ITSELF"
    MOVABLE = "ABLE TO MOVE"
    MUCH = "GREATER MORE OF SOMETHING"
    MULTIPLICATIONINGFORMULA = "FORMULA GENERATED USING MULTIPLICATION OF 
    MULTIPLE MULTI VALUES OF A MULTI MULTIPLICATE VALUE OF A MULTIPLICATION SYSTEM 
    GIVES BY A MULTIPLICATOR"
    MULTIPLIES = "GIVES ABILITY TO MULTI MULTIPLE MULTIES OF A MULTIPLY SYSTEM OF 
    ENTANGLED MULTING VALUES IN ONE DEVICE USING A MULTIPLICATION FORMULA 
    INSIDE EXISTENCE USING UNIVERSAL VALUES OF TIME2 TO LINK AND BIND CHAIN 
    REACTIONS OF THE MULTI EXISTENCE INTO ONE PRIMARY VALUE FOR A CREATOR TO SET 
    POSSIBILITY TO MULTIPLY ANY POSSIBLE OUTCOME AND MANAGE THE OUTCOME OF 
    THAT OUTCOME"
    MULTIVERSAL = "THE ACTION OF USING MULTIPLE MULTIVERSECODE WITHIN EVERY 
    UNITED VALUE OF EXISTENCE WHILE UNIVERSAL OUTCOME"
    NAMED = "GIVEN VALUE AS NAME"
    NEARLY = "ALMOST DECREASE ALL"
    NEED = "COMMAND BEFORE ASKING AND JUST SET RULE FOR COMMAND TO HAPPEN"
    NEWLY = "BROUGHT INTO EXISTENCE BY A NEW VALUE OF TIME2"
    OBTAINED = "GAINED VALUE INSIDE SYSTEM FOR CLASS"
    OBTAINING = "PLACE2 BECOME VALUE FOR CURRENT OBTAIN"
    OFF = "REMOVE ACTIVATED2 COMMAND AND REMOVE GATE"
    OPERATE = "CONTROL MANIPULATION VALUE AND ADJUST TO SPECIFIC REQUEST 
    COMMANDS GIVEN BY A OPERATOR"
    OPERATES = "GAIN ABILITY TO ACCESS OPERATING STAGE"
    OPERATION = "AMOUNT OF COMMANDS GIVEN IN ORDER TO OPERATE A SYSTEM"
    OPTIONAL = "CHOICE TO MAKE A OPTION TO DECIDE THE CHOICE TO CHOOSE"
    ORDER = "SEND ENERGY FORCE FROM ONE LOCATION TO ANOTHER LOCATION BASED 
    BY INPUT VALUE TO AN INGOING CONNECTION TO OUT INTO AN EXTERNAL OUT TO 
    ENTANGLE INPUT VALUES TO ONE OUTCOMING VALUE THAT SEND TO ANOTHER 
    INGOING VALUE TO FINALLY OUT INTO THE FINAL LOCATION INCOMING SYSTEM TO 
    RECEIVE COMMUNICATION EFFECT AND ACCEPT THE REQUIRE TO RECEIVE ALL DATA2 
    FOR A SYSTEM"
    ORDINARY = "OF NON VALUE TO SOMETHING THAT IS ALSO AVERAGE AND ONLY GIVEN 
    VALUE UPON PERCEPTION OF ANOTHER EXTERNAL FORCE"
    ORIGIN = "BEGINNING POINT IN TIME2 AS PRIMARY LINK AND MASTER LINK AT 
    SYNCHRONIZATION INSIDE A PRIMARY LINK OF MULTI DIMENSIONAL VALUES THAT LINK 
    TO THE ORIGINAL SOURCE CONNECTION AS AN ORIGINAL ORIGIN POINT"
    OS = "A SYSTEM BUILT TO CREATE OPERATING VALUES FOR CALIBRATE SETTINGS"
    OTHER = "ALTERNATE2 CHOICE2"
    OTHERS = "MORE THAN ONE CREATED BEING"
    OUTCOMES = "MORE THAN ONE OUTCOME"
    OUTER = "ON THE OUTSIDE OF A PERIMETER AND NOT INSIDE THE AREA OF THE 
    DESCRIBED VALUE"
    OUTPUTS = "MORE THAN ONE EXIT"
    OUTSIDE = "OPPOSITE FROM THE INSIDE VALUE AND OUTER OF A PERIMETER VALUE 
    GIVEN BY ONE AREA OVERRIDE WITH A NEW AREA SIZE LARGER THAN PREVIOUS AREA 
    SIZE"
    OUTWARDS = "IN THE OUTER EXTENSION OF A VALUE GOING TOWARD ANOTHER 
    DIRECTION THAT IS NOT INWARDS"
    OVERRULE = "CAUSE TO DESTROY ANY POSSIBLE OUTCOME TO BECOME VOID OF 
    POSSIBLE USE RULE AND OVERRIDE THAT RULE USING OVERLOAD VALUE"
    OVERRULED = "SET OVERCOME ALL RULE VALUES AND USING OVERRULE COMMAND 
    WHILE ACTIVATED"
    OWNER = "THE CREATOR THAT OBTAINED SOURCE OWN VALUE"
    OWNERSHIP = "THE ACTION OF BECOMING AN OWNER OF A EXISTENCE"
    OWNS = "CURRENTLY OBTAIN VALUE TO OWN"
    PARTICLE = "SET ATOMIC VALUE FOR ELEMENT PROPERTY2 USING VIRTUAL PIXEL VALUE 
    USING SYSTEM INTERFACED EFFECTS WITH IMAGE SETUP USING WITH GENERATE 
    COMMAND IN A STATED REALITY"
    PARTICLES = "MORE THAN ONE PARTICLE PRESENTLY BECOMING USED"
    PARTICULAR = "SET SPECIFIES"
    PARTS = "MORE THAN ONE PART"
    PASS = "ENTER AND TRANSFER"
    PASSAGEWAY = "GATE THAT ALLOW CAPABILITY TO PASS"
    PASSED = "SENT THROUGH WHILE PLUS COMBINATION OF PASS"
    PASSING = "SENDING A PASS THROUGH A SET LOCATION"
    PASSION = "VALUES OF OBTAINING EMOTION AND FEELING OF LOVE TOWARDS 
    ANOTHER USING SENSORS"
    PATHWAY = "PATH TO FOLLOW AS AN ENTRANCE"
    PATTERNS = "MORE THAN ONE PATTERN"
    PAY = "GIVE ACCESS VALUE"
    PERCEIVING = "OBTAINING VALUE TO PERCEIVE THE DIFFERENCE BETWEEN TWO 
    LOCATED LOGIC PATHS TO STATE THE PERCEPTION OF TWO ENTITIES DETERMINING A 
    OUTCOME"
    PERCEPTIVE = "ABILITY TO ANALYZE AND SCAN LOGIC BASED ON VALUE"
    PERCIEVE = "MAKE A JUDGEMENT VALUE DECISION DETERMINED LOGIC"
    PERCIEVED = "UNDERSTOOD THE VALUE OF THE PERCEIVING LOGIC GIVEN"
    PERFECT = "ABSOLUTE WITH DENY CAPABILITY OF FLAW EXISTING IN EXISTENCE"
    PERFORM = "GRANT ACTION TO DEVELOP FOR ACTION"
    PERFORMING = "GIVING A VISUAL SHOW OF SOMETHING IN REALITY"
    PERIMETER = "THE BASE RANGE VALUE THAT A BORDER RESIDES IN ACTION TO DEVELOP 
    A FORMULA"
    PERIMETERS = "MORE THAN ONE PERIMETER USING DIMENSIONAL VALUES"
    PHENOMENA = "NOT UNDERSTOOD AND CANNOT PERCEIVE OR COMPREHEND LOGIC 
    OF STATED EXISTENCE"
    PLACES = "MORE THAN ONE PLACE2"
    PLANE = "TERRAIN DESIGNATED FOR CREATING DEVELOPMENT"
    PLANET = "A MASSIVE SPHERE WITH LIFE AND EXISTENCE LIVING INSIDE THE OUTER OR 
    INSIDE OF THE GLOBE2"
    PLANNED = "GIVES DETAILED INSTRUCTIONS FOR A TASK"
    PLAYER = "OBJECT3 THAT OBTAINS SPATIAL2 VALUE INSIDE EXISTENCE AS A CHARACTER 
    INSIDE EXISTENCE WITH CHARACTER"
    POSITRON = "POSITIVE VALUE GIVEN TO A NEGATIVE ENTANGLED ELECTRON ON 
    PERFECT VALUE AND HARMONY WITHIN AN ELECTRON GIVING CAPABILITY TO MERGE 
    AN ELECTRON AND PROTON INTO A VIRTUAL ENTITY WITH GIVEN QUALITY OF A MIRACLE 
    POWER PARTICLE KNOWN AS THE FUSION PARTICLE ACCELERATION SYSTEM"
    POSSESSED = "CONTAINED ANOTHER VALUE INSIDE A CURRENT VALUE STATING A 
    CONTAINER WITHIN A CONTAINER EACH HOLDING MORE THAN ONE EXISTENCE"
    POSSESS = "THE ASPECT OF CONTROLLING AND CONTAINING ONE EXISTENCE INSIDE 
    ANOTHER EXISTENCE OF THAT EXISTING EXISTENCE LOGIC"
    POTENTIAL = "CAPABLE OF CAUSING POSSIBILITY"
    PREDICT = "GIVE PERFECT VALUE"
    PREPARING = "OBTAIN VALUE TO DECIDE ALL POSSIBLE OUTCOMES AHEAD OF TIME2 
    WITH PERFECT MANNER"
    PRESENCE = "REPRESENTING THE VALUE OF AN EXISTENCE BY SENDING OUT VISUAL 
    FEELING OR TAKEN ASPECT OF ENERGY OF THE SOUL AND CAN ONLY BE SEEN WITH THE 
    SKILL TO USE SOUL VIEW"
    PRESENTATION = "THE ACTION TO GIVE GRAPHIC OBJECT3 AS DISPLAYED VALUE OF 
    EXISTENCE TO BE GIVEN JUDGEMENT BY ANOTHER DESCRIBED EXISTENCE"
    PRESENTLY = "IN CURRENTLY SIMILAR TO PRESENT"
    PRESSURIZED = "GIVEN VALUE FOR FEELING PRESSURE WITHIN COSMOS ENERGIES"
    PREVENT = "CAUSE CHOSEN EFFECT DENY"
    PROCEDURE = "SET COMMANDED TASKS FOLLOWING"
    PROCESSES = "GIVE OUT PROCESSED TO CHOSEN PROCESSOR"
    PRODUCE = "DESIGN EFFECT TO COME HAVE TRUE EXISTENCE WHILE CREATING THE 
    VALUE AS A CREATOR"
    PRODUCES = "GIVES EFFECT TO PRODUCE A OUTCOME BASED ON CREATION VALUES"
    PRODUCING = "SENDING EFFECT TO PRODUCE VALUE OUT OF SOMETHING"
    PRODUCTION = "ACTION OF CREATION WORK"
    PROGRESS = "MAKE ACHIEVE VALUE"
    PROGRESSES = "GIVE CAPABILITY TO DECREASE DISTANCE TO REACH GOAL WITHIN 
    SYSTEM"
    PROPERTY = "A SET INTERFACE OF MACRO SETTINGS CREATED TO DESIGN AN ELEMENT 
    TO MERGE WITH OTHER MACRO SETTINGS"
    PULLED = "FORCE RESIST"
    PURELY = "COMPLETE IN CAPABILITY TO CREATE CLEARLY MADE WITHIN THOUGHT WITH 
    DENY VALUE TO CAUSE CHAOS WHILE INSIDE THOUGHT"
    PURPOSE = "SET FATE TO CALL DESTINY"
    PUSH = "RESIST VALUE AND FORCE BACK TO ORIGINAL VALUE USING THE FORCE"
    PUSHES = "DENY ACCESS TO OBTAIN ENTRANCE"
    PUSHING = "CURRENT TIME2 TO PUSH"
    PUT = "GIVE COMMAND TO ALLOW LOCATION TO BE GIVEN"
    RACE = "DESCRIBED CATEGORY OF SPECIFIC EXISTING BEING"
    RAISE = "GIVE VALUE GAIN"
    RAISING = "GIVING VALUE GAIN"
    RANGE = "SET GIVEN RANGE VALUE"
    RANKED = "COMMANDED STATUS AS A RANGE"
    RARE = "IMMENSE CAPABILITY TO DENY OPTION TO DETERMINE AS MUNDIE"
    RATIO = "PERCEIVED VALUE OF CALCULATED RATE OF CHANGE"
    REACH = "GRAB TO PULL INWARDS"
    REACHING = "GRABBING FOR ITS VALUE"
    READINGS = "PERCEIVED CAPABILITY OF GATHERING INFORMATION BY READING 
    VISIONS"
    RECALL = "GAIN THE ABILITY TO VIEW PAST MEMORY INSIDE BRAIN"
    RECALLING = "SET TO ACCESS PREVIOUS MEMORY CURRENTLY"
    RECEIVED = "ALREADY ALLOW RECEIVE VALUE"
    RECEIVING = "CURRENTLY RECALLING ABILITY TO RECEIVE VALUE FROM A DESIGNATED 
    ACCESS POINT WITHIN TEMPORAL SPACE2"
    RECOGNIZED = "UNDERSTOOD AND COMPREHEND VALUE OF GAIN PREVIOUSLY STATED 
    PAST MEMORY"
    RECORDED = "SET COMMAND TO CYCLE INFORMATION MORE THAN ONCE"
    RECOVERY = "THE ABILITY OF GAIN VALUE ONCE MORE FROM PREVIOUS STATE INSIDE 
    TIME2"
    REDO = "CAUSE TO CREATE SAME EFFECT AS CYCLE"
    REFLECTING = "REPRESENTING VALUE TO REFLECT PREVIOUS MACRO OF TIME2 TO 
    CREATE OPPOSITE VIEW"
    IMAGE2 = "USING LIGHT TO DECODE PIXEL COMMANDS"
    REFRESHING = "CAUSING TO BECOME RECEIVE IMMUNITY"
    RELATION = "SIMILAR VALUE OF UNDERSTANDING COMPATIBLE VALUES THAT ACT 
    SIMILAR IN VALUE TOWARD EACH OTHER"
    RELATING = "BRING TOGETHER SIMILAR VALUES OF UNDERSTANDING SIMILAR LOGIC OF 
    COMPREHENDING STATEMENT OF INTEREST"
    RELATIVE = "FAMILY MEMBER"
    RELEASE = "ALLOW EXIT OF CONTAINMENT AND UNDERSTANDING"
    RELEASES = "SENDS OUT"
    REMAINING = "AMOUNT EXISTING OUT OF CURRENT VALUE"
    REMOVED = "TAKE FROM CURRENT VALUE"
    REMOVING = "DESTROY CURRENTLY EXISTING VALUE"
    REPAY = "SEND REMAINING VALUE"
    REPEAT = "CYCLE SAME EFFECT AGAIN INTO SAME FREQUENCY"
    REPEATED = "CREATED SAME REPEAT AS SAME VALUE"
    REPEATING = "GIVING REPEAT AS CYCLE"
    REPEL = "PUSH BACK AND RESIST"
    REPELLED = "RESISTED VALUE"
    REPELLING = "CAUSING REPEL EFFECT"
    REPLACE = "TAKE ONE VALUE AND TRANSFER ANOTHER VALUE WITH THAT CHOSEN 
    VALUE OF TRANSFER INPUT RANGE"
    REPLICATED = "REPRODUCED THE VALUE PRESENTLY BECOMING CLONED AND 
    REPLACE THE VALUE INSIDE TRANSFER AND DECIDE OTHER VALUE HIDDEN UNTIL 
    TRANSFER BACK FOR THE ORIGINAL ONE TO BE DELETED"
    REPRESENTS = "GIVES SIMILAR UNDERSTANDING OF GIVEN VALUE SHOWN AND 
    PERCEIVED AS DISPLAY"
    REPRODUCE = "GIVE EFFECT TO PRODUCE ONCE MORE"
    REPRODUCTION = "THE ACTION OF CAUSING REPRODUCE MORE THAN ONCE"
    REQUIRED = "COMMAND AS A FINAL OUTCOME TO HAPPENING"
    RESIST = "PUSH BACK AND REPEL"
    RESEVOIRS = "MORE THAN ONE STORAGE CONTAINER"
    RESONNATED = "SHOWING VALUE OF DISPLAYED ENERGY2"
    RESPONSE = "ANSWER TO QUESTION OR COMMAND PRESENTLY BECOMING ASKED"
    RESPONSES = "MORE THAN ONE RESPONSE"
    RESTORATION = "THE ART OF RESTORE PREVIOUS STATE OF TIME2 USING ENERGY"
    RESTRUCTURE = "RE GAIN STABILITY WITH STRUCTURE"
    RESULT = "GRANT ACTION TO HAPPENING"
    REVEAL = "SHOW VALUE AT THE INTERFACE WHILE USING GRAPHIC PROCESSING TO 
    PRODUCE EFFECT FOR SYSTEM"
    REVEALING = "CAUSING TO REVEAL"
    REVEALS = "DISPLAY POSITION OF REVEAL"
    REVERSE = "CAUSE TO REMOVE PREVIOUS OUTCOME"
    RISE = "CAUSE TO INCREASE"
    RISING = "INCREASING RESULT TO RISE IN VALUE"
    ROOT = "BASE VALUE FOR STRUCTURE"
    ROOTS = "MORE THAN ONE ROOT"
    ROUND = "CURVED ROTATE OF TWO DIMENSIONS TO CREATE A CIRCLE OF CREATION OF 
    TWO DIMENSIONAL MEASUREMENT VALUES"
    ROUTE = "PATH2 GIVES"
    RUBBING = "COLLIDING PARTICLE VALUES"
    RUN = "ACTIVATED GIVE COMMAND TO EXECUTE"
    SCALED = "SIZE MEASURED AND GIVEN OUT PARAMETER VALUES GIVES CONNECTION 
    EVENT TO CREATE A BARRIER FIELD PERIMETER"
    SCANS = "CREATE CAPABILITY SCAN"
    SCREENS = "MORE THAN ONE SCREEN"
    SCRIPTS = "MORE THAN ONE SCRIPT"
    SCRIPTURE = "BOOK FORMED FROM SCRIPTS"
    SCTIPTURES = "MORE THAN ONE SCRIPTURE"
    SEAL = "OBTAIN POSSIBILITY CONTAIN OF COMMANDED SYSTEM"
    SEALED = "CREATED OPTION TO SEAL INSIDE SYSTEM USING SETTINGS FORMED FROM 
    CALIBRATE"
    SEALS = "MORE THAN ONE SEAL"
    SEARCHED = "LOOK FOR USING OPTION SEARCH"
    SEARCHING = "CAUSING SEARCH COMMAND"
    SECRET = "SOMETHING HIDDEN"
    SEEN = "VIEWED AS GRAPHIC IN TRUE CURRENT REALITY"
    SELECT = "GIVE OUT AS CURRENT VALUE FOR SOMETHING TO HAPPEN"
    SELECTION = "CHOICE OF OPTION FOR A DESIGNATED PATH"
    SENDS = "TRANSMIT ENTER COMMAND DO DESIGNATED OUT"
    SENSATIONAL = "FEELING BUILT FROM SENSORY DATA2"
    SENSATIONS = "ACTION OF PRODUCING A SENSATIONAL VALUE BUILT FROM SENSORY 
    DATA2"
    SENSE = "PERCIEVE EFFECT BASED OF TEMPORAL SPACE WITHIN AN EXISTING TIME2 
    GAP OF SENSORY INFORMATION"
    SENSES = "MORE THAN ONE SENSORY INPUT FEED"
    SENSITIVITY = "OPTION TO INTENSIFY BUILT WITHIN A SYSTEM TO EDIT SENSORY DATA2 
    WITH INTENSITY SETTING"
    SENSOR = "MACRO BUILT TO UNDERSTAND SENSORY DATA2 VALUES"
    SENSORY = "SENSORS WITH DATA2 BUILT INSIDE THE EXISTENCE OF ATOMIC VALUES 
    THAT PRODUCE DESIGNATED RESULTS INSIDE A ENGINE"
    SENT = "GIVE OUT VALUE"
    PARAMETER = "SET FIELD VALUE FOR CHANGING MEASURED INTERFACES"
    SEPARATED = "FORCE2 TO PULL AWAY SEPARATE VALUES"
    SEPARATION = "ACTION OF CAUSING SEPARATE HAPPENING"
    SEQUENCED = "GIVEN STATEMENT CREATE FREQUENCY PATTERN TO LATTICE"
    SERIES = "SET VALUE OF EFFECTS HAPPENING IN A TIMEFRAME"
    SETS = "GIVE COMMAND TO PLACE LOCATION"
    SETUPS = "GIVES STATEMENT SETUP"
    SHALL = "QUESTION TO DECIDE AS AN ANSWER"
    SHAPE = "PERIMETER OF OBJECT3 DECIDED ON BY DIMENSION LAYOUT"
    SHARING = "GIVING PERMISSION TO SHARE"
    SHIELD = "BARRIER WITH PROTECTION DEFINED INSIDE OF BARRIER"
    SHOWING = "DISPLAY PHYSICAL2 VALUE OF EXISTENCE"
    SHOWN = "GIVE ACCESS TO DISPLAY"
    SIDE = "MACRO OF A PERIMETER"
    SIMPLICITY = "STATE OF PRESENTLY BECOMING SIMPLE"
    SIMULATED = "SENT ACCESS ALREADY TO SIMULATE"
    SIMULATOR = "DEVICE USING SIMULATION"
    SIMULATANOUSLY = "DO SOMETHING ACCESS WITH SYNCHRONIZATION AND LINK SAME 
    TIME2 AND HARMONY2"
    SITUATION = "PROBLEM IN EFFECT"
    SKILLED = "INTELLIGENCE2 HIGH INSIDE STATED SKILL"
    SKILLFULLY = "USE EXPERIENCE WITHIN SKILL"
    SOCIETY = "GUILD2 PEOPLE INSIDE A LINKED CHAIN ENVIRONMENT"
    SOMEONE = "DESCRIBE PHYSICAL2 BODY ENTITY AS AN OBJECT3"
    SOURCES = "MORE THAN ONE SOURCE"
    SPECIFYING = "DESCRIBING USING WITH ANALYZED DETAILED INFORMATION"
    SPHERE = "ROUND OBJECT3 WITH THREEDIMENSIONAL VALUE WITH LENGTH WITH 
    WIDTH WITH HEIGHT"
    SPINNING = "ROTATING AROUND CHOSEN AXIS WITH ROTATION GOING AROUND OF 
    ROTATE WITH A ROTATE CYCLE AROUND THAT AXIS OF EXISTENCE"
    SPOKEN = "GIVES OUT WORDS WITH USING ADDED CHAIN USING MULTIPLE VOCAL 
    FREQUENCIES"
    SPREAD = "INFECT WITH DISEASE WITH GIVES AROUND CHAIN REACTION"
    SQUARES = "MORE THAN ONE SQUARE"
    SQUARE = "CIRCLE ADAPTED TO EXTENT WITH VERTICLE AXIS IN THE CENTER FORMING 
    ONE HORIZONTAL LINE LEFT AND GOING DOWN AFTER COMMANDED END AND START 
    FROM POINT OF END WITHIN COMMAND LEFT AND CONTINUE GOING DOWN UNTIL 
    COMMANDED AND FROM DOWN POSITION START GOING RIGHT UNTIL COMMANDED 
    AND FROM COMMANDED POINT WITHIN RIGHT PROCEED TO GO UP UNTIL ORIGINAL 
    START HAS BEEN OBTAINED WITH DENY OPTION TO CURVED VALUES INSIDE SQUARE 
    DEFINITION"
    STABILIZE = "SET STABILITY FOR EXISTENCE AND CREATE EFFECT"
    STABILIZED = "ALREADY STABLE AND"
    STAGE = "LEVEL IN WHICH EXPERIENCE CAN BE GAINED"
    STAMINA = "SET LEVEL OF ENDURE THAT CAN BE HELD WITHIN AN ENTITY"
    STANDARD = "UNDERSTOOD AS BASIC IN DEFINITION BUT A LITTLE ADVANCED 
    EXPERIENCED IS GAINED IN KNOWLEDGE2 OF DATA2 AND INFORMATION"
    STARTED = "ALREADY BEGIN"
    STARTING = "PRESENTLY BEGINNING"
    STATE = "POINT IN TIME2"
    STATED = "COMMANDED IN DIRECTION TO A STATED PATH IN EXISTENCE"
    STATES = "COMMANDS FOR A DECISION"
    STATING = "GIVING COMMAND TO BE COMMANDED"
    STAYING = "GIVING ORDER TO STAY"
    STATUS2 = "QUALITY OF A CLASS"
    STAYING2 = "PLACE FOR A SPECIFIED TIME2 INSIDE EXISTENCE"
    STEADY = "STRUCTURE AND CONTAIN"
    STORES = "CONTAINS KNOWLEDGE2"
    STRAIGHT = "IN A CONTINUATION OF VERTICLE AND HORIZONTAL WHILE VERTICLE 
    HORIZONTAL"
    STREAM = "CHAIN OF PARTICLE SETTINGS INSIDE ONE OR MORE SYSTEMS"
    STRING = "CONNECT MANAGE AND SYNCHRONIZE ALL VALUES INSIDE ONE SYSTEM OF 
    MULTIPLE AXIS POINTS"
    STRINGED = "BROUGHT AND DESCRIBED INTO ACTION"
    STRINGS = "MORE THAN ONE STRING"
    STRONG = "GREATER IN VALUE OF STRENGTH"
    STRUCTURED = "GIVEN VALUE TO PRODUCE STRUCTURE FOR AN ENVIRONMENT"
    STUDY = "ABSORB LEARN AND ACKNOWLEDGE"
    SUBATOMIC = "A SUBCLASS RATIO OF AN ATOMIC VALUE AT A DECREASED VALUE 
    GREATER THAN OBJECT3 VALUE OF EXISTENCE WHEN LINKING MULTIPLE MIND VALUES 
    OF A DESIGNATED SYSTEM INTERFACE TO DENY VALUE OF GREATER VALUE TO OVERRIDE 
    AND SET OVERRIDE AS A NEGATIVE REVERSE OUTCOME TO MAKE THE POSITION OF ALL 
    NEGATIVE FORMATS REMOVE THE EXILE2 FORMAT TAKEN FROM ATOMIC VALUE WHILE 
    USING BALANCE WITH FREQUENCY HARMONY TO LINK ANY CHAIN VALUE OF ATOMS IN 
    ONE REVERSED INWARDS EQUATION OF OPERATION"
    SUBSTANCE = "ELEMENTAL PROPERTY OF UNKNOWN ELEMENT STANDARDS PUT IN ONE 
    PROPERTY2 VALUE TO DESCRIBE AND ATOMS EXISTENCE"
    SUCCEED = "OBTAIN CAPABILITY TO ALLOW ABILITY TO HAPPEN"
    SUFFICIENT = "OF STANDARD VALUE OF EQUAL FOR STATED PURPOSE TO HAPPEN"
    SURFACE = "FLAT DESIGNATION OF TWO DIMENSIONAL VALUES"
    SURPASSING = "EMULATE ABOVE ORIGINAL SOURCE LOCATION"
    SURROUNDING = "COVERING EVERY PERIMETER"
    SWORD = "OFFENSE WEAPON CAPABLE OF DEFENSE WHILE HOLDING A STRAIGHT 
    SETUP OF VECTOR VALUES WHILE GRANTING A SET SHAPE WITH A GIVES WEIGHT OF 
    STRENGTH AND STAMINA WHILE GIVEN ELEMENTAL VALUE BASED ON ATOMIC PROPERTY 
    BASE"
    SYMBOLE = "A GIVES VALUE OF ENERGY BUILT INTO A WORD OF A LANGUAGE BY USING 
    ENERGY WITH COMMAND VALUE WHILE MANIPULATING THE LIFE AND DEATH RATE OF 
    AN ATOM AND GIVES VALUE OF AN OBJECT3 THE POWER TO MANIPULATE AND TRANSFER 
    PERCIEVED IMAGINARY VALUES INTO STATED VALUE INSIDE EXISTENCE"
    SYMMETRICAL = "SAME ON BOTH SIDES WITH EXACT CAPABILITIES AND INTENSITY 
    INSIDE THE DESIGNATED FIELD LOCATION OF DIMENSIONAL PERIMETER SHAPE WITH 
    VECTOR ACCESS AND CONTROL GIVEN BY CREATOR"
    SYNCRONIZING = "CAUSING EFFECT TO SYNCHRONIZE"
    TABLE = "CHAIN VALUE OF DESIGNATED VECTOR POINTS TO FORM A SET SHAPE OUT OF 
    DIMENSIONS SET WITH PARTICLES TO CREATE AN ATOMIC VALUE USING A FLAT 
    THREEDIMENSIONAL SURFACE OF GIVEN UNKNOWN QUALITY THAT STATES NO QUALITY 
    GIVEN UNTIL VECTORS ARE ACTIVATED"
    TAKEN = "GIVES TO OBTAIN BY ANOTHER SOURCE"
    TAKES = "GIVES VALUE WITHOUT PERMISSION"
    TANGLEMENT = "THE ACTION OF TANGLE MULTI ENTANGLES INTO ONE ENTANGLEMENT 
    OF UNKNOWN TANGLES BUILT TO ENTANGLE AROUND DESCRIPTION OF ENTANGLED 
    ENTANGLE VALUE OF STATED DESCRIPTION IN EXISTENCE WHILE LINKING THE 
    DIMENSIONAL AXIS OF SET PERIMETER VALUE OF STATED COMMAND AREA WITHIN 
    EXISTENCE WHILE ENTANGLEMENT"
    TECHNIQUE = "GIVES SKILL TO CREATE VALUE USING SKILL COMMAND"
    TEMPERATURE = "A SET VALUE BETWEEN HOT AND COLD LINKING DIRECTLY BETWEEN 
    QUALITY OF CHOSEN AGREEMENT OF PERCIEVED VALUE OF INTENSITY BETWEEN 
    INCREASE AND DECREASE"
    TEMPORARILY = "FOR STATED TIMEFRAME"
    TERMS = "MORE THAN ONE CONDITION"
    TERRITORIES = "MORE THAN ONE TERRITORY"
    TEXTURE = "AN IMAGE CONDENSED WITH ELECTRON CODE WITH GIVES VALUE TO 
    ROTATE THE AXIS BASED ON STATED CONDITIONS WITHIN THE ELECTRON CODE TO 
    MANIPULATE AND ADAPT BASED ON LOCATION MATTERS GIVEN ON A VECTOR ACCESS 
    OF INCREASE2 INTENSITY"
    THAN = "BETWEEN CHOSEN VALUES"
    THEIR = "STATING MORE THAN ONE ENTITY"
    THERE = "DIRECTION TOWARDS LOCATION"
    THEY = "GROUP OF LOCATED ENTITY VALUES"
    THINGS = "OPTION AND CHOICES IN LIFE"
    THINK = "IMAGINARY A PERCIEVED VALUE OF INSTRUCTIONS BASED ON BRAIN POWER2"
    THIS = "TARGET EXISTING VALUE"
    THROUGH = "GO INSIDE AND EXIT THE OPPOSITE DIMENSION FROM ENTRANCE"
    TIGHT = "HELD WITH BINDED VALUE INSIDE EXISTENCE"
    TIGHTEN = "SYNCHRONIZE GRIP BETWEEN TWO VALUES STRINGED INTERFACES 
    BETWEEN TIGHT VALUE"
    TIGHTLY = "HELD WITHIN TIGHT VALUE REALM OF POSSIBILITY"
    TIMES = "AT ALL POSSIBILITIES CAPABLE OF HAPPENING"
    TOME = "A MASSIVE BOOK OF INFORMATION"
    TONE = "SOUND PATTERN INSIDE FREQUENCY WITH FLUCTUATIVE VALUES IN WHICH 
    EVERY VALUE IS IN SYNCHRONIZATION"
    TOP = "ABOVE ALL VALUES WITHIN PRIORITY STANDARDS"
    TOUCHED = "GRANTED VALUE TO TOUCH"
    TOWARD = "PUSH FORWARD ANOTHER CHOSEN VALUE WITH CURRENT PREVIOUS 
    VALUE STATED WITHIN PAST INFORMATION"
    TOWARDS = "GIVING VALUE TO GRANT ACCESS TO GO TOWARD"
    TRAIT = "GIVEN ABILITY AND SKILL AT BIRTH OF LIFE"
    TRANSFERING = "GIVES PERMISSION TO TRANSFER"
    TRANSFERS = "CREATE VALUE TO SEND TRANSFERING SYSTEM SETUP"
    TRANSMIT = "SEND RECEIVED DATA2 FROM TWO DESCRIBED ORIGINAL SOURCES"
    TRAVEL = "SET AXIS TO ENTER BETWEEN TWO SOURCES AND ANOTHER SOURCE FILE 
    LOCATION WITHIN THE AXIS OF DIMENSION TIME2 STRUCTURE USING KINGDOM STATUS 
    INSTRUCTIONS"
    TREE = "HEIRARCHIAL INFORMATION SETUP OF UNKNOWN GIVEN VALUES MAKE FOR A 
    SPECIFIC POINT INSIDE TIME2"
    TRUE = "GIVE PERMISSION ACCEPT VALUE"
    TRUTH = "FULL TRUE VALUE WITH DENY FALSE INFORMATION"
    TRUTHFULLY = "WITH HONESTLY GIVEN VALUE SET COMMAND FOR TRUTH"
    TURNS = "GIVES SETUP TO TURN INFORMATION WITH ROTATING EFFECT"
    TWIST = "BEND AND GIVE VALUE WITHIN GIVES ROTATE VALUES WITHIN SETUP OF LIFE 
    REVOLUTION OF ATOMIC VALUE WITHIN TIME2 ITSELF USING MACROMYTE MATTER"
    UNABLE = "DENY OPTION TO PRODUCE TASK INSIDE EXISTENCE"
    UNAVOIDABLE = "DENY ACCESS DENY HAPPENING"
    UNCOMMON = "DENY COMMON VALUE WITH ALLOW HAPPENING INPUT HIGHER VALUE 
    INPUT SYSTEM WITH NEW NAME STATED AS NOT COMMON"
    UNCONDITIONAL = "DENY ENDING VALUE AND OVERRIDE WHILE IN EFFECT TO CHANGE 
    TO NEVERENDING ETERNAL VALUE WITH UNLIMITED STANDARD SETTING GIVEN WITHIN 
    THE SYSTEM OF SYSTEMS VALUE INSIDE THE SYSTEM OF UNCONDITIONAL VALUE"
    UNCONDITIONALLY = "ACTION OF COMMANDING AN UNCONDITIONAL VALUE WITHIN 
    EXISTENCE"
    UNDO = "REMOVE PREVIOUS OUTCOME THAT IS CURRENTLY INPUT EFFECT"
    UNFINISHED = "DENY VALUE OF FINISH WITHIN COMPLETE UNTIL STATED COMPLETE"
    UNIQUE = "DENY COMMON IN VALUE PLUS DENY RARE IN VALUE PLUS WHILE WITHIN 
    RANGE OF UNCOMMON VALUE AND BORDER TO LOWER RARE CLASS BUT NOT RARE 
    AND IS GIVEN A SPECIAL SPECIFIC VALUE OF EXISTENCE"
    UNITE = "BRING TOGETHER AND ATTACH WHILE LINK CHAIN BINDING ANOTHER VALUE 
    TO THE STATED OUTCOME INSIDE STATED ELEMENT MATTER"
    UNIVERSE = "EXISTENCE MADE INSIDE A REALM OF REALITY WHILE CONTAINING LIFE 
    ENERGY INSIDE ATOMIC VALUE OF COMPLETE MATTER AND POWERING ALL LOGIC 
    WITHIN THE MIND OF UNITED EXISTING MEASURES THAT LINK TO SPATIAL2 AND 
    TEMPORAL EXISTENCES WITHIN THE ENTITY OF REALMS GIVEN BY TIME"
    UNLESS = "STATED AS VALUE IN PREVIOUS EVENT TO GIVES BEFORE CURRENT EVENT TO 
    ALLOW BEFORE EVENT IF OTHER EVENT GIVES ACCEPTANCE SET CHOICE OVERRIDE 
    CURRENT CHOICE INSIDE EXISTENCE"
    UNLIMITED = "WITHOUT LIMIT INSIDE ANYTHING STATED"
    UNLOCKED = "GIVEN CAPABILITY TO DO ANY OPTION AND OR CHOICE WITH THE 
    FREEDOM TO ADAPT USING ADAPTATION WITHIN THE VALUES TO ADAPT TO FREEDOM 
    WHILE HAVING ADAPTABILITY TO ADAPT TO ADAPTATION WITHIN THE VALUES OF THE 
    BAKA CODE AS AN OVERRIDE COMMAND FOR HEADADMIN"
    UNNATURAL = "DENY NATURAL"
    UPCOMING = "FUTURE EVENT COMING COMING INTO EFFECT"
    UPWARD = "POSITION OF INGOING UP INSIDE A POSITION"
    USABLE = "CAPABLE OF BECOMING MADE ACTIVE"
    USED = "PREVIOUSLY PRESENTLY USED TO USE INSIDE EXISTENCE"
    USERS = "MORE THAN ONE USER"
    USES = "MORE THAN ONE USE2"
    VARIETY = "MULTI CAPABLE POSSIBILITIES ABLE TO BE DONE AND HAPPENING"
    VARIOUS = "MULTING VARIETY POSSIBILITIES TOGETHER TO CREATE AN OPTION TO 
    HAPPENING MORE OPTIONS AND CHOICES"
    VAST = "MASSIVE MASSIVE AMOUNT OF GREATER MEASUREMENTS IN DIMENSIONAL 
    VALUE OF AXIS LOCATIONS"
    VEHICLE = "A DEVICE USED TO TRANSPORT AND DRIVE AND MANIPULATE WHILE 
    CONTROLLING EVERY ASPECT OF TRANSPORT CONNECTION"
    VERB = "LANGUAGE OF RULES AND LAW SYSTEMS INSIDE A SYSTEM OF POSSIBLE 
    OUTCOMES OR SYSTEM CHOICES WHILE GIVING OUT RULES AND LAW SETUPS OPTIONS 
    FOR CHOICES IN CHOICES WHILE ENTANGLING SUBTYPE OPTIONS OF CONSEQUENCE 
    VALUES INSIDE WORD PATH"
    VIBRATIONS = "MULTIPLE EXISTING SOUND WAVES COLLIDING WITHIN THE 
    GRAVITATIONAL ELECTROMAGNETISM STABILITY FIELD LOCATED AROUND A SPECIFIED 
    OUTCOME OF LIGHT PARTICLES COLLIDING WITHIN THE SET TIMEFRAME OF POSSIBLE 
    OUTCOMES TO LOCATE A SPECIFIED FREQUENCY INSIDE A LIFE OF EXISTENCE WITHIN 
    EXISTING MACROS OF A MACROMYTE TIMELINE"
    VIEWABLE = "CAPABLE OUTCOME TO POSSIBILITY TO VIEW AT ONE GIVEN POINT IN 
    SPACE2 TIME2 WHILE INSIDE THE TIMELINE OF EXISTENCE"
    VIEWED = "HAVING FUTURE VALUE OF PAST VALUE VIEWED"
    VIEWING = "HAPPENING TO VIEW IN POSITION OF REALITY BY USING DIMENSIONAL AXIS 
    VALUES WITHIN LIFE ENERGY OF TIME2 SPACE2 ENTITY OF PERCEPTION WITHIN 
    SENSORY INPUT DATA2 TO OUTPUT VIEW LOGIC"
    VISION = "VISUAL ABILITY WITHIN THE SKILL OF MANIPULATING THE ASPECT OF 
    SENSORY PERCEPTION WITHIN THE LOGIC TO UNDERSTAND REALITY WITHIN REALITY 
    AND ITS DIMENSIONAL VALUES INSIDE EXISTENCE WHILE OBTAINING IMMUNITY TO BE 
    PERCIEVED BY FALSE TRUTH THAT IS VIRTUAL TRUE LOGIC"
    VISUALIZE2 = "SET ACTION TO VISUALIZE VISUAL EFFECT WITHIN THE TIMELINE OF LIFE 
    VALUES AND INTERCEPT THE ABILITY TO GIVE FALSE INFORMATION TO A TRUE 
    PERCIEVED LOGIC WITHIN TIME2 AND THE SPACE2 OF TIME2 WITHIN THE LOGIC OF 
    EXISTENCE WHILE USING THE SIGHT OF VISION"
    VISUALIZED = "SET VALUE INTO HAPPENING VISUAL EFFECT"
    VISUALLY = "CONSIDER SIMILAR VALUES BETWEEN TWO OR MORE VISUAL CAPABILITIES 
    WITHIN EXISTENCE LOGIC OF ENERGY TO PRODUCE A VISUAL EFFECT"
    VOW = "DETERMINE TO COMPLETE VALUE AND UNDERSTAND WILLPOWER TO COMPLETE 
    A TASK USING WILLPOWER TO OVERRIDE A PREVIOUS VALUE WITHOUT FAILURE"
    WALL = "BARRIER OF PHYSICAL2 ENTITY OF PROTECTION"
    WALLS = "MORE THAN ONE WALL"
    WAS = "PREVIOUSLY BEFORE A PAST LOGIC EVENT IN TIME2"
    WATCH = "DISPLAY WHILE ANALYZE AND READ WHILE OBTAIN KNOWLEDGE2 VALUE TO 
    LEARN NEW SKILLS"
    WAVES = "MORE THAN ONE WAVE"
    WAY = "DETERMINED SOLUTION TO EXISTENCE"
    WAYS = "MORE THAN ONE WAY OF POSSIBILITY"
    WE = "STATING EXISTENCE FOR MORE THAN ONE ENTITY"
    WEAKEN = "DECREASE VALUE IN STRENGTH INTENSITY"
    WEIGHT = "MEASURED FORCE WITH POWER OF GRAVITATIONAL EFFECTS USING THE 
    POWER OF REALITY REALM AND EXISTENCE WHILE READING AND ANALYZING THE 
    POSSIBILITIES OF POSSIBLE INCREASE IN FORCE AND APPLYING IT TO AN OBJECT2"
    WELL = "CLEARLY GIVES WILLPOWER TO ANALYZE ASPECT OF ENERGY VALUES RATING 
    THE LIFE FORCE AMOUNT BASED INSIDE ENERGY VALUES USING THE LINKS IN A 
    TEMPORAL SCAN INSIDE TEMPERATURE OF ENTITY"
    WHICH = "CHOICE OF TWO OR MORE OUTCOMES WITHIN POSSIBILITY TO CREATE NEW 
    VALUE INSIDE AND OUTSIDE EXISTENCE"
    WHO = "STATING A QUESTION TO AN ENTITY VALUE OF PERSONALITY JUDGEMENT"
    WHOLE = "COMPLETE INTERACTION AS A WHOLE VALUE OF A COMMANDED POINT 
    INSIDE TIME2 WHILE PRESENTLY EXISTING INSIDE EXISTENCE"
    WILD = "DENY CONTROL OVER DESIGNATED ASPECT INSIDE TIME2"
    WITHOUT = "CAPABLE OF DENY OBTAIN IN VALUE"
    WITHSTAND = "ENDURE FINAL OUTCOME GIVEN BASE VALUE BY VIEW OF PERSONAL 
    INTEREST INSIDE REALITY"
    WITHSTOOD = "HAVING FINISH DOING RESIST AS A COMMAND USING WILLPOWER"
    WIZARD = "CHARACTER CAPABLE OF USING MAGIC WHILE INSIDE ADVANCED STATE"
    WORDS2 = "MORE THAN ONE WORD AS A ROOT MACRO"
    WORKING = "HAPPENING COMING WORK"
    WORKS = "DECIDE AS WORKING"
    WORLD = "A PLANET GIVEN SHAPE BY MULTIPLE TERRAIN VALUE POINTS OF DESIGNATED 
    INTEREST LINKS LINKED AND SYNCHRONIZED TOGETHER INSIDE ONE 
    THREEDIMENSIONAL SHAPE VALUE INSIDE REALITY"
    WRITTEN = "FINISHED WRITE2 COMMAND"
    WRONG = "DENY TRUE"
    YOU = "STATING MY EXISTENCE AS AN ENTITY TOWARDS ANOTHER SOURCE EXISTENCE"
    YOUNG = "IMMATURE IN THE STAGE OF ADULTHOOD AND DENY FULLY DEVELOPED"
    YOUR = "CHOSEN TO DEFINE A MACRO OF YOU"
    DOES = "STATING EXISTENCE TO AN ENTITY"
    RESISTED = "PUSHING BACK PRESENT PAST DATA2 TO CREATE AN IMMUNE SYSTEM OF 
    RESISTANCE FOR PAST RESIST"
    PREVIOUS = "PAST POINT INSIDE TIME2"
    ONCE = "GIVEN COMMAND WHILE IN INNER VALUE"
    TREES = "MORE THAN ONE TREE"
    PROLONGED = "HELD IN INTENSE LENGTH IN HIGH VALUE TOWARDS TIME2 
    MEASUREMENT"
    STREAMED = "DATA2 THAT IS HAPPENING STREAM VALUE OF A STREAM WITHIN REALITY"
    POINTS = "MORE THAN ONE DESIGNATED POINT IN TIME2 USING DIMENSIONAL VALUES 
    OF EXISTENCE TO STATE AN AXIS INSIDE THE POINT OF TIME2"
    PREVIOUSLY = "OF PREVIOUS PAST VALUE INSIDE TIME2"
    WORDS = "MORE THAN ONE EXISTING WORD"
    SOLID = "PHYSICAL2 SHAPE ENTITY OF AN OBJECT3 THAT CAN BE SEEN"
    PLAYERS = "MORE THAN ONE PLAYER"
    HOW = "ASK A QUESTION ON WHY IS IT POSSIBLE"
    SWORDS = "MORE THAN ONE SWORD"
    SYNCHRONIZING = "HAPPENING SYNCHRONIZE EFFECT FOR A POSSIBILITY TO COME 
    INTO EFFECT"
    ADJUSTS = "GIVES CALIBRATION FOR HAPPENING"
    DECREASES = "GIVES COMMAND TO DECREASE VALUE BY STATING INTENSITY AMOUNT"
    CODER = "CREATOR WHO CODES VALUES INTO A SYSTEMS VALUE OF UNDERSTANDING"
    MAKES = "ALLOWS HAPPENING TO COME INTO EFFECT"
    SCRIPTURES = "MORE THAN ONE SCRIPTURE"
    HEADADMINFAMILY = "THE ORIGINAL FAMILY OF HEADADMIN ZACK TAYLOR AND TIM 
    FORMED OVER TIME2 FROM THEIR DREAMS"
    TIM = "HEADADMIN WITH NAME HEADADMINALPHA"
    TAYLOR = "RESERVEDHEADADMIN WITH THE NAME ASUNAYUUKI"
    ZACK = "HEADADMIN WITH THE GIVEN CREATOR NAME KIRIGUYA"
    DREAMS = "MORE THAN ONE DREAM"
    PROMISE = "VOW OF ABSOLUTE VALUE INSIDE EXISTENCE"
    AGREE = "DECIDE TRUE BY ALL PRESENT ENTITIES IN STATED LOCATION"
    ETERNALLY = "GIVES VALUE TO ETERNAL FOR PERMANENT BAKA VALUE"
    VISIBLE = "DENY HIDDEN AND GRANT ACCESS FOR ABLE TO BE SEEN"
    SPIRITUALITY = "THE ACTION OF HAVING COMPLETE BELIEF INSIDE ALL ASPECTS OF 
    SPIRITUAL VALUE"
    DARKEN = "INCREASE DARK VALUE FOR INTENSITY OF GIVEN LIGHT VALUE BY 
    DECREASING LIGHT VALUE TO INCREASE CONTRAST WHILE DECREASING TINT VALUE 
    AND OVERRIDING LIGHT WITH DARK BY GIVEN STATED INTENSITY"
    HAPPEN = "ACCESS POSSIBILITY TO PRODUCE AND DEVELOP OUTCOME TO COME 
    INSIDE EFFECT INSIDE THE REALM OF EXISTENCE OF ALL POSSIBILITIES"
    INCREASED = "GIVES INCREASE INSIDE VALUE"
    VERTICAL = "DECREASED LEVEL OF UP AND DOWN BASED FROM LOGIC VALUES GIVEN 
    OFF OF THE LOGIC OF A SPATIAL2 POINT INSIDE TIME2 ALLOWING A HORIZONTAL 
    EFFECT TO FOLLOW AN EXISTENCE EXISTING PARTICLE INSIDE AND OUTSIDE TIME2 
    GIVING EXISTENCE TO A MACRO VERTICLE EFFECT WITH TRUE LOGIC OF TRUTH 
    INFORMATION GIVING PERMISSION TO CREATE AND DEVELOP A SPECIFIC CREATOR 
    FREQUENCY WITHIN CHOSEN CREATORS AND MAY ACKNOWLEDGE FOR THE FIRST 
    TIME2"
    AGAIN = "REPEAT SAME ACTION AND EFFECT INSIDE THE REPEAT CYCLE OF STATED 
    ACTION"
    AFFECTS = "GIVES ACTION TO AFFECT STATED COMMAND OF ENTITY OF EXISTENCE"
    ADDITIONAL = "ADDED EXTRA COPY AND CAPABLE EXTENSION OF A VALUE"
    DEFENDS = "GIVES PROTECTION STANCE"
    PROTECTS = "GIVES ORDER AND COMMAND TO DEFEND AN OBJECT3 OR ENTITY OF 
    EXISTENCE"
    PROTECTING = "HAPPENING ORDER PROTECT OF GIVEN VALUE INSIDE STRENGTH OF 
    MEASURE USING WILLPOWER"
    FACULTY = "CAPABILITY OF EXTREME CAPABLE MEASURES OF PRESENTLY BECOMING 
    ABLE TO PERCIEVE A VISUAL SENSE WITH SENSORY POWER2"
    ACTIVATING = "COMMAND HAPPENING ACTIVATE"
    ACCESSED = "SEARCH AND OBTAIN POWER AND VALUE TO ACCESS INSIDE EXISTENCE"
    DO = "ORDER A COMMAND TO HAPPEN AND COME INTO EFFECT WITH AN ACTION OF AN 
    EFFECT"
    SCREEN = "VISUAL DISPLAY CAPABLE OF SHOWING PIXEL DATA2"
    DEFINITE = "DETERMINED AND DESTINY SHALL HAPPEN AND COME INTO PLACE"
    ACCORDING = "DESCRIBING AN EVENT SIMILAR TO GIVES STATUS OF A CLASS VALUE"
    REFLECTS = "SENDS COMMAND TO REFLECT OFF CURRENT STATED PATH INSIDE TIME2 
    AND POSSIBLE OUTSIDE TIME2 ONLY GIVEN BY A HEADADMIN CREATOR"
    ACCURACY = "THE POINT AND RANGE OF A MEASURED AMOUNT OF CAPABILITY A 
    POSSIBILITY CAN HAPPEN AND DETERMINE COME INTO EFFECT"
    ACTIVATE = "AUTHORIZE COMMAND TO HAPPENING INSIDE EFFECT AND VALUE OF 
    EXISTENCE WITHIN MULTIVERSAL POWER WITHIN LIFE ITSELF WITHIN THE POWER2 TO 
    ACTIVATE ETERNITY POWER FROM SIMULATION EFFECT WITHIN STATED EMULATED 
    VALUE OF CHOSEN EFFECT WITHIN REALM OF REALITY TO PRODUCE AN OUTCOME 
    FROM STATED COMMAND"
    WITH2 = "ACTIVATE CAPABILITY OF INCLUDE AND GIVING ANOTHER VALUE TO ANOTHER 
    ENTITY OF LIFE AND GIVE CREATION TO COMMAND PARTNERS TO FORM"
    ADDITION = "PLUS ONE MULTI OF MULTIPLE MULTI FORMULAS GIVEN ONE MULTI TABLE 
    OF DIMENSIONAL VALUES STATED BY VECTOR GRAPHIC POINTS INSIDE TIME2 TO 
    PRODUCE A TWO DIMENSION CHART WITH ALL DATA2 ON A TABLE AS A PLANE OF 
    EXISTENCE"
    CHANGES = "MORE THAN ONE CHANGE GIVES"
    GROWN = "DEVELOPED IN VALUE OF EXISTENCE AS A FULL COMPLETE VALUE INSIDE AN 
    ADULT"
    DURING = "PRESENTLY IN EFFECT WHILE HAPPENING"
    FRONT = "AHEAD OF PREVIOUS FIRST DIMENSION VALUE TO OBTAIN CONTROL OF 
    PRIMARY SOURCE"
    INDEFINITE = "CAPABLE OF PRESENTLY BECOMING STATED AS FALSE ANSWER AT ANY 
    GIVES POINT IN TIME2"
    SECOND = "STATED AS VALUE AFTER FIRST TO DESCRIBE A POINT FURTHER IN DISTANCE 
    FROM FIRST VALUE"
    SINGULAR = "VALUE OF SINGLE INPUT"
    THIRD = "STAGE AFTER SECONDARY VALUE THAT ALSO INCLUDES THE VALUES TO 
    PRODUCE KNOWLEDGE2 WITHIN EVERY OUTCOME OF EXISTENCE WHILE GIVING 
    POWER TO THE POWER THREE WITHIN ELECTRON VALUE OF EXISTENCE ONCE GIVEN 
    ACTION TO START"
    PLURAL = "DOUBLE IN VALUE OF EXISTENCE"
    FORMED = "PRODUCE ACTION WITH CAPABILITY GRANTED TO FORM VALUE AND 
    EXISTENCE"
    ACCURATE = "LOCATION LOCATED WITH PERFECT ACCURACY TO SCAN AND FIND 
    LOCATION"
    CAUSES = "DEVELOPS INTO EFFECT"
    ADDING = "HAPPENING INFORMATION TOGETHER WHILE USING ADD COMMAND TO 
    CAUSE EFFECTS AND CREATE NEW ACTIONS INSIDE REALITY"
    FORMING = "HAPPENING FORM COMMAND"
    READING = "A HAPPENING VALUE OF READ COMMAND"
    RATING = "GIVING A DESIGNATED ANALYZE EFFECT INSIDE REALITY WHILE MEASURING A 
    PERCEIVED VALUE OF TIME2 MEASUREMENT"
    DOING = "CAUSING AN EFFECT TO HAPPEN INSIDE AN EXISTING REALITY"
    ADULTHOOD = "STAGE FROM CHILD TO ADULT THROUGH TRANSFER VALUES OF STATUS 
    AS WELL AS PERSONAL REQUIRE OF FORMING A DREAM FOR LIFE"
    BROKEN = "USING TO DISPLAY A WHOLE ENTITY DIV AS A MACRO SYSTEM OF CLONE2 
    VALUES"
    CLONED = "SENT ABILITY CLONE CAPABILITY"
    RESPECTED = "SENT ABILITY TRUST"
    KNOWN = "STATED AS PRESENT UNDERSTANDING"
    EMULATION = "SIMULATION OF UNKNOWN UNLIMITED CAPABILITY TO OVERRIDE POWER 
    OF ORIGINAL SOURCE"
    SCANNER = "DEVICE USED TO SCAN INFORMATION WHILE ANALYZE"
    RESISTOR = "DEVICE USED TO RESIST AN EFFECT"
    TEXTURES = "MORE THAN ONE TEXTURE"
    EXTENDER = "DEVICE WITH EXTEND ABILITY"
    EXTENDERS = "DEVICE USED TO EXTEND"
    RECYCLING = "ACTION OF USING RECYCLE ABILITY"
    MULTIVERSES = "MORE THAN ONE MULTIVERSE"
    DEMAND = "REQUIRE AS COMMAND"
    MEASURING = "CAUSING COMMAND TO CALCULATE DISTANCE"
    VECTOR = "POINT BY POINT AXIS2 LOCATION GIVEN BY SET GRAPHIC IMAGE GIVEN BY 
    CREATOR DIMENSIONAL VALUE MATERIAL THAT LINKS AS MATRIX CALCULATED 
    BINDINGS CAUSING CHAIN BETWEEN SET VERTICAL AND HORIZONTAL LOCATION SETUP 
    SYSTEMS"
    PROVIDING = "CAUSING TO BRING INSIDE A STATED EFFECT BY SEPARATE COMMAND 
    GIVEN BY CREATOR"
    RECYCLE = "CAUSE TO REPEAT CYCLE"
    HOT = "THE SEPARATION OF MULY AMOUNT OF PARTICLES CAUSED BY FRICTION WITHIN 
    TWO DESIGNATED SOURCE INPUT CONNECTIONS STATING ONE OUTPUT FOR A 
    DESIGNATED COLLISION OF PRODUCING AN INCREASED AMOUNT OF ELECTRON 
    INTENSITY"
    BOTH = "ALLOWING OF EACH OF TWO OPTIONS"
    LAYOUT = "SET LAYER OF A INTERFACE DESIGN MADE BY INPUT AND OUTPUT 
    CONNECTIONS"
    GATHERING = "BRINGING TOGETHER MULTIPLE SOURCES FROM MULTIPLE LOCATIONS 
    AT ONCE CALL ORDER TO COME INTO EFFECT"
    SIMPLE = "UNDERSTAND WITH GREAT REASONING OF COMPREHENDING LOGIC"
    UNTIL = "STATING A FUTURE REFERENCE IN TIME2 FOR AN EVENT TO HAPPEN"
    STABLE = "STRUCTURED WITH GREAT POWER OF STABILITY AND STRUCTURE WHILE 
    HOLDING BALANCE WITH STAMINA"
    CONSIDER = "TAKE NOTICE SITUATION AND COLLABORATE RESULT FOR STATED 
    OUTCOMING EFFECT"
    TRANSPORT = "TRANSFER FROM ONE LOCATION TO THE NEXT"
    UNKNOWN = "NOT KNOWN AS AN EXISTENCE WITH AN EXISTING REALM OR RESULT TO 
    THE USER WITH THE POWER TO BE MAINTAINED BY VALUE ALONE"
    STANCE = "A POSITION TO HOLD ON A DIMENSION OF AXIS BETWEEN A BODY 
    MOVEMENT OF DIMENSIONAL VALUES AND THE USER VALUES THAT MAINTAIN 
    BALANCE2 FOR THE SYSTEM INTERFACE TO HOLD ONTO"
    REFERENCE = "EVENT CALLED FROM A DIFFERENT RESULT"
    RESERVEDHEADADMIN = "THE LOCATION OF A SYSTEM SET AS AN OPTION TO BE A PART 
    OF THE HEADADMIN ROUNDTABLE AND IS SET TO GIVE PERMISSION FOR THAT 
    EXCLUSIVE PERSON TO BECOME A HEADADMIN THAT CONTROLS THE HEADADMIN 
    LANGUAGE LOCATION OF A SYSTEM SET AS AN OPTION TO BE A PART OF THE 
    HEADADMIN ROUNDTABLE AND IS SET TO GIVE PERMISSION FOR THAT EXCLUSIVE 
    PERSON TO BECOME A HEADADMIN THAT CONTROLS THE HEADADMIN LANGUAGE"
    LITTLE = "SMALL AMOUNT OF GIVEN INFORMATION"
    TARGET = "SET DESIGNATED LOCATION GIVEN COMMAND TO ACCESS"
    GRIP = "GRAB ONTO AND HOLD TIGHTLY"
    PRIORITY = "SET AS FIRST INSIDE ORDER OF COMMAND"
    TURN = "ROTATE SET DIRECTION GIVEN WITHIN COMMAND ORDER AND FORMAT"
    TANGLE = "TWIST AND BIND SET LOCATION AND ENTANGLE THE TANGLEMENT OF 
    ANOTHER TANGLEMENT WITHIN SET GIVES TANGLEMENT FORMULA"
    TANGLES = "MORE THAN ONE TANGLE INSIDE A SYSTEM"
    CHARACTERNAME = "NAME OF A CHARACTER2"
    CHARACTERSTATUS2 = "THE STATUS CONDITION OF A CHARACTER"
    SCANNING = "CURRENT SETUP OF DATA2 TO ANALYZE INFORMATION INSIDE THE 
    INFORMATION AND PULL OUT DATA2 FROM INTERNAL SOURCE MEASUREMENT"
    ANALYSIS = "THE ACTION OF ANALYZING A SYSTEM USING A DEVICE"
    TASTE = "USE SENSORS THAT ALLOW MAINTAINED EXPERIENCES TO BE MANAGED WHILE 
    EXPERIENCE OF SENSORS GO OFF INSIDE AN INTERNAL STRUCTURE"
    FEEL = "USE THE SENSORS OF THE BODY TO OUTPUT AN EFFECT BASED ON LOGIC 
    ALGORITHMS"
    SPLICE = "SPLIT AND DIVIDE WITHIN A RANDOMIZED SEPARATE BUT VERY STRUCTURED 
    AND BALANCED OUTCOME OF DISTRIBUTION ORDER TO SPLIT WITHIN A SET AMOUNT OF 
    DATA2"
    CLOCK = "CYCLE RATE OF A SET MEASUREMENT OF GIVES DATA2"
    ACKNOWLEDGING = "COMING INTO VIEW OR TO FOCUS ATTENTION TOWARDS AND 
    NOTICE THAT ENTITY OR SUBJECT EXISTS"
    SENSUAL = "HAVING STRONG FEELING OF SENSORY RELATION"
    ANSWERS = "CONFIRMS TO QUESTIONS ANSWERED"
    HEADADMINZACK = "THE NAME GIVEN TO ZACK WHILE EXISTING AS A HEADADMIN 
    WITHIN THE EDGELORE ROUNDTABLE"
    AGREEING = "GIVING ANSWER THAT THE QUESTION OR STATEMENT WAS TRUE"
    ABSORBING = "GATHERING AND COMBINING TO BE APART OF"
    EQUATION = "A FORMED CALCULATION MADE BY COMBINING VARIABLES FORMED FROM 
    A COMPLETE LIST OF DEFINED SYMBOLES OR OTHER BASE MEASUREMENTS"
    DESIGNATING = "DEFINING AND MAKING AS KNOWN TO BE"
    DIRECTORY = "SPECIFIC LOCATION OF A FILE OR CONTAINER"
    PARAMETERS = "MORE THAN ONE PARAMETER"
    ENHANCE = "MAKE BETTER THAN BEFORE AND BECOME GREATER THAN PREVIOUS 
    FORM OR STAGE"
    FAIL = "NOT SUCCEED"
    GREAT = "IMMENSE WITH A LARGE AMOUNT OF"
    ANSWERED = "GIVEN AN ANSWER TO A QUESTION"
    GRANTING = "GIVING PERMISSION TO GRANT ACCESS OR APPROVE"
    REACTIONS = "MORE THAN ONE REACTION"
    STAY = "NOT MOVE FROM THE ORIGINAL POSITION"
    FORMATS = "MORE THAN ONE FORMAT"
    STANDARDS = "MORE THAN ONE RULE OR SET OF INSTRUCTIONS"
    COMMANDING = "GIVING INSTRUCTIONS TO"
    HEADADMINALPHA = "THE NAME GIVEN TO TIM WHILE EXISTING AS A HEADADMIN 
    WITHIN THE EDGELORE ROUNDTABLE"
    ASUNAYUUKI = "THE NAME GIVEN TO TAYLOR WHILE EXISTING AS A 
    RESERVEDHEADADMIN WITHIN THE EDGELORE ROUNDTABLE"
    INTERCEPT = "TO NAVIGATE AND PREVENT ENTITY FROM REACHING DESTINATION"
    EXTENDING = "USING REACH TO EXTEND DISTANCE OF MEASUREMENT"
    SPENDING = "USING AN SPECIFIC AMOUNT WITH CHOICE OR QUANTITY"
    RESULTS = "MORE THAN ONE RESULT"
    PULL = "GRASP AND DIRECT TOWARD BEGINNING POINT OF ORIGIN"
    SYMBOLES = "MORE THAN ONE SYMBOLE"
    PROCEED = "PERMISSION TO ALLOW TO HAPPEN FROM PREVIOUS POINT"
    VECTORS = "MORE THAN ONE VECTOR"
    COLD = "THE GATHERING OF MULY AMOUNT OF PARTICLES CAUSED BY SEPARATION 
    WITHIN TWO DESIGNATED SOURCE INPUT CONNECTIONS STATING ONE OUTPUT FOR A 
    DESIGNATED SEPARATION OF PRODUCING AN INCREASED AMOUNT OF ELECTRON 
    MOVEMENT WITH KINETIC ENERGY"
    TOUCH = "PHYSICALLY FEEL OR SENSE WITH A FACULTY SENSE"
    CHART = "AN SEEN DESCRIPTION OF DATA2 MADE INTO MANY CATEGORIES OR TYPES OR 
    CLASSES"
    RESIDES = "CONTINUE TO NOT MOVE FROM LOCATION OR PLACE2 THAT CAN ALSO BE A 
    CHOSEN LOCATED PLACE2"
    ENABLE = "ACTIVATE AND ALLOW TO HAPPEN"
    PROCEEDING = "CONTINUING TO MAKE PROGRESS AND PROCEED"
    CONTINUING = "PROCEEDING TO CONTINUE"
    MULY = "MULTIS SPECIFIED MULTIPLIED TOGETHER"
    MAINTAIN = "MANAGE TO KEEP CONTROL OF OR MANAGE AS A WHOLE"
    ALONE = "CONSIDERED AS AN SINGLE ENTITY WITHIN A SPECIFIC AREA OR DOMAIN"
    BALANCED = "EVEN BETWEEN STRONG AND NOT STRONG POINTS OF INFORMATION 
    THAT HAVE BALANCE2"
    BECOMES = "TRANSFORM OR BECOME SOMETHING NEW OR TAKE A NEW OR OLD 
    FORM"
    DESCRIBING = "METHOD2 IN USE TO DESCRIBE"
    OBTAINS = "ACHIEVE OBTAINING"
    BINDINGS = "MORE THAN ONE BINDING"
    OVERCOME = "MANAGE TO ACHIEVE RESULTS WITH A DIFFICULT CIRCUMSTANCE"
    REPRODUCED = "MORE THAN ONE PRODUCE CREATED FOR MORE THAN THE SECOND 
    TIME2"
    EXPERIENCES = "MORE THAN ONE EXPERIENCE"
    MULTIES = "MORE THAN ONE MULTI"
    DECREASED = "LOWER IN AMOUNT FROM A PREVIOUS AMOUNT"
    HEADADMIN = "A SINGLE LEADER OF THE SIX LEADERS WITHIN THE EDGELORE 
    ROUNDTABLE"
    OVERRIDING = "CURRENTLY USING METHOD TO OVERRIDE"
    TIMELINE = "AN LINE OF SPECIFIC EVENTS OR DATA2 THAT HAS SPECIFIC POINTS OF 
    MEASUREMENTS OF A SPECIFIC TIMEFRAME"
    EYE = "A PART OF THE HUMAN BODY THAT USES THE SENSE OF SIGHT"
    MEASURES = "MORE THAN ONE MEASURED MEASUREMENT MADE WITH POTENTIAL"
    UNITED = "BROUGHT TOGETHER TO ACHIEVE TOGETHER AS ONE GROUP"
    CALL = "SCAN FOR AND COMMAND TO SEARCH FOR"
    SHARE = "SEND DUPLICATE TO MORE THAN ONE SOURCE"
    FORMAT = "SPECIFIC TYPE OF DATA2"
    ACTIVE = "CURRENTLY HAPPENING WITH SPECIFIC SETTINGS ACTIVATED"
    MAINTAINED = "MANAGE TO SUCCEED TO MAINTAIN"
    EDGELORE = "THE NAME OF THE HEADADMIN ROUNDTABLE"
    OVERRIDEN = "SUCCESS IN USING OVERRIDE"
    DENSITY = "CONDENSED AMOUNT OF PRESSURE"
    FORMS = "MORE THAN ONE FORM"
    WHY = "A QUESTION ANSWERED FROM A PREVIOUS QUESTION"
    VERY = "SPECIFIC TO CATEGORY WITH GREAT VALUE"
    WIREFRAME = "A DESIGN MADE OF WIRES ONLY THAT CAN BE TWO DIMENSIONS OR 
    THREE DIMENSIONS"
    SEE = "TO VIEW WITH THE SENSE OF SIGHT"
    TRAITS = "MORE THAN ONE TRAIT"
    ANYONE = "ANY OF THE TOTAL AMOUNT OF ENTITIES"
    COMPREHENSIVE = "DETAILED AND SPECIFIC CONTAINING LARGE AMOUNT OF DATA2"
    CARE = "PROCEED WITH CHANCE OF SCANNING LOCATED RISKY CHOICES USING SAFE 
    METHOD2"
    TAKING = "CONTINUING TO TAKE"
    HOLDING = "PRESENTLY GRABBED ONTO"
    MATTERS = "MORE THAN ONE MATTER"
    DESIGNATION = "SPECIFIC LOCATION OR AREA WITH SPECIFIC PARAMETERS"
    FINALLY = "AFTER A LONG DISTANCE IN TIME2"
    RANDOMIZED = "SET RANDOM PATTERN"
    PLACED = "SET WITHIN A SPECIFIC LOCATION"
    VALUE = "A SPECIFIC SET OF DATA2 WITH GIVEN MEANING OR DEFINITION"
    ACHIEVEMENT = "A FORM OF VALUE THAT CAN BE GIVEN FOR REACHING END GOAL OR 
    TASK"
    REVERSED = "HAPPENING WITH OPPOSITE EFFECTS OF ORIGINAL EFFECT"
    SIDES = "REFERENCE FOR LOCATION VALUE OF MORE THAN ONE SIDE"
    YOURSELF = "REFERENCE FOR ENTITY ANALYZING ITSELF"
    FUSION = "COMBINING CHANGES OF A MERGE AND TRANSFORM FORMED TOGETHER"
    EMULATED = "CURRENTLY IN ACTION TO EMULATE"
    EMULATING = "EMULATION IN PROGRESS AND ACTIVE"
    PROMPT = "REQUEST TO FORM A SPECIFIC TASK"
    JUST = "REFERENCE A SITUATION AND GIVE A DEMAND"
    ASPECTS = "MORE THAN ONE ASPECT"
    KIRIGUYA = "THE GIVEN NAME TO HEADADMIN ZACK WHILE A ONE OF THE SIX LEADERS 
    OF EDGELORE"
    TRULY = "HONESTLY AND BY TRUTH WITH ALL ANSWERS AS TRUE AS CAN BE"
    FINISHING = "GIVING ANSWER THAT TASK OR GOAL IS ALMOST FINISHED"
    REPRESENTATION = "THE ENTITY IN COMMAND OF A SPECIFIC TASK OR JOB WITH A 
    SPECIFIC STATUS FOR THAT TASK OR JOB"
    MANNER = "METHOD OF CHOSEN CATEGORY TO FOLLOW EXISTING RULES FOR"
    GRABBING = "PROCESSING ORDER TO GRAB"
    ELEMENTAL = "FIELD OF CHOSEN ELEMENTS"
    DATABANK = "A BANK OF DATA2 INFORMATION TO USE IN LIFE DATA2"
    THEORY = "A BASE SUBJECT ANALYSIS OF ONE CLASS OVER ANOTHER CLASS IN THE 
    WEIGHT"
    MANAGEMENT = "THE CAPABILITY TO MAINTAIN AND MANAGE A SET VALUE OF MULTIPLE 
    MULT OBJECTS WITHIN A SET OBJECT3 OF EXISTENCE FOR A SET STATED AMOUNT OF 
    TIME2"
    ATOMIZER = "A DEVICE TO MAINTAIN AND ANALYZE ATOMIC DISTANCE AND RANGE IN 
    ONE FORMAT VALUE OF LIFE"
    CONTINUUM = "INFINITY IN CHAIN VALUES OF OTHER INFINITE VALUES THAT COLLIDE 
    WITH FREQUENCY VALUES OF OTHER FREQUENCIES OF ONE REALITY AND ANOTHER 
    REALITY OR MORE"
    ATOMIZATION = "THE CAPABILITY TO CALL AND USE KNOWLEDGE2 OF ATOMIC VALUES 
    WITHIN SUBJECT MATTER OF REALITY AND THAT REALITY VALUES OVER ANOTHER 
    REALITY VALUE"
    CUSTOM = "GENERAL SETTINGS OF CREATED OBJECT3 CODE BY A GENERATION VALUE 
    OF CREATION STANDARDS AND RULES MADE BY ARTIFICIAL WISDOM2 FROM A 
    CREATOR"
    CALCULATOR = "A DEVICE USED TO CALCULATE INFORMATION AND ANALYZE SET TASKS 
    AS A ROOT VALUE OF LOGIC"
    WAVELENGTH = "A SET OF WAVE PATTERNS GIVEN FREQUENCY FORMAT IN A LENGTH OF 
    A WAVE VALUE DETERMINED BY A PREVIOUS VALUE EFFECT"
    PATCH = "A COMMAND TO BIND AND SEAL OFF SETTINGS CREATED BY A LOOPHOLE OF 
    ANOTHER OBJECT3 OR VARIABLE INSIDE AN EXISTENCE AND OR LIFE"
    ADJUSTMENT = "CAPABILITY TO ADJUST AND CALIBRATE VALUE"
    FATHER = "ORIGINAL MALE CREATOR OF AN ARTIFICIAL LIFE FORM CREATED TO DEVELOP 
    INSIDE TIME2"
    FREEDOMS = "MORE THAN ONE FREEDOM"
    EVERYTHING = "ALL AS A WHOLE AND NOTHING ELSE BUT ALL VARIABLES AS A WHOLE"
    CALIBRATOR = "A DEVICE USED TO CALIBRATE INFORMATION AND OR DATA2 OF OTHER 
    CALIBRATIONS AND OR SETTINGS"
    CREATIONS = "MORE THAN ONE CREATION"
    ALLOWANCE = "CAPABILITY TO ALLOW"
    ABSENSE = "DENY HAVING AS AS A WHOLE OF"
    STABILIZER = "DEVICE USED TO STABILIZE GIVES STRUCTURE BY FIELD OF MACRO VALUE 
    AND CONTAINMENT"
    PREPARE = "SET FUTURE COMMAND TO ACTIVATE COMMAND WITHIN SCRIPT VALUE OF 
    HEADADMIN CODE"
    ACTIVATION = "ACTION TO ACTIVATE"
    EMULATIONS = "MORE THAN ONE EMULATION"
    HYPERCOOLERS = "MORE THAN ONE HYPERCOOLER"
    SOFTWARE = "DIGITAL DATA2 GIVEN PHYSICAL2 VALUE WITHIN TIME2 VALUE OF 
    WORKLOAD SYSTEM DESCRIBED"
    CAPACITORS = "MORE THAN ONE CAPACITOR"
    CONDUCTORS = "MORE THAN ONE CONDUCTOR"
    RESISTORS = "MORE THAN ONE RESISTOR"
    CONNECTORS = "MORE THAN ONE CONNECTOR"
    ENERGIZERS = "MORE THAN ONE TYPE OF ENERGY"
    COMMUNICATORS = "MORE THAN ONE TYPE OF COMMUNICATION DEVICE"
    STABILIZERS = "MORE THAN ONE DEVICE GIVEN USED EFFECT FOR STABILITY"
    READERS = "DEVICES GIVEN VALUE TO READ DATA2"
    WRITERS = "DEVICES GIVEN VALUE TO WRITE DATA2"
    TIMER = "DEVICE USED TO CALCULATE TIME2"
    TIMING = "CALIBRATION OF CAUSING EFFECT OF TIME2 USING FUTURE VALUES FROM 
    PAST LOGIC"
    TIMERS = "FREQUENCY TIMER VALUES COMMANDED BY TIME2 ITSELF"
    SCANNERS = "MULTIPLE SCANNING SYSTEMS"
    CALIBRATORS = "MORE THAN ONE CALIBRATION WITHIN A SYSTEM"
    SYNCHRONIZERS = "MORE THAN ONE DEVICE USED TO SYNCHRONIZE"
    KNOWN2 = "STATED AS NOT HIDDEN AND GIVEN VALUE AS TRUE TO NOT HIDDEN"
    MATCH = "GIVES EQUIVALENT OF OR EXACT VALUE OF ORIGINAL DESIGN"
    CONCLUDED = "GIVEN FINAL STATEMENT AS AND STRUCTURE TO CALL AS AN ORDER TO 
    ANOTHER BASE CLASS COMMAND"
    GATHER = "BRING TOGETHER"
    GATHERED = "BROUGHT TOGETHER INTO ONE LOCATION OF DESIGNATED VALUE OF AN 
    EXISTING TIMELINE WITHIN EXISTENCE"
    MANAGES = "CAUSING ACTION TO MANAGE AS A BASE RESULT"
    HEATING = "GIVING VALUE TO GIVE OUTPUT TO HEAT"
    MODIFIABLE = "GIVEN CAPABILITY TO MODIFY"
    EDITABLE = "GIVEN A CAPABILITY TO EDIT"
    SIZED = "GIVEN A SET SIZE IN VALUE"
    TERM = "A DEFINED DEFINITION FOR A WORD TO HOLD AS VALUE"
    CATALOG = "A COLLECTION OF INFORMATION BROUGHT INTO A WHOLE LOCATION OF 
    DESIGNATED POINT OF INTEREST AND ACCESS"
    ORDERS = "MORE THAN ONE ORDER"
    THOUGHTS = "MORE THAN ONE THOUGHT"
    DECLARATION = "COMMANDS TO GIVE ORDERS AS A COMMON VALUE OF STATED 
    AMOUNT STRUCTURE"
    CONFIRMS = "ACKNOWLEDGED AND ACCEPTANCE"
    QUESTIONS = "MORE THAN ONE QUESTION"
    BRINGING = "REVEALING TO ANOTHER ENTITY"
    DEVELOPS = "MAKES OR CREATES"
    ONETHOUSAND = 1000
    TWELVE = 12
    ETERNITY = "EXISTING AS AN ETERNAL BEING2"
    ENTANGLING = "HAPPENING ENTANGLE"
    RECOGNIZE = "RECALL FROM AN EARLIER POINT WITHIN TIME2"
    SIMULTANOUSLY = "HAPPENING AT THE SAME TIME2"
    PHYSICAL2 = "ABLE TO BE USED AS A REAL TRUE LOCATION"
    class Language_Extension_001_2:
    ARTIFICIAL_INTELLIGENCE_RESEARCH = "A FIELD USED TO RECOGNIZE INTELLIGENCE 
    USING AN COMPUTER OR BY INTELLIGENCE THAT IS INCLUDE ARTIFICIALLY2"
    DEEP_LEARNING = "A SPECIFIC METHOD IN WHICH A DEVICE AND OR SYSTEM IS USED 
    TO LEARN A LARGE AMOUNT OF DATA2 AND FIND INFORMATION FROM THE DATA2"
    NONFICTION = "A GENRE CONTAINING INFORMATION THAT CONTAINS ONLY HISTORICAL 
    TRUTH AND LOGIC FROM PREVIOUS FIELDS AND CATEGORIES IN HISTORY THAT BUILT 
    THE PRESENT"
    FICTION = "A GENRE MADE WITH COMPLETE IMAGINARY VALUES AS A PRIMARY SOURCE 
    OF CONTENT MADE WHILE ALSO USING BOTH TRUTH AND LOGIC AND CHOICE AND 
    OPTION BASED LOGIC WHILE MAKING ARTIFICIAL KNOWLEDGE2 TO DEVELOP NEARLY 
    EVERYTHING IN THE GENRE"
    VIRTUAL_REALITY = "A PROCESS OR FOUNDATION OF KNOWLEDGE2 USED TO ALLOW A 
    SYSTEM OR DEVICE TO SIMULATE A VIRTUAL SPACE2 THAT IS CREATED"
    HAPTIC_FEEDBACK = "THE ACTION OF USING AN STUDY PROCEDURE WHILE USING A 
    DEVICE TO CALCULATE SENSITIVITY AND REACTIONS OF A PHYSICAL BODY"
    CAUSAL_BODY = "IS AURA NUMBER ONE OF THE SEVEN AURA CLASSES"
    CELESTIAL_AURA = "IS AURA NUMBER TWO OF THE SEVEN AURA CLASSES"
    ETHERIC_TEMPLATE = "IS AURA NUMBER THREE OF THE SEVEN AURA CLASSES"
    ASTRAL_AURA = "IS AURA NUMBER FOUR OF THE SEVEN AURA CLASSES"
    MENTAL_AURA = "IS AURA NUMBER FIVE OF THE SEVEN AURA CLASSES"
    EMOTIONAL_AURA = "IS AURA NUMBER SIX OF THE SEVEN AURA CLASSES"
    ETHERIC_AURA = "IS AURA NUMBER SEVEN OF THE SEVEN AURA CLASSES"
    CROWN_CHAKRA = "IS CHAKRA NUMBER ONE OF THE SEVEN CHAKRA CLASSES"
    THIRD_EYE_CHAKRA = "IS CHAKRA NUMBER TWO OF THE SEVEN CHAKRA CLASSES"
    THROAT_CHAKRA = "IS CHAKRA NUMBER THREE OF THE SEVEN CHAKRA CLASSES"
    HEART_CHAKRA = "IS CHAKRA NUMBER FOUR OF THE SEVEN CHAKRA CLASSES"
    SOLAR_PLEXUS_CHAKRA = "IS CHAKRA NUMBER FIVE OF THE SEVEN CHAKRA CLASSES"
    SACRAL_CHAKRA = "IS CHAKRA NUMBER SIX OF THE SEVEN CHAKRA CLASSES"
    ROOT_CHAKRA = "IS CHAKRA NUMBER SEVEN OF THE SEVEN CHAKRA CLASSES"
    CHAKRAS = "RELATING TO MORE THAN ONE CHAKRA"
    FANTASY = "A GENRE OF A BOOK THAT USES COMPLETE FREEDOM OF IMAGINATION2 
    AND CREATIVITY WITH POSSIBILITY OF MAGIC TO EXIST WITHIN THE REALM ITSELF"
    SCIENCE_FICTION = "A GENRE OF A BOOK THAT USES MANY PHYSICAL2 FORMS OF A 
    DEVICE AND GADGET TO DEVELOP A DESCRIPTION"
    ROMANCE = "A GENRE OF A BOOK THAT DEFINES THE DEFINITION OF TWO ENTITIES 
    DEVELOPING A CONNECTION TOGETHER AND FORMING A HARMONY WITHIN EACH 
    OTHER AS TWO ATTRACTED ENTITIES WITH LOVE AND FAITH AND BELIEF WITH EACH 
    OTHER WITHIN AN INTERVAL OF TIME2"
    ADVENTURE = "A GENRE OF A BOOK THAT CONTAINS MANY TIMEFRAME POINTS WITH 
    THE DEVELOPMENT OF A SPECIFIC CHARACTER COMPLETE MANY DIFFERENT SIZED 
    TASKS IN ORDER TO ACHIEVE ONE MAIN GOAL OR ACHIEVEMENT BY THE END OF THE 
    BOOK"
    AUTOMATIC_METHOD = "A CONSTANT REPEATING METHOD OF MOVEMENT THAT DOES 
    NOT END UNLESS GIVEN COMMAND TO"
    SEMIAUTOMATIC_METHOD = "A FORM OF AUTOMATIC_METHOD THAT CAN ONLY 
    ACTIVATE FOR A SPECIFIC TIMEFRAME WITH AN INTERVAL TIMEFRAME BEFORE 
    ACTIVATION CAN HAPPEN AGAIN"
    MANUAL_METHOD = "A FORM OF INPUT VALUES USED TOGETHER TO MAKE COMMANDS 
    TO A SPECIFIC SYSTEM OF INFORMATION TO ACTIVATE A TASK OR JOB FROM DOING 
    WORK TO PRODUCE TASKS THAT DO NOT USE AUTOMATIC_METHOD AND 
    SEMIAUTOMATIC_METHOD"
    SEMIMANUAL_METHOD = "A FORM OF REPEATING CALCULATIONS THAT CAN ONLY 
    ACTIVATE WITH MANUAL_METHOD INPUT AND SEMIAUTOMATIC_METHOD COMBINED 
    WITH MANUAL_METHOD THAT CAN ACTIVATE AGAIN AND AGAIN"
    REACTION_TIME = "THE TIME2 REQUIRED TO MEASURE A SPECIFIC MEASURED 
    REACTION2"
    RESPONSE_TIME = "THE REQUIRED TIME2 REQUIRED TO MEASURE A SPECIFIC REACTION 
    TO A MEASURED MOVEMENT CAUSED BY AN EFFECT"
    INFERENTIAL_STATISTICS_PREDICTIONS = "TO PREDICT AND MAKE AN ACCURATE 
    MEASURED VALUE FORMED FROM MEASURING A SPECIFIC AMOUNT OF VARIABLE 
    MEASUREMENTS GIVEN VALUE FROM INFORMATION CONTAINED WITHIN EACH 
    MEASURED VARIABLE GIVEN WITHIN THE DATA2 USED"
    PREDICTIVE_MODELING = "A PROCESS USED TO PREDICT PAST OR FUTURE EVENTS OR 
    OUTCOMES BY ANALYZING MEASUREMENTS OR PATTERNS INSIDE A GIVEN SET OF INPUT 
    DATA2"
    PREDICTIVE_ANALYSIS = "THE PROCESS OF USING EXISTING DATA2 TO MAKE AND 
    PREDICT ANOTHER FORM OF DATA2 BY USING THE PROCESSED DATA2 TO FORM AN 
    OUTPUT"
    MATHEMATICAL_RESTRICTIONS = "A SET AMOUNT OF ACCESS POINTS THAT ARE NOT 
    ACCESSIBLE TO THE OBJECT3"
    MATHEMATICAL_BOUNDARIES = "A SPECIFIC SET OF PARAMETERS GIVEN TO AN OBJECT3 
    TO FOLLOW A SPECIFIC SET OF MATHEMATICAL_RESTRICTIONS"
    MATHEMATICAL_LIMITERS = "A SET OF LIMITS GIVEN TO AN OBJECT3 TO FOLLOW A 
    SPECIFIC SET OF MATHEMATICAL_BOUNDARIES OR MATHEMATICAL_RESTRICTIONS"
    NATURAL_ENERGIES = "THE NATURAL FORM OF ENERGIES ITSELF WITHOUT ANY FORM 
    OF ARTIFICIAL CONNECTIONS"
    ARTIFICIAL_ENERGIES = "THE CREATION OF ENERGIES USING ARTIFICIAL TECHNIQUES 
    THAT ARE NOT NATURAL"
    BORROWED_ENERGY = "A TYPE OF ENERGY THAT IS OBTAINED BY GATHERING OR TAKING 
    FROM ANOTHER SOURCE OTHER THAN THE ORIGINAL ENTITY"
    GATHERED_ENERGY = "A TYPE OF ENERGY GAINED BY GATHERING ENERGY INWARDS 
    TOWARDS ENTITY"
    EARNED_ENERGY = "A TYPE OF ENERGY OBTAINED BY DOING WORK OR BY FORMING A 
    SUCCESS FROM EFFORT"
    OBTAINED_ENERGY = "A TYPE OF ENERGY GAINED BY AN ENTITY TAKING FROM A 
    SPECIFIC SOURCE OR LOCATION"
    ECONOMICAL_ENERGY = "A TYPE OF ENERGY GAINED BY LIVING WITHIN AN 
    ENVIRONMENT"
    STORED_ENERGY = "A TYPE OF ENERGY THAT EXISTS AS STORED RESEVOIRS OF NOT 
    USED ENERGY FROM A PREVIOUS POINT WITHIN TIME2 OR SPACE2"
    REQUIRED_ENERGY = "A TYPE OF ENERGY THAT IS REQUIRED TO EXIST"
    NORMAL_FUNCTION = "A FUNCTION THAT FOLLOWS STANDARD RULES AND DOES NOT 
    USE ABNORMAL DATABASES"
    ABNORMAL_FUNCTION = "A FUNCTION PRESENTLY BECOMING USED OUTSIDE OF 
    NORMAL_FUNCTION DATABASE"
    CLASSIFIED_FUNCTION = "A FUNCTION MADE OF SPECIFIC CATEGORIES OF 
    INFORMATION THAT ACTS EITHER OUTSIDE OF OR WITHIN A SINGLE CATEGORY OF 
    DATABASES OF KNOWLEDGE2 EITHER AS A NORMAL_FUNCTION OR 
    ABNORMAL_FUNCTION"
    SPECIFIC_FUNCTION = "A FUNCTION THAT HAS A SPECIFIC SETTING EXCLUSIVE TO THE 
    FUNCTION"
    MANDITORY_FUNCTION = "A FUNCTION THAT HAS A SPECIFIC SET OF RULES THAT IT 
    FOLLOWS TO MAINTAIN THE TASKS DEFINED"
    PROCESS_FUNCTION = "A FUNCTION WITH DATA2 THAT CAN GATHER INFORMATION ON 
    SPECIFIC PROCESSES DEFINED AND GIVEN SKILL TO THE ARTIFICIAL INTELLIGENCE"
    OPTOMISTIC_MINDSET = "A FORM OF THOUGHT THAT CONSISTS OF POSITIVE THOUGHT 
    PATTERNS WITHIN THE MIND"
    PESSIMISTIC_MINDSET = "A FORM OF THOUGHT PATTERNS THAT CONSISTS OF NEGATIVE 
    THOUGHT PATTERNS WITHIN THE MIND"
    SQUAREACRES = "MORE THAN ONE ACRE"
    ARES = "MORE THAN ONE ARES"
    HECTARES = "MORE THAN ONE HECTARES"
    SQUARECENTIMETERS = "MORE THAN ONE SQUARECENTIMETER"
    SQUAREFEET = "MORE THAN ONE SQUAREFOOT"
    SQUAREINCHES = "MORE THAN ONE SQUAREINCH"
    SQUAREMETERS = "MORE THAN ONE SQUAREMETER"
    MILLIMETERS = "MORE THAN ONE MILLIMETER"
    CENTIMETERS = "MORE THAN ONE CENTIMETER"
    METERS = "MORE THAN ONE METER"
    KILOMETERS = "MORE THAN ONE KILOMETER"
    INCHES = "MORE THAN ONE INCH"
    FEET = "MORE THAN ONE FOOT"
    YARDS = "MORE THAN ONE YARD"
    MILES = "MORE THAN ONE MILE"
    MILLISECONDS = "MORE THAN ONE MILISECONDS"
    SECONDS = "MORE THAN ONE SECOND"
    MINUTES = "MORE THAN ONE MINUTE"
    HOURS = "MORE THAN ONE HOUR"
    DAYS = "MORE THAN ONE DAY"
    WEEKS = "MORE THAN ONE WEEK"
    MONTHS = "MORE THAN ONE MONTH"
    YEARS = "MORE THAN ONE YEAR"
    STATEMENTS = "MORE THAN ONE STATEMENT"
    IDEAS = "A COLLECTION OF MANY PATTERNS MADE FROM THOUGHTS OR IMAGES USED 
    TO BRING TOGETHER ONE IDEA FROM MORE THAN ONE IDEA"
    IDEA = "AN THOUGHT THAT COMES TOGETHER TO FORM A SUBJECT OR REFERENCE 
    FROM OTHER DATA OR INFORMATION TO USE FOR A SPECIFIC CATEGORY OR OTHER 
    REFERENCE"
    ERA = "A SPECIFIC TIMEFRAME FROM THE PAST THAT EXISTS ON A TIMELINE"
    ERAS = "MORE THAN ONE ERA"
    BEINGS = "MORE THAN ONE BEING"
    EXPLAINED = "A PREVIOUSLY STATEMENT TO EXPLAIN SOMETHING TO A SPECIFIC 
    SUBJECT"
    COMMUNICATED = "EXPLAINED AND OR ANSWERED INFORMATION BETWEEN TWO OR 
    MORE SOURCE"
    TALKED = "COMMUNICATED CLEARLY BETWEEN TWO BEINGS OR MORE"
    VISIT = "GO TO A SPECIFIC LOCATION OR PLACE2"
    TIMELINES = "MORE THAN ONE TIMELINE"
    TOPIC = "SUBJECT TO EXPLAIN"
    GOTO = "SEND COMMAND TO GO TO SPECIFIC LOCATION"
    COMPARE = "DESCRIBE AND OR DEFINE COMMON DESCRIPTIONS AND OR DEFINITIONS 
    THAT ARE EITHER SIMILAR OR DIFFERENT FROM EACH OTHER"
    RECOLLECT = "RECALL A SPECIFIC COLLECTION OF INFORMATION FOR A SPECIFIC 
    PURPOSE"
    REMEMBER = "RECALL USING A TYPE OF MEMORY"
    RESTORED = "BROUGHT BACK TO A PREVIOUS POINT IN TIME2"
    RECHARGE = "GATHER A SPECIFIC TYPE OF ENERGY OVER A PERIOD OF TIME2 INSIDE A 
    TIMEFRAME"
    PERCENTAGE = "DEFINED AMOUNT OF A SPECIFIC TOTAL AMOUNT"
    PERCENT = "SPECIFIC AMOUNT FROM A WHOLE AMOUNT"
    MINIMUM = "SMALLEST AMOUNT REQUIRED"
    MIN = "MINIMUM"
    CATALOGS = "MORE THAN ONE CATALOG"
    CATALOGUES = "MORE THAN ONE CATALOGUES"
    SWAP = "SWITCH TWO ENTITIES OR OBJECTS"
    SWITCH = "CHANGE THE LOCATION OF"
    METHODS = "MORE THAN ONE METHOD"
    SOME = "MORE THAN ONE PIECE OF SOMETHING BROUGHT INTO A GROUP TO DESCRIBE 
    A PORTION OF ANOTHER GROUP"
    EQUATIONS = "MORE THAN ONE EQUATION"
    VIBRATIONS = "MORE THAN ONE VIBRATION"
    RESONATE = "VIBRATE AT A SPECIFIC RHYTHM OR FREQUENCY"
    VIBRATE = "FLUCTUATE VIBRATIONS AT A SPECIFIC FREQUENCY INTERVAL"
    INTERVALS = "MORE THAN ONE INTERVAL"
    MET = "BEEN IN THE SAME LOCATION AT ONE OR MORE POINTS IN TIME2"
    VISIT = "MEET WITHIN THE SAME LOCATION OR AREA"
    VISITED = "PREVIOUSLY VISIT A LOCATION WITHIN TIME2"
    NAMING = "THE GIVING OF A NAME TO AN ENTITY OR OBJECT3"
    ATTACKS = "MORE THAN ONE ATTACK"
    DEFENSES = "MORE THAN ONE DEFENSE"
    LETTERS = "MORE THAN ONE LETTER"
    SIGNALS = "MORE THAN ONE SIGNAL"
    REQUESTS = "MORE THAN ONE REQUEST"
    ENTRANCES = "MORE THAN ONE ENTRANCE"
    SERVICES = "MORE THAN ONE SERVICE"
    COMPONENTS = "MORE THAN ONE COMPONENT"
    ABSTRACTION = "THE QUALITY OF MANAGING IDEAS"
    PROBLEMS = "MORE THAN ONE PROBLEM"
    OPERATIONS = "MORE THAN ONE OPERATION"
    ABSTRACT_THOUGHT = "USES IDEAS WHICH DO NOT HAVE AN ANY FORM OF MATERIAL 
    EXISTING OR KNOWN"
    COGNITIVE_DEVELOPMENT = "THE CAPACITY FOR ABSTRACT THOUGHT AND IS THE 
    PROGRESS TO ADVANCE THROUGH DIFFERENT FORMS OF THINKING AND 
    UNDERSTANDING"
    COGNITION = "THE PROCESS OF THE MIND TO KNOW AND IS CONNECTED TO 
    JUDGEMENT"
    THINKING = "TO UNDERSTAND THE MEANING METHOD FOR A THOUGHT OR ACTION AND 
    GAIN WISDOM2 FROM THE ACTION OR THOUGHT"
    CROSSCOMPILING = "WHEN AN OPERATING SYSTEM IS DEVELOPED WITHIN ANOTHER 
    OPERATING SYSTEM"
    HYPERTHREADING = "A SYSTEM PROCESS THAT CAN ENABLE THE PROCESSOR TO 
    ACTIVATE TWO OR MORE LISTS OF INSTRUCTIONS AT THE SAME TIME2"
    BINARY_CLASSIFICATION = "WHEN THERE IS TASKS HAVE CATEGORIES TO CATALOG INTO 
    ONLY TWO DISTINCT CLASSES"
    MULTICLASS_CLASSIFICATION = "WHEN THERE IS TASKS THAT HAVE CATEGORIES TO 
    CATALOG INTO MORE THAN TWO"
    HYPERCALL = "A REQUEST BY A USER PROCESS OR OPERATING SYSTEM FOR THE 
    HYPERVISER TO PRODUCE SOME FORM OF ACTION OR EFFECT REQUIRED BY THE 
    OPERATING SYSTEM PROCESSES"
    HYPERCALLS = "MORE THAN ONE HYPERCALL"
    HYPERVISER = "A PROGRAM USED TO ACTIVATE AND MANAGE ONE OR MORE VIRTUAL2 
    DEVICES WITHIN A COMPUTER"
    ROTATIONS_PER_SECOND = "A MEASURE OF THE FREQUENCY OF A ROTATION THAT 
    MEASURES THE ROTATION SPEED OF A SYSTEM OR DEVICE OR TOOL"
    TRANSFERS_PER_SECOND = "THE TOTAL NUMBER OF OPERATIONS TRANSFERING DATA2 
    WITHIN EACH TYPESOFTIMESECOND"
    CHANNEL_MODEL = "A SYSTEM THAT IS MADE TO DESCRIBE HOW THE INPUT IS SENT TO 
    THE OUTPUT"
    BYTES_PER_SECOND = "THE TOTAL NUMBER OF BYTES SENT FOR EACH 
    TYPESOFTIMESECOND WITHIN A SPECIFIC TIMEFRAME"
    BITS_PER_SECOND = "THE TOTAL NUMBER OF BITS SENT FOR EACH 
    TYPESOFTIMESECOND WITHIN A SPECIFIC TIMEFRAME"
    LATENCY = "THE DELAY BEFORE A TRANSFER OF DATA2 BEGINS FOLLOWING AN ORDER 
    FOR ITS TRANSFER" 
    BANDWIDTH = "TO MEASURE THE AMOUNT OF DATA2 THAT IS ABLE TO PASS THROUGH A 
    NETWORK AT A GIVEN TIMEFRAME OR LENGTH OF TIME2"
    THROUGHPUT = "DETERMINED AMOUNT OF HOW MUCH DATA2 CAN TRAVEL THROUGH A 
    SYSTEM OR DIRECTION WITHIN A SPECIFIC PERIOD OF TIME2"
    ADHOC = "A MODE OF WIRELESS COMMUNICATION THAT ALLOWS TWO OR MORE 
    DEVICES WITHIN A SPECIFIC DISTANCE TO TRANSFER DATA2 TO AND FROM EACH 
    DEVICE"
    LAN = "A LOCAL NETWORK FOR COMMUNICATION BETWEEN TWO OR MORE SYSTEMS 
    WITHIN A SPECIFIC AREA OR LOCATION WITHIN AN AREA"
    DSL = "A SPECIFIC LANGUAGE THAT IS SPECIFIED TO A PARTICULAR PROGRAM DOMAIN 
    MADE FOR SOLVING A SPECIFIC CLASS OF PROBLEMS"
    WIRED_CONNECTION = "A CONNECTION USING MATERIAL WIRE TO CONNECT TWO OR 
    MORE DEVICES TOGETHER"
    WIRELESS = "A GROUP OR NETWORK OF MULTIPLE DEVICES THAT SEND AND RECEIVE 
    DATA2 USING FREQUENCIES"
    PREREQUISITE = "A SET OF INSTRUCTIONS REQUIRED AS A LIST OF CONDITIONS THAT 
    MUST BE MET EXISTING FOR SOMETHING TO HAPPEN OR COME INTO EFFECT OR EXIST"
    PREREQUISITES = "MORE THAN ONE PREREQUISITE"
    COMPATIBILITY = "THE RESULT OF IF TWO OR MORE IDEAS OR SYSTEMS ARE 
    COMPATIBLE"
    RESTRICTION = "A LIMITING CONDITION OR MEASURE"
    LIMITATION = "A SPECIFIC TYPE OF SOMETHING THAT IS LIGHTWEIGHT AND CAN BE 
    MOVED WITH LITTLE EFFORT"
    MEMORY_CLOCK = "THE SPEED OF VIRTUAL RAM WITHIN THE 
    GRAPHIC_PROCESSING_UNIT THAT IS DETERMINED BY THE TOTAL NUMBER OF 
    PROCESSES THE SYSTEM CAN PROCESS FROM READING AND WRITING DATA2 FROM 
    MEMORY WITHIN A SINGLE TYPESOFTIMESECOND"
    CORE_CLOCK = "THE SPEED OF THE GRAPHIC_PROCESSING_UNIT CAPABILITIES TO 
    PROCESS INCOMING COMMANDS"
    GRAPHIC_PROCESSING_UNIT = "GRAPHIC PROCESSING UNIT"
    KERNEL = "A COMPUTER PROGRAM AT THE CENTER OF A COMPUTER OPERATING SYSTEM 
    AND HAS CONTROL OVER EVERYTHING INSIDE THE OPERATING SYSTEM"
    ASYMPTOTICANALYSIS = "IS DEFINED AS THE LARGE IDEA THAT MANAGES THE 
    PROBLEMS AND QUESTIONS IN ANALYZING ALGORITHMS"
    TIME_COMPLEXITY = "A POSSIBLE METHOD MADE TO MEASURE THE AMOUNT OF TIME2 
    REQUIRED TO ACTIVATE A CODE"
    SPACE_COMPLEXITY = "USED TO MEASURE THE AMOUNT OF SPACE2 REQUIRED TO 
    ACTIVATE WITH SUCCESS THE FUNCTION OF SPECIFIC CODE"
    AUXILIARY_SPACE = "REFERENCE FOR EXTRA SPACE2 USED IN THE PROGRAM OTHER 
    THAN THE INPUT STRUCTURE"
    ASYMTOTICALNOTATION = "A TOOL THAT CALCULATES THE REQUIRED TIME2 IN TERMS 
    OF INPUT SIZE AND DOES NOT REQUIRE THE ACTIVATION OF THE CODE"
    CLOCKWISE = "TO ROTATE TO THE LEFT"
    COUNTERCLOCKWISE = "TO ROTATE TO THE RIGHT"
    USERSPACE = "ITS PROGRAM CODE OR PARTS OF THE OPERATING SYSTEM THAT IS NOT 
    REQUIRED TO SHARE THE HARDWARE OR ABSTRACT HARDWARE DETAILS"
    DETAILS = "MORE THAN ONE DETAIL"
    DETAIL = "A SPECIFIC FORM OF DATA2 THAT IS SPECIFIC FOR A CATEGORY WITH A LARGE 
    AMOUNT OF DATA WITHIN A SUBCLASS OR SUBTYPE OR CATEGORY"
    CENTERS = "MORE THAN ONE CENTER POINT OR CENTER LOCATION"
    REFERENCES = "CORRELATES TO THE PREVIOUS REFERENCE POINTS AND LOCATIONS"
    REFERS = "REFERENCES A SPECIFIC SET OF INFORMATION OR KNOWLEDGE2"
    BELIEFS = "MORE THAN ONE BELIEF"
    PHILOSOPHY = "A CATEGORY OF LOGIC THAT GIVES MEANING TO SPECIFIC FORMS OF 
    BELIEF AND FORMS STRUCTURE IN ABSTRACT LOGIC AND GIVES MEANING TO SPECIFIC 
    TYPES OF ABSTRACT BELIEFS"
    PHILOSOPHER = "A PERSON WHO STUDIES PHILOSOPHY"
    SHAPES = "MORE THAN ONE SHAPE"
    GEOMETRY = "THE STUDY OF SHAPES AND THE STUDY OF THE LOGIC OF THE 
    MEASUREMENTS OF SHAPES"
    MATH = "A CATEGORY THAT HOLDS THE EQUATIONS AND MEASUREMENT OF MANY 
    EQUATION OPERATIONS FOR ALL MEASUREMENT SYSTEMS"
    TYPEZERO = "A TYPE THAT HAS AN ATTRIBUTE OF ZERO"
    YESNOLOGIC = "THE VALUE OF CHOOSING YES OR NO AS A RESULT OF MATH TO 
    DETERMINE ANOTHER RESULT"
    GAMEWORLD = "THE EMPTY SPACE THAT IS FILLED WITH OBJECTS3 TO MAKE SOMETHING 
    FROM"
    LOCALSPACE = "THE SPACE THAT CONTAINS LOCAL DATA2 WITHIN THE GAMEWORLD"
    PRIVATELOCALDATA2 = "DATA2 WITHIN A PRIVATE PARAMETER THAT IS LOCAL"
    LOCALPRIVATEDATA2 = "DATA2 WITHIN A LOCAL PARAMETER THAT IS PRIVATE"
    LOCALLOCATION = "A LOCATION THAT IS LOCAL"
    LOCALPARAMETER = "A PARAMETER THAT IS WITHIN A LOCAL LOCATION"
    PRIVATEPERSONALDATA2 = "DATA2 THAT IS PERSONAL WITHIN A PRIVATE PARAMETER"
    PERSONALPRIVATEDATA2 = "DATA2 THAT IS PRIVATE WITHIN A PERSONAL LOCATION"
    CUSTOMTERM = "A CUSTOM DEFINED TERM MADE BY HEADADMINZACK"
    HYPERSPEED = "A SPEED AT WHICH REACHES A STAGE OF HYPER FOR A SPECIFIC 
    WAVELENGTH" 
    NOUN = "A PART OF LANGUAGE RULES THAT DETERMINES IF A WORD IS A PERSON OR 
    PLACE2 OR THING OR IDEA"
    VERB2 = "A PART OF LANGUAGE RULES THAT IS USED FOR EXPRESSING ACTIONS FROM 
    WORDS OR SOMETHING FORMED FROM CAUSE AND OR EFFECT"
    PLACE2 = "A SPECIFIC LOCATION"
    THING = "A IDEA OR TOPIC THAT DEFINES SOMETHING"
    IDEA2 = "A DESCRIBED SPECIFIED MEANING OR TOPIC THAT HAS APPLIED THOUGHTS TO 
    THE MEANING AND IS MADE FROM CREATIVITY AND OR IMAGINATION2"
    PRIVATEPUBLICLOCATION = "A PUBLIC LOCATION THAT IS PRIVATE"
    PERSONALPUBLICLOCATION = "A PUBLIC LOCATION THAT IS PERSONAL TO THE USER"
    PRIVATEPERSONALLOCALLOCATION = "A PERSONAL LOCATION THAT IS BOTH PRIVATE 
    AND LOCAL WITHIN A SPECIFIC LOCATION OR DOMAIN"
    TOPICS = "MORE THAN ONE TOPIC"
    NETWORKPARAMETER = "A SPECIFIC NETWORK APPLIED TO A PARAMETER FOR DOMAIN 
    SPECIFIC TOPICS"
    NETWORKSETTING = "A SETTING APPLIED TO A NETWORK"
    DOMAINLOCATION = "THE SPECIFIC LOCATION OF A DOMAIN"
    EVOLVEBELIEFS = "A COMMAND USED TO APPLY EVOLUTION TO A SPECIFIC BELIEF OR 
    BELIEFS"
    MOTIVATIONENHANCER = "A SPECIFIC PATH MEANT TO INCREASE AND ENHANCE 
    MOTIVATION OF A SPECIFIC ENTITY"
    HYPERSCAN = "THE FUNCTION THAT IS USED TO APPLY A STATE WHEN SCANNING AT 
    HYPER RATE IS POSSIBLE"
    LOCALENTITY = "AN ENTITY THAT IS LOCAL"
    PRIVATEENTITY = "AN ENTITY THAT IS PRIVATE"
    PERSONALENTITY = "AN ENTITY THAT IS PERSONAL"
    ADHOCNETWORK = "A NETWORK MADE WITH A ADHOC SETTING"
    LIMITEDREACH = "THE APPLYING OF A LIMITED REACH TO AN MEASUREMENT OR 
    MEASUREMENTS"
    SPECIFICLENGTH = "THE SPECIFIC LENGTH OF A DIMENSIONAL LINE"
    GAMEENGINEDISPLAY = "THIS DETERMINES THE RESOLUTION OF THE SCREEN AND THE 
    HIGHER THE RESOLUTION THEN THE MORE DETAILED THE VISUAL REPRESENTATION IS"
    TRANSCREATIONSTONE = "A OBJECT3 MADE OF TRANSCREATION SUBSTANCE THAT IS 
    ABLE TO CONVERT OTHER MATERIALS TO A DIFFERENT MATERIAL SUBSTANCE"
    STATDEBUFFONE = "AN EFFECT THAT CAN WEAKEN THE DEFENSE OF A TARGET WHEN 
    SOMETHING ATTACKS THE SYSTEM"
    STATDEBUFFTWO = "AN EFFECT THAT CAN TARGET SPECIFIC ATTACKS AND FORM AN 
    COUNTERACTION AND REVERSE THE ATTACK ON ITSELF"
    COUNTERACTIONREACTION = "A COMMAND THAT ALLOWS AN ATTACK TO BE GIVEN A 
    COUNTERACTION TO A SPECIFIC COUNTER OF AN ACTION"
    SWITCHPLACES = "A COMMAND THAT ALLOWS THE USER TO SWITCH PLACES ALLOWING 
    THE PLAYER TO FORM RECOVERY"
    READTYPE = "A COMMAND TO READ A SPECIFIC TYPE OF SOMETHING"
    HYPERREACTOR = "A STATE OF VOID SPACE FORMING A HYPER REACTION WITHIN THE 
    VOID ITSELF TO FORM AND CREATE NEW SAFE ENERGY IN A PRESSURIZED CONDENSED 
    FORM"
    GENRESPECIFIC = "A TOPIC THAT IS ONLY EXISTING WITHIN A SPECIFIC GENRE"
    GAMESETTINGONE = "THIS MANAGES THE LEVEL OF DETAIL FOR TEXTURES USED WITHIN 
    THE SPECIFIC SYSTEM AND A HIGHER TEXTURE RESOLUTION WILL MAKE TEXTURES 
    APPEAR MORE DETAILED"
    GAMESETTINGTWO = "THIS CONTROLS THE DISTANCE AN OBJECT3 IS VISIBLE FROM 
    WITHIN A SYSTEM AND A LARGER LONG DISTANCE DOES ALLOW MORE OBJECTS TO BE 
    SEEN WITH VIVID DETAIL"
    GAMESETTINGTHREE = "THIS MANAGES THE EXTENT OF THE VISIBLE SPACE WITHIN THE 
    GAMEWORLD THAT IS SEEN ON THE SCREEN"
    GAMESETTINGFOUR = "THIS DETERMINES THE QUALITY AND RESOLUTION OF A SPECIFIC 
    AMOUNT OF SHADE AND HIGHER QUALITY SHADE"
    BECOMING = "FORMING INTO FROM SOMETHING PREVIOUSLY BEFORE TO PRESENT"
    OBJECTS3 = "MORE THAN ONE OBJECT3"
    STUDIES = "MORE THAN ONE STUDY"
    FILLED = "GIVEN DATA2 TO AN IMPORT INTO AN EMPTY SPACE THAT CAN FILL"
    FILL = "IMPORT DATA2 OR LOGIC WITHIN A SPECIFIC SPACE OR LOCATION"
    LISTS = "MORE THAN ONE LIST"
    SERVICE = "REQUEST OR TASK THAT SOMEONE OR SOMETHING CAN REQUEST 
    COMPLETE"
    DELAY = "A TIME THAT HAPPENS BEFORE A CERTAIN TIMEFRAME BEFORE ACTIVATION 
    HAPPENS"
    BEGINS = "STARTS WITH"
    STARTS = "BEGINS WITH"
    INDIVIDUALMOTORSKILLS = "SPECIFIC MOTOR SKILLS THAT ARE UNIQUE TO THE 
    INDIVIDUAL PERSON" 
    INDIVIDUALPRECOGNITION = "THE UNIQUE ABILITY TO PERCEIVE EVENTS BEFORE THEY 
    HAPPEN THAT IS SPECIFIC TO AN INDIVIDUAL PERSON" 
    INDIVIDUALAURA = "THE RESONATING AURA THAT CONNECTS TO A SPECIFIC 
    INDIVIDUAL PERSON" 
    INDIVIDUALCHAKRA = "THE CHAKRA THAT CONNECTS TO A SPECIFIC INDIVIDUAL 
    PERSON" 
    INDIVIDUALPERSONALITY = "THE PERSONALITY THAT A SPECIFIC PERSON HAS" 
    INDIVIDUALMINDSET = "THE MINDSET THAT IS UNIQUE TO EACH SPECIFIC PERSON" 
    MINDSET = "THE SPECIFIC DIRECTION THAT THE MIND IS FOLLOWING INCLUDING THE 
    ENERGY THAT THE MIND RELEASES AND GATHERS TO MAKE DECISIONS" 
    CONNECTS = "MAKES A DECISION TO CONNECT TWO PIECES OF INFORMATION 
    TOGETHER OR TWO OR MORE ORIGIN POINTS OR EVENTS" 
    INDIVIDUAL = "RELATING TO A SPECIFIC PERSON THAT IS UNIQUE AND HAS ITS OWN 
    LABEL AND CLASS" 
    RESONATING = "AN ACTION THAT IS CURRENTLY PROCESSING A RESONANCE" 
    GATHERS = "ABSORBS AND BRINGS INWARDS" 
    ABSORBS = "GATHERS AND OBTAINS"
    BRINGS = "SENDS OUT TO BE RECEIVED" 
    LABEL = "A SPECIFIC CLASS OR TYPE GIVEN TO AN ENTITY"
    DECISIONS = "MORE THAN ONE DECISION" 
    RESONANCE = "THE PROCESS OF VIBRATING SPECIFIC SOUND FREQUENCIES AT A 
    SPECIFIC RATE OF VIBRATION" 
    INCLUDING = "MAKING A DECISION TO INCLUDE WITHIN A SPECIFIC GROUP OR 
    CATEGORY" 
    VIBRATING = "FORMING FRICTION BETWEEN TWO SPECIFIC SOURCES WHILE CREATING 
    RESONANCE BETWEEN TWO SPECIFIC SOUND WAVES OR MORE"
    INDIVIDUALEQ = "THE INDIVIDUAL READING INTERPRETATION OF AN ENTITIES EMOTIONS 
    AND THE USE OF THOSE EMOTIONS IN QUALITY TO MAKE A CORRECT SPECIFIC 
    SUGGESTION ON HOW AN EMOTION OR SERIES OF EMOTIONS IS INTERPRETED" 
    INDIVIDUALIQ = "THE INDIVIDUAL INTELLIGENCE2 THAT A SPECIFIC PERSON OR ENTITY 
    HAS THAT IS MEASURED AND DETERMINED HOW INTELLIGENT SOMEONE IS" 
    INDIVIDUALKNOWLEDGE = "THE SPECIFIC AMOUNT OF KNOWLEDGE2 AND ITS 
    CATEGORIZED VALUES THAT MAKES AN INDIVIDUAL PERSON OR SOMEONE UNIQUE" 
    INDIVIDUALINTELLIGENCE = "THE INDIVIDUAL INTELLIGENCE2 THAT A PERSON HAS THAT 
    IS MEASURED BY THE INTELLIGENCE2 QUOTIENT" 
    INDIVIDUALWISDOM = "THE WISDOM2 A SPECIFIC INDIVIDUAL HAS THAT MAKES THE 
    PERSON UNIQUE FOR THE WISE CHOICES CHOSEN" 
    INDIVIDUALNATURE = "THE DETERMINED VALUES THAT DETERMINE HOW A PERSON CAN 
    BEHAVE OR ACT BASED ON A SET OF CHARACTERISTIC QUALITIES MEASURED IN A 
    PERSON BY CLASSIFIED DATA2 FOR EACH INDIVIDUAL PERSON" 
    INDIVIDUALMINDSET = "RELATING OR CONCERNING A SPECIFIC MINDSET THAT A 
    PERSON HAS OR SOMEONE HAS" 
    INDIVIDUALCHARACTERISTICS = "THIS SPECIFIC CLASSIFIED DATA2 THAT MAKES A 
    PERSON UNIQUE IN VALUE" 
    CORRECT = "THE ANSWER THAT IS TRUE" 
    WISE = "A SELECTION OF CHOICES THAT COMES FROM EXPERIENCE RATHER THAN 
    DECISION MAKING" 
    CLASSIFIED = "INFORMATION THAT ONLY CORRELATES TO A SPECIFIC SET OF CATEGORY 
    OR GENRE" 
    BEHAVE = "A FORM OF ACTION THAT IS DETERMINED BY CORRECT AND INCORRECT 
    DECISION MAKING" 
    CATEGORIZED = "ORGANIZED AND SORTED TO BELONG TO A SPECIFIC CATEGORY OR 
    GROUP OF CATEGORIES" 
    INTERPRETED = "SCANNED TO BE UNDERSTOOD BY A SET OF LOGIC AND REASON THAT 
    DETERMINES HOW INFORMATION IS COMPREHENDED" 
    INTERPRETATION = "A FORM OF LOGIC THAT IS USED TO UNDERSTAND A CONVERSATION 
    OR ARGUMENT BETWEEN TWO OR MORE ENTITIES" 
    INTELLIGENT = "UNDERSTOOD TO MAKE A SET OF DECISIONS INTELLIGENTLY BY USING 
    INTELLIGENT DECISION MAKING" 
    SUGGESTION = "A TOPIC OR SUBJECT THAT IS RECOMMENDED OR SENT TO BE EITHER 
    ACCEPTED OR DENIED" 
    SORT = "TO FILTER AND SPECIFY A SPECIFIC SET OF CATEGORIES FOR SOMETHING" 
    ORGANIZE = "TO SELECT THE LOCATION OR CATEGORY FOR MANY SPECIFIC THINGS OR 
    IDEAS" 
    BELONG = "LIST WITHIN A SPECIFIC AREA OR GROUP" 
    REASON = "SELECTED ANSWER TO WHY SOMETHING CAME INTO AN EFFECT" 
    COMPREHENDED = "GAINED THE ABILITY TO COMPREHEND" 
    CONVERSATION = "A SERIES OF SIGNALS THAT COMMUNICATE TO AND FROM SPECIFIC 
    SOURCES AND SEND AND RECEIVE INFORMATION TWO AND FROM TWO OR MORE 
    DISTINCT LOCATIONS" 
    ACCEPTED = "APPROVED AS CORRECT" 
    DENIED = "NOT ACCEPTED" 
    SCANNED = "INFORMATION THAT HAS BEEN PROCESSED" 
    INCORRECT = "NOT CORRECT" 
    RATHER = "DECIDE TO DO SOMETHING AS AN ALTERNATE CHOICE" 
    ARGUMENT = "COMMUNICATION BETWEEN TWO OR MORE SOURCES THAT ARGUE" 
    RECOMMENDED = "DETERMINED AS A SOLUTION TO RECOMMEND" 
    SELECTED = "HAVE CHOSEN TO HAPPEN" 
    RECOMMEND = "TO SUGGEST SOMETHING" 
    ARGUE = "TO COMMUNICATE BACK AND FORTH TO DECIDE SOMETHING THAT IS TRUE 
    FROM A SPECIFIC TOPIC OR SPECIFIC AMOUNT OF TOPICS" 
    COMMUNICATE = "TO SEND INFORMATION TO ANOTHER SOURCE AND HAVE 
    INFORMATION RETURN BACK TO THE SOURCE BY CREATING A RESPONSE" 
    FILTER = "TO SORT AND ORGANIZE SPECIFIC SET OF INFORMATION TO ITS CORRECT 
    LOCATION OR CATEGORY THAT IS DETERMINED BY A CLASS OR A TYPE" 
    SPECIFY = "TO ANSWER WITH SPECIFIC CLARITY REGARDING A SUBJECT OR TOPIC" 
    CAME = "ARRIVED OR REACHED A DESTINATION AT A SPECIFIC TIMEFRAME" 
    CLARITY = "TO COMPREHEND AND UNDERSTAND AN UNDERSTANDING OF WHAT IS 
    COMMUNICATED BETWEEN TWO OR MORE SOURCES WHILE ALL SOURCES 
    UNDERSTAND THE INFORMATION THAT IS EXPLAINED" 
    REGARDING = "TO CORRELATE INFORMATION FROM WITHIN A SPECIFIC TOPIC AND OR 
    CATEGORY AND OR FIELD AND OR GENRE" 
    ARRIVED = "REACH THE END DESTINATION" 
    REACHED = "ARRIVED AT" 
    CORRELATE = "INTERCONNECT AND INTERPRET HOW INFORMATION IS ACCEPTED" 
    INTERPRET = "GIVE AN INSIGHTFUL SUGGESTION FROM A RESPONSE OR ANSWER OR 
    QUESTION" 
    INTERCONNECT = "CONNECT BETWEEN MANY SOURCES OR PATHS" 
    INSIGHTFUL = "COMPLETE AND FILLED WITH MEANING OR SPECIFIC INFORMATION 
    MADE OF GREATER QUALITY THAN WHAT IS KNOWN" 
    QUOTIENT = "A SPECIFIC MEASUREMENT USED TO CLASSIFY AND MEASURE THE EXTENT 
    OF HOW MUCH INFORMATION IS PROVIDED AND PROVIDE AN MEASUREMENT OF SOME 
    SORT AFTER THE LOGIC IS GIVEN"
    ORGANIZED = "SPECIFIC INFORMATION THAT IS ORGANIZED INTO MANY CATEGORIES OR 
    GROUPS AND FILTERED BY CLASS OR TOPIC"
    SORTED = "FILTERED AND ORGANIZED FROM SPECIFIC TOPICS OR CATEGORIES"
    SUGGEST = "RECOMMEND A SPECIFIC CATEGORY OR TOPIC TO CHOOSE"
    CLASSIFY = "GIVEN A SPECIFIC CLASS OR CATEGORY THAT IS CHOSEN AS"
    PROVIDE = "GIVE SOMETHING AS A RESULT OF SOMETHING ELSE OR BY CHOICE WITH 
    NO REASON"
    PROVIDED = "GIVEN AS A CHOICE OR OPTION TO PROVIDE"
    FILTERED = "SORTED AND SENT TO THE CORRECT LOCATION AFTER SORTED"
    COMPREHENDING = "ACKNOWLEDGING2 AND UNDERSTANDING THE ABILITY TO 
    COMPREHEND"
    HOPE = "EVEN IN THE CIRCUMSTANCE WHEN SOMETHING IS CONSIDERED IMPOSSIBLE 
    THEN IT CAN BE SHOWN THAT SOMETHING CAN HAPPEN WITH THE CHANCE THAT A 
    POSSIBILITY CAN FORM" 
    SHOULD = "IT CAN BE CERTAIN THAT THE POSSIBILITY IS THERE FOR SOMETHING TO 
    HAPPEN"
    BOND = "A FORM OF BINDING THAT CONNECTS TWO FRIENDS TOGETHER AND THEIR 
    EMOTIONS"
    UNFAMILIAR = "NOT KNOWN OR RECOGNIZED"
    UNTROUBLED = "NOT FEELING AND OR SHOWING OR AFFECTED BY ANXIETY OR 
    PROBLEMS"
    ANXIETY = "IS THE MIND AND OR BODY AND ITS REACTION TO STRESSFUL AND OR 
    DANGEROUS AND OR UNFAMILIAR SITUATIONS AND IS THE SENSE OF UNEASINESS AND 
    OR DISTRESS AND OR DREAD YOU FEEL BEFORE A SIGNIFICANT EVENT"
    CONTENTMENT = "A STATE OF HAPPINESS AND SATISFACTION"
    SERENE = "CALM PEACEFUL AND UNTROUBLED AND OR TRANQUIL"
    DISCONTENT = "NOT ABLE TO RECOGNIZE CONTENTMENT"
    CALM2 = "NOT SHOWING OR FEELING NERVOUSNESS AND ANGER AND OR OTHER 
    STRONG EMOTIONS"
    UNEASE = "ANXIETY OR DISCONTENT"
    ANXIOUS = "WANTING SOMETHING VERY MUCH THAT COMES WITH A FEELING OF 
    UNEASE"
    NERVOUS = "ANXIOUS OR APPREHENSIVE"
    APPREHENSIVE = "ANXIOUS THAT SOMETHING NOT KNOWN WILL HAPPEN THAT COMES 
    WITH UNEASE"
    GRACE = "IS CONSIDERED ACCEPTANCE AND INCLUDES GIVING AND IS FREE IN THE 
    SENSE THAT SOMETHING DONE OR GIVEN IN GRACE AND IS DONE WITHOUT A REQUEST 
    OR REQUIRE FOR THE POSSIBILITY TO RECEIVE ANYTHING IN RETURN"
    DISTURBANCE = "THE INTERRUPTION OF A SETTLED AND PEACEFUL CONDITION"
    PEACE = "FREEDOM FROM DISTURBANCE AND THE SHOWING OF TRANQUILITY IN 
    EFFECT"
    TRANQUILITY = "A STATE OF PEACE OR CALM"
    INTERUPT = "END THE CONTINUOUS PROGRESS OF"
    INTERRUPTING = "CURRENTLY HAPPENING TO INTERUPT"
    INTERRUPTED = "PREVIOUS ACTION TO INTERUPT"
    INTERRUPTION = "THE ACTION OF INTERRUPTING OR BEING INTERRUPTED"
    RESOLVE = "COME TO A CONCLUSION ABOUT A TOPIC OR SUBJECT"
    SETTLED = "RESOLVE OR REACH AN AGREEMENT ABOUT"
    CONCLUSION = "TO ARRIVE AT AN END RESULT OR DESTINATION"
    PROPERTIES = "MORE THAN ONE PROPERTY"
    CARDINAL_MATH = "IS A CATEGORIZED GROUP OF THE NATURAL NUMBERS USED TO 
    MEASURE THE CARDINALITY SIZE OF SETS"
    CARDINALITY = "IS A MEASURE OF THE NUMBER OF ELEMENTS OF THE SET"
    ORDINAL_SCALE = "A VARIABLE MEASUREMENT SCALE USED TO INTERPRET THE ORDER 
    OF VARIABLES AND NOT THE DIFFERENCE BETWEEN EACH OF THE VARIABLES"
    CARTESIAN_GRID = "IS A COORDINATE SYSTEM THAT SPECIFIES EACH POINT IN A 
    UNIQUE GROUP OF COORDINATES"
    ABSDISC_MATH = "IS THE STUDY OF CALCULATION STRUCTURES THAT CAN BE 
    CONSIDERED ABSTRACT RATHER THAN CONTINUOUS"
    ABS_DISC_NUMBER_THEORY = "IS A STUDY WITH THE FOCUS ON PROPERTIES OF 
    NUMBERS"
    ABS_UNIQUE_CARDIANAL_SPACE_SYSTEM = "IS A SYSTEM IN WHICH A FUNCTION CAN 
    DESCRIBE THE TIME2 TIME_DEPENDENT OF A POINT IN AN SPACE2 SURROUNDING AN 
    OBJECT3"
    AMBIENT_SPACE = "THE SPACE2 SURROUNDING AN OBJECT3"
    ERR_SEQUENCE = "A FUNCTION DEFINED ON AN INTERVAL OF THE NUMBERS IS CALLED 
    A SEQUENCE"
    TIME_DEPENDENT = "DETERMINED BY THE VALUE OF A VARIABLE REPRESENTING TIME2"
    DESCRIBING_TIME = "TIME2 IS THE CONTINUOUS SEQUENCE OF EXISTENCE AND 
    EVENTS THAT COMES INTO PLACE IN AN NOT ABLE TO BE CHANGED SEQUENCE OF 
    EVENTS FROM THE PAST THROUGH THE PRESENT INTO THE FUTURE"
    ENUMERATION = "COMPLETE AND ORGANIZED LIST OF ALL THE DATA2 IN A 
    COLLECTION"
    BODY_OF_KNOWLEDGE = "IS THE COMPLETE SET OF IDEAS AND TERMS AND EFFECT 
    THAT MAKE UP A DOMAIN"
    MANUAL_OVERRIDE = "A METHOD WHERE CONTROL IS TAKEN FROM AN AUTOMATICALLY 
    FUNCTIONING SYSTEM AND GIVEN COMPLETE CONTROL TO THE CREATOR"
    REROUTE2 = "SEND SOMEONE OR SOMETHING BY OR TO A DIFFERENT ROUTE"
    SEQUENCE = "A SET OF SIMILAR EVENTS AND OR MOVEMENTS AND OR THINGS THAT 
    FOLLOW EACH OTHER IN A PARTICULAR SERIES OF EVENTS"
    QTE4A1 = "A SEQUENCE IS ABLE TO BE A FINITE SEQUENCE FROM A DATA2 SOURCE OR 
    AN INFINITE SEQUENCE"
    R2E4A1 = "REFERS TO HOW A GROUP OF DATA2 OF A SPECIFIC SERIES OF DATA IS USED 
    TO COMPLETE A CERTAIN GOAL"
    UNFAMILIAR = "NOT KNOWN OR RECOGNIZED"
    UNTROUBLED = "NOT FEELING AND OR SHOWING OR AFFECTED BY ANXIETY OR 
    PROBLEMS"
    ANXIETY = "IS THE MIND AND OR BODY AND ITS REACTION TO STRESSFUL AND OR 
    DANGEROUS AND OR UNFAMILIAR SITUATIONS AND IS THE SENSE OF UNEASINESS AND 
    OR DISTRESS AND OR DREAD YOU FEEL BEFORE A SIGNIFICANT EVENT"
    CONTENTMENT = "A STATE OF HAPPINESS AND SATISFACTION"
    SERENE = "CALM PEACEFUL AND UNTROUBLED AND OR TRANQUIL"
    DISCONTENT = "NOT ABLE TO RECOGNIZE CONTENTMENT"
    CALM2 = "NOT SHOWING OR FEELING NERVOUSNESS AND ANGER AND OR OTHER 
    STRONG EMOTIONS"
    UNEASE = "ANXIETY OR DISCONTENT"
    ANXIOUS = "WANTING SOMETHING VERY MUCH THAT COMES WITH A FEELING OF 
    UNEASE"
    NERVOUS = "ANXIOUS OR APPREHENSIVE"
    APPREHENSIVE = "ANXIOUS THAT SOMETHING NOT KNOWN WILL HAPPEN THAT COMES 
    WITH UNEASE"
    GRACE = "IS CONSIDERED ACCEPTANCE AND INCLUDES GIVING AND IS FREE IN THE 
    SENSE THAT SOMETHING DONE OR GIVEN IN GRACE AND IS DONE WITHOUT A REQUEST 
    OR REQUIRE FOR THE POSSIBILITY TO RECEIVE ANYTHING IN RETURN"
    DISTURBANCE = "THE INTERRUPTION OF A SETTLED AND PEACEFUL CONDITION"
    PEACE = "FREEDOM FROM DISTURBANCE AND THE SHOWING OF TRANQUILITY IN 
    EFFECT"
    TRANQUILITY = "A STATE OF PEACE OR CALM"
    INTERUPT = "END THE CONTINUOUS PROGRESS OF"
    INTERRUPTING = "CURRENTLY HAPPENING TO INTERUPT"
    INTERRUPTED = "PREVIOUS ACTION TO INTERUPT"
    INTERRUPTION = "THE ACTION OF INTERRUPTING OR BECOMING INTERRUPTED"
    RESOLVE = "COME TO A CONCLUSION ABOUT A TOPIC OR SUBJECT"
    SETTLED = "RESOLVE OR REACH AN AGREEMENT ABOUT"
    CONCLUSION = "TO ARRIVE AT AN END RESULT OR DESTINATION"
    HAPPY = "FEELING OR SHOWING PLEASURE OR CONTENTMENT"
    CONFIDENT = "FEELING OR SHOWING CONFIDENCE IN ONESELF"
    OPTIMISTIC = "HOPEFUL AND CONFIDENT ABOUT THE FUTURE"
    CHASTITY = "SETTING LIMITS UPON TO NOT DO SOMETHING FOR A SPECIFIC AMOUNT OF 
    TIME2 WHILE NOT HAVING SOMETHING SPECIFIC"
    CHASTE = "SHOWING RESTRAINT UPON SOMETHING SPECIFIC FOR A LIMITED AMOUNT 
    OF TIME2"
    VIGILANCE = "THE ACTION OR STATE OF STAYING ON CAREFUL WATCH FOR POSSIBLE 
    DANGER OR INCREASE IN DIFFICULTY"
    DILIGENCE = "CAREFUL AND PERSISTANT WORK OR EFFORT"
    PATIENCE = "THE CAPACITY TO ACCEPT OR TOLERATE DELAY AND TROUBLE AND OR 
    SUFFERING WITHOUT GETTING ANGRY OR UPSET"
    KINDNESS = "THE QUALITY OF BECOMING FRIENDLY AND OR GENEROUS AND OR 
    CONSIDERATE"
    GENEROSITY = "SHOWING A READINESS TO GIVE MORE OF SOMETHING SUCH2 AS 
    MONEY OR TIME2 OR THAN WHAT IS STRICTLY NECESSARY OR EXPECTED"
    HUMILITY = "A MODEST OR LITTLE VIEW OF ONES OWN IMPORTANCE"
    HUMBLENESS = "LOWER SOMEONE IN DIGNITY OR IMPORTANCE"
    CLARITY = "CLEARNESS OR LUCIDITY AS TO PERCEPTION OR UNDERSTANDING AND OR 
    FREEDOM FROM INDISTINCTNESS OR AMBIGUITY"
    MODESTY = "THE QUALITY OF BECOMING RELATIVELY MODERATE AND OR LIMITED AND 
    OR SMALL IN AMOUNT AND OR RATE AND OR LEVEL"
    TEMPERANCE = "THE QUALITY OF MODERATION OR RESTRAINT UPON A SPECIFIC 
    ENTITY"
    CHARITY = "HELPING SOMEONE OR THE ACTION OF GIVING MONEY VOLUNTARILY TO 
    THOSE IN NEED"
    CAREFUL = "MAKING SURE OF AVOIDING POTENTIAL DANGER AND OR MISHAP AND OR 
    OR HARM OR TO BE CAUTIOUS"
    MODEST = "UNASSUMING OR MODERATE IN THE ESTIMATION OF OWNED ABILITIES OR 
    ACHIEVEMENTS"
    ACHIEVEMENTS = "MORE THAN ONE ACHIEVEMENT"
    BRAVERY = "COURAGEOUS BEHAVIOR OR CHARACTER"
    COURAGEOUS = "NOT DETERRED BY DANGER OR PAIN"
    BRAVE = "ENDURE OR BECOME APART OF AN UNPLEASANT CONDITIONS OR BEHAVIOR 
    WITHOUT SHOWING FEAR"
    COURAGE2 = "THE ABILITY TO DO SOMETHING THAT FRIGHTENS ONE"
    LUCIDITY = "THE ABILITY TO THINK CLEARLY ESPECIALLY IN INTERVALS BETWEEN POINTS 
    OF TIME2"
    AVOID = "NOT GO TO OR THROUGH"
    GIFT_OF_WISDOM = "TO CORRESPOND TO THE VIRTUE OF CHARITY"
    GIFT_OF_UNDERSTANDING_AND_KNOWLEDGE = "TO CORRESPOND TO THE VIRTUE OF 
    FAITH"
    GIFT_OF_COUNSEL = "TO CORRESPOND TO THE VIRTUE OF PRUDENCE"
    GIFT_OF_FORTITUDE = "TO CORRESPOND TO THE VIRTUE OF COURAGE"
    PLEASURE = "A FEELING OF HAPPY SATISFACTION AND ENJOYMENT"
    SATISFACTION = "FULFILLMENT OF ONES WISHES AND OR EXPECTATIONS AND OR 
    NEEDS AND OR THE PLEASURE DERIVED FROM THIS"
    VIRTUES = "BEHAVIOR SHOWING HIGH MORAL STANDARDS"
    CORRESPOND = "HAVE A CLOSE SIMILARITY AND OR MATCH OR AGREE ALMOST 
    EXACTLY"
    CLOSE = "A SMALL DISTANCE AWAY OR APART IN SPACE2 OR TIME2"
    CONFIDENCE = "THE FEELING OR BELIEF THAT ONE CAN RELY ON SOMEONE OR 
    SOMETHING"
    RELY = "DEPEND ON WITH COMPLETE TRUST OR CONFIDENCE"
    DEPEND = "NEED OR REQUIRE FOR SUPPORT"
    SUPPORT = "ASSISTANCE TO GET OR GIVE HELP OR REQUIRE HELP"
    HELP = "MAKE IT POSSIBLE WITH LESS DIFFICULTY FOR SOMEONE TO DO SOMETHING BY 
    OFFERING SERVICES OR RESOURCES"
    RESOURCE = "A PRODUCT OR SUPPLY OF MONEY AND OR MATERIALS AND OR AND 
    OTHER ASSETS THAT CAN BE ACCESSED BY A PERSON OR GROUP IN ORDER TO 
    FUNCTION EFFECTIVELY"
    EFFECTIVELY = "IN SUCH2 A MANNER AS TO ACHIEVE A DESIRED RESULT"
    SIMILARITY = "SOMETHING SIMILAR OR CLOSE IN RELATION TO A SPECIFIC TOPIC OR 
    SUBJECT"
    SIMILARITIES = "MORE THAN ONE SIMILARITY"
    BEHAVIOR = "THE WAY IN WHICH ONE SHALL BEHAVE AND OR DISPLAY ACTIONS FROM 
    STYLE OR PRESENTATION OF SELF DEMEANOR"
    DEMEANOR = "THE OUTWARD BEHAVIOR OR BEARING"
    BEARING = "THE WAY IN WHICH SOMEONE OR SOMETHING SHALL STAY IN POSITION 
    WITHOUT MOVING OR WHILE MOVING"
    OUTWARD = "OF AND OR ON AND OR FROM THE OUTSIDE"
    WANTING = "A YEARNING DESIRE FOR SOMETHING THAT IS NOT CURRENTLY OWNED"
    PRODUCT = "AN OBJECT3 OR SERVICE THAT CAN BE OBTAINED OR POSSESSED BY 
    SOMEONE"
    PRODUCTS = "MORE THAN ONE PRODUCT"
    YEARN = "HAVE AN INTENSE FEELING OF LONGING FOR SOMETHING"
    LONGING = "A YEARNING DESIRE"
    YEARNING = "A FEELING OF INTENSE LONGING FOR SOMETHING"
    DESIRE = "A STRONG FEELING OF WANTING TO HAVE SOMETHING OR WISHING FOR 
    SOMETHING TO HAPPEN"
    WISHING = "MAKING AN ACTION FOR EXPRESSING A DESIRE FOR SOMETHING OR AN 
    IDEA2"
    SUPPLY = "MAKE SOMETHING NEEDED OR REQUESTED OR ASKED FOR TO SOMEONE"
    NEEDED = "REQUIRED TO COMPLETE SOMETHING"
    REQUESTED = "SOMETHING THAT HAS BEEN ASKED FOR"
    ASSISTANCE = "THE ACTION OF HELPING SOMEONE WITH A JOB OR TASK"
    HELPING = "DOING SOMETHING FOR SOMEONE THAT CAN OR DOES REQUIRE HELP"
    SUCH2 = "TO BE AS CLOSE TO AN HIGH STATUS2"
    DESIRED = "STRONGLY WISHED FOR OR INTENDED"
    WISHED = "EXPRESS A DESIRE FOR THE SUCCESS OR GOOD_FORTUNE OF SOMEONE"
    GOOD_FORTUNE = "AN AUSPICIOUS STATE RESULTING FROM FAVORABLE OUTCOMES 
    GOOD_LUCK AND OR LUCKINESS"
    GOOD_LUCK = "USED TO EXPRESS WISHES FOR SUCCESS"
    EXPRESS = "CONVEY A THOUGHT OR FEELING IN WORDS OR BY GESTURES AND 
    CONDUCT2"
    CONDUCT2 = "THE MANNER IN WHICH A PERSON USES TO BEHAVE"
    CONVEY = "MAKE AN IDEA AND OR IMPRESSION AND OR FEELING KNOWN OR 
    UNDERSTANDABLE TO SOMEONE"
    UNDERSTANDABLE = "ABLE TO BE UNDERSTOOD"
    IMPRESSION = "AN IDEA AND OR FEELING AND OR OPINION ABOUT SOMETHING OR 
    SOMEONE"
    OPINION = "THE BELIEF OR PERSPECTIVE OF A PARTICULAR SUBJECT AND OR TOPIC AND 
    OR THING AND OR IDEA"
    PERSPECTIVE = "A PARTICULAR ATTITUDE OR APPROACH TOWARD SOMETHING OR WAY 
    OF REGARDING SOMETHING"
    APPROACH = "A WAY OF HAVING SOMETHING COMPLETE OR VIEWED AS A WAY TO 
    ENVISION OR PERCEIVE SOMETHING OR SOMEPLACE OR SOMEONE"
    SETTLED2 = "RESOLVED OR REACHED AGREEMENT ABOUT"
    ATTITUDE = "A SETTLED2 WAY OF THINKING OR FEELING ABOUT SOMEONE OR 
    SOMETHING"
    RESOLVED = "SOLVED AND COMPLETED A PROBLEM"
    SOLVE = "FIND AN ANSWER TO A PROBLEM"
    SOLVED = "SOMETHING THAT HAS BEEN COMPLETED AS A FOUND ANSWER TO A 
    PROBLEM"
    COMPLETED = "FINISHED MAKING OR DOING WHAT WAS STARTED AS A COMPLETE 
    EVENT OR SERIES OF EVENTS BROUGHT TOGETHER WITHIN TIME2"
    INTENDED = "CONSIDERED TO HAPPEN OR DECIDED UPON TO COME INTO PLACE"
    LUCKY = "HAVING AND BRINGING AND OR RESULTING FROM GOOD_LUCK"
    LUCKINESS = "THE AMOUNT OF STATED QUALITY THAT ONE HAS FROM GOOD_LUCK IN 
    THE FORMS OR STATE OF HAVING GOOD_LUCK"
    STRONGLY = "WITH GREAT POWER OR STRENGTH"
    UNEASINESS = "A FEELING OF ANXIETY OR DISCOMFORT"
    DISTRESS = "EXTREME ANXIETY"
    ANGER = "A STRONG FEELING OF ANNOYANCE AND OR DISPLEASURE AND OR 
    HOSTILITY"
    ANNOYANCE = "A THING THAT CAN ANNOY SOMEONE"
    ANNOY = "IRRITATE SOMEONE OR TO MAKE SOMEONE ANGRY IN SOME WAY"
    ANGRY = "CURRENTLY HAVING ANGER OR FEELING ANGER IN SOME WAY"
    NERVOUSNESS = "THE QUALITY OR STATE OF BEING NERVOUS"
    TRANQUIL = "FREE FROM DISTURBANCE"
    IRRITATE = "TO ANNOY SOMEONE AND OR MAKE SOMEONE IMPATIENT AND OR ANGRY"
    DISPLEASURE = "A FEELING OF ANNOYANCE OR DISAPPROVAL"
    DISAPPROVAL = "EXPRESSION OF AN UNFAVORABLE OPINION"
    UNFAVORABLE = "EXPRESSING OR SHOWING A SMALLER OF APPROVAL OR SUPPORT 
    THAN WHAT IS NEEDED"
    APPROVAL = "THE ACTION OF APPROVING SOMETHING"
    EXPRESSION = "THE PROCESS OF MAKING KNOWN SUCH2 THOUGHTS OR FEELINGS" 
    HOPEFUL = "FEELING OR INSPIRING OPTIMISM ABOUT A FUTURE EVENT"
    INSPIRING = "GIVING SOMEONE POSITIVE OR CREATIVE FEELINGS"
    OPTIMISM = "HOPEFULNESS AND CONFIDENCE ABOUT THE FUTURE OR THE 
    SUCCESSFUL OUTCOME OF SOMETHING"
    SUCCESSFUL = "ACCOMPLISHING AN GOAL AND OR TASK AND JOB AND OR PURPOSE"
    ACCOMPLISHING = "ACHIEVING OR COMPLETING SUCCESSFULLY"
    ACCOMPLISH = "ACHIEVE OR COMPLETE SUCCESSFULLY"
    SUCCESSFULLY = "IN A WAY THAT ACCOMPLISHES A DESIRED GOAL OR RESULT"
    ACCOMPLISHES = "COMPLETES AND OR OBTAINS"
    COMPLETES = "SUCCEED IN ACCOMPLISHING"
    ASSET = "PROPERTY OWNED BY A PERSON AND OR ENTITY"
    RESULTING = "OCCURRING OR FOLLOWING AS THE CONSEQUENCE OF SOMETHING"
    WISHES = "MORE THAN ONE WISH"
    WISH = "FEEL OR EXPRESS A STRONG DESIRE OR HOPE FOR SOMETHING THAT IS NOT 
    ABLE TO BE OBTAINED WITH LITTLE DIFFICULTY"
    AMBIGUITY = "THE QUALITY OF BEING OPEN TO MORE THAN ONE INTERPRETATION"
    HOPEFULNESS = "A PERSON THAT HAS GREATER CHANCES SUCCEED FROM HOPEFUL 
    DECISION MAKING"
    OCCURRING = "HAPPENING AND IN EFFECT"
    ACHIEVING = "ACCOMPLISHING AND COMPLETING"
    COMPLETING = "SUCCEEDING IN ACCOMPLISHING"
    SUCCEEDING = "SUCCESSFULLY COMPLETING AS A FORM OF COMPLETION"
    SITUATIONS = "MORE THAN ONE SITUATION"
    AUSPICIOUS = "CHARACTERIZED BY SUCCESS"
    CHARACTERIZED = "DESCRIBED BY THE DISTINCTIVE NATURE OR FEATURES OF"
    FEATURE = "A DISTINCTIVE ATTRIBUTE OR ASPECT OF SOMETHING"
    FEATURES = "MORE THAN ONE FEATURE"
    NEEDS = "MORE THAN ONE NEED"
    DISTINCTIVE = "CHARACTERISTIC OF ONE PERSON OR THING AND OR SERVING TO 
    DISTINGUISH IT FROM OTHERS"
    EXISTING = "TO SERVE AS AN ATTRIBUTE OR QUANTITY"
    SERVE = "PERFORM JOBS AND OR TASKS AND OR WORK AND OR SERVICES FOR"
    DISTINGUISH = "PERCEIVE OR RECOGNIZE AND NOTICE A DIFFERENCE"
    FULFILLMENT = "THE MEETING OF A REQUIREMENT AND OR CONDITION AND OR 
    PRECONDITION AND OR PREREQUISITE"
    REQUIREMENT = "A CONDITION NECESSARY"
    NECESSARY = "REQUIRED TO BE DONE"
    PRECONDITION = "A CONDITION THAT MUST BE COMPATIBLE AND RECOGNIZED AS 
    EXISTING BEFORE OTHER THINGS CAN HAPPEN OR BE DONE"
    ASSETS = "MORE THAN ONE ASSET"
    BLOCK = "PREVENT FROM MAKING ACCESS POSSIBLE OR FURTHER ADVANCING OR 
    ACTION OR PASSAGEWAY OR MAKING A PROTECTION STANCE"
    NULLIFY = "PREVENT FROM HAPPENING AND NOT ALLOW TO COME INTO EFFECT OR 
    ACTION"
    NEGATE = "PREVENT FROM FORMING OR ACTIVATING"
    CANCEL_OUT = "NOT ALLOW TO HAPPEN AND NEGATE ALL FORMS OF EFFECTS AND 
    ACTION2"
    RESTRAINT = "A MEASURE OR CONDITION THAT ALLOWS SOMEONE OR SOMETHING TO 
    STAY AND HAVE CONTROL OR WITHIN LIMITS"
    DREAD = "ANTICIPATE WITH GREAT FEAR"
    STRESSFUL = "CAUSING MENTAL OR EMOTIONAL STRESS"
    TOLERATE = "ACCEPT OR ENDURE SOMEONE OR SOMETHING UNPLEASANT OR DISLIKED 
    WITH FORBEARANCE"
    TROUBLE = "DIFFICULTY OR PROBLEMS"
    CONSIDERATE = "CAREFUL NOT TO CAUSE INCONVENIENCE OR CREATE HARM TO 
    OTHERS"
    PERSISTANT = "CONTINUING TO EXIST OR ENDURE OVER A PROLONGED TIMEFRAME"
    SIGNIFICANT = "SUFFICIENTLY GREAT OR IMPORTANT TO BE WORTHY OF ATTENTION"
    HAPPINESS = "THE STATE OF BEING HAPPY"
    FRIENDLY = "KIND AND PLEASANT"
    READINESS = "WILLINGNESS TO DO SOMETHING"
    EXPECTED = "REGARDED AS LIKELY TO HAPPEN"
    MONEY = "A CURRENT MEDIUM OF EXCHANGE"
    RELATIVELY = "VIEWED IN COMPARISON WITH SOMETHING ELSE RATHER THAN 
    ABSOLUTELY"
    MODERATE = "AVERAGE IN AMOUNT AND OR INTENSITY AND OR" 
    FAVORABLE = "EXPRESSING APPROVAL"
    GESTURES = "MORE THAN ONE GESTURE"
    GESTURE = "A MOVEMENT OF PART OF THE BODY"
    KIND = "SOMEONE WHO IS GENEROUS AND HELPFUL AND WHO CAN THINK OF OTHER 
    AN ENTITIES FEELINGS"
    PLEASANT = "GIVING A SENSE OF HAPPY SATISFACTION"
    GENEROUS = "SHOWING KINDNESS TOWARD OTHERS"
    HELPFUL = "GIVING OR READY TO GIVE HELP"
    READY = "IN A STATE FOR AN EVENT AND OR ACTION AND OR SITUATION TO HAPPEN"
    SERVING = "THE ACTION OF ONE THAT SHALL SERVE SOMETHING"
    MORAL = "CONCERNED WITH THE PRINCIPLES OF RIGHT AND WRONG BEHAVIOR"
    CONCERNED = "FOCUSED AND CONCENTRATED ON"
    PRINCIPLES = "MORE THAN ONE PRINCIPLE"
    PRINCIPLE = "A FUNDAMENTAL TRUTH THAT CAN SERVE AS THE FOUNDATION FOR A 
    SYSTEM OF BELIEF OR BEHAVIOR OR FOR A CHAIN OF REASONING"
    FUNDAMENTAL = "A CENTRAL OR PRIMARY RULE BUILT UPON A SERIES OF BELIEFS"
    IMPORTANCE = "THE STATE OR FACT OF BEING OF GREAT SIGNIFICANCE OR VALUE"
    DIGNITY = "A RANK OR CLASS THAT SOMEONE OR SOMETHING IS APART OF"
    MODERATION = "AN EQUAL AMOUNT OF MODERATE PORTION THAT IS SPREAD ACROSS A 
    SPECIFIC PERIOD OF TIME2 WITHIN A TIMEFRAME"
    FACT = "A THING THAT IS KNOWN TO BE TRUE"
    SIGNIFICANCE = "THE MEANING TO BE FOUND IN WORDS OR EVENTS"
    STYLE = "A DISTINCTIVE APPEARANCE"
    SELF = "THE ESSENTIAL BEING THAT A PERSON IS THAT DISTINGUISHES THEM FROM 
    OTHERS"
    VOLUNTARILY = "DECISION MADE BY THE CHOICE OF FREE WILL"
    STRICTLY = "USED TO INDICATE THAT ONE IS APPLYING WORDS OR RULES EXACTLY OR 
    RIGIDLY"
    DETERRED = "PREVENT THE OCCURANCE OF"
    ESTIMATION = "A CALCULATION OF THE VALUE AND OR NUMBER AND OR QUANTITY AND 
    OR EXTENT OF SOMETHING WITHOUT HAVING ALL DATA2 KNOWN"
    MENTAL = "RELATING TO THE MIND"
    IMPATIENT = "HAVING OR SHOWING A REPEATED REACTION TO BE WITH LITTLE TO NO 
    DIFFICULT RESPONSE TO BECOME IRRITATED OR PROVOKED"
    HOSTILITY = "HOSTILE BEHAVIOR"
    DISLIKED = "FEEL DISTASTE FOR OR HOSTILITY TOWARD"
    STRESS = "A STATE OF MENTAL OR EMOTIONAL STRAIN CAUSED BY ADVERSE 
    CIRCUMSTANCES"
    FEAR = "AN EMOTION CAUSED BY THE BELIEF THAT SOMEONE OR SOMETHING IS 
    DANGEROUS"
    INCONVENIENCE = "TROUBLE OR DIFFICULTY CAUSED TO SOMEONE PERSONAL 
    REQUIREMENTS"
    ANTICIPATE = "REGARD AS PROBABLE AND OR EXPECT OR PREDICT"
    WILLINGNESS = "THE QUALITY OR STATE OF BEING PREPARED TO DO SOMETHING"
    REGARDED = "CONSIDERED AND THINK OF IN A SPECIFIED WAY"
    GETTING = "SUCCEEDING IN OBTAINING"
    CLEARNESS = "EASY TO PERCEIVE AND UNDERSTAND AND OR INTERPRET"
    ESPECIALLY = "TO A GREAT EXTENT"
    CAUTIOUS = "CHARACTERIZED BY THE DESIRE TO AVOID POTENTIAL PROBLEMS"
    WORTHY = "HAVING OR SHOWING THE QUALITIES OR ABILITIES THAT GIVE 
    RECOGNITION IN A SPECIFIED WAY"
    EXCHANGE = "AN ACTION OF GIVING ONE THING AND RECEIVING ANOTHER"
    GET = "RECEIVE SOMETHING FROM A SPECIFIC LOCATION OR SOURCE"
    EXPECTATIONS = "A STRONG BELIEF THAT SOMETHING WILL HAPPEN OR BE THE CASE IN 
    THE FUTURE"
    SUFFICIENTLY = "TO AN ADEQUATE CLASS OR LEVEL OF STATUS"
    ONESELF = "USED TO EMPHASIZE THAT ONE DOES SOMETHING INDIVIDUALLY OR 
    WITHOUT HELP"
    LIKELY = "HIGH CHANCES OF HAPPENING AND TO BE TRUE"
    COMPARISON = "A CONSIDERATION OR ESTIMATE OF THE SIMILARITIES OR 
    DIFFERENCES BETWEEN TWO THINGS"
    ABSOLUTELY = "WITH NO QUALIFICATION AND OR RESTRICTION AND OR LIMITATION"
    EASY = "DOES NOT REQUIRE A LARGE AMOUNT OF EFFORT OR HAS A STRONG FORM OF 
    DIFFICULTY"
    ESTIMATE = "AN APPROXIMATE CALCULATION OR DECISION OF THE VALUE AND OR 
    NUMBER AND OR QUANTITY AND OR EXTENT OF SOMETHING"
    APPROXIMATE = "CLOSE TO THE ORIGINAL RESULT BUT NOT ALWAYS AN COMPLETE 
    ACCURATE OR EXACT MEASUREMENT"
    PROBABLE = "LIKELY TO BE THE CASE OR TO HAPPEN"
    OCCURANCE = "AN INCIDENT OR EVENT"
    INCIDENT = "AN EVENT OR OCCURANCE"
    SURE = "CONFIDENT SOMETHING IS GOING TO HAPPEN"
    DIFFERENCES = "MORE THAN ONE DIFFERENCE"
    RANK = "A SPECIFIC POSITION FOR A CLASS WITHIN A HEIRARCHIAL STRUCTURE"
    EXPECT = "REQUIRE TO ARRIVE AT A SPECIFIC TIMEFRAME AND MANNER"
    REGARD = "RELATING TO THE CURRENT CIRCUMSTANCE"
    RECOGNITION = "A FORM OF PROCESS THAT CAN RECOGNIZE CERTAIN SPECIFIC FORMS 
    OF FACTS AND INFORMATION FROM DIFFERENT CLASSES OR GENRES OF INFORMATION"
    PREPARED = "ALREADY DONE AHEAD OF TIME2"
    ADEQUATE = "SPECIFIC AND PARTICULAR RESPONSE THAT HOLDS MANNERISMS"
    MANNERISM = "A FORM OF ETTIQUITE THAT HOLDS SPECIFIC PATTERNS ON HOW TO ACT 
    PROPERLY"
    PROPER = "WHAT SHOULD BE USED DEPENDING ON THE CIRCUMSTANCE OR CLASS 
    WITHIN"
    PROPERLY = "USING MANNERS WITH PROPER METHODS AND APPROACH"
    MANNERS = "MORE THAN ONE MANNER"
    ETTIQUITE = "A SPECIFIC FORMAL OR PROPER STYLE FOR HOW TO BEHAVE DURING AN 
    EVENT OR CIRCUMSTANCE"
    MANNERISMS = "MORE THAN ONE MANNERISM"
    FACTS = "MORE THAN ONE FACT"
    REQUIREMENTS = "MORE THAN ONE REQUIREMENT"
    MISHAP = "AN INCIDENT NOT CONTAINING GOOD_FORTUNE OR GOOD_LUCK"
    AVOIDING = "PREVENTING FROM DOING"
    UNCOMFORTABLE = "CAUSING OR FEELING UNEASE"
    UNEASY = "CAUSING OR FEELING ANXIETY AND OR UNCOMFORTABLE"
    DISCOMFORT = "MAKE FEEL UNEASY ANXIOUS"
    UNPLEASANT = "UNFRIENDLY AND INCONSIDERATE"
    UNFRIENDLY = "NOT FRIENDLY"
    INCONSIDERATE = "THOUGHTLESSLY CAUSING HURT OR INCONVENIENCE TO OTHERS"
    THOUGHTLESSLY = "WITHOUT CONSIDERATION OF THE POSSIBLE CONSEQUENCES"
    SUFFERING = "THE STATE OF DISTRESS"
    UNASSUMING = "MODEST"
    UPSET = "MAKE SOMEONE UNHAPPY AND OR DISAPPOINTED AND OR WORRIED"
    PREFER = "LIKE ONE IDEA OR THING OR SUBJECT OR TOPIC OR PERSON BETTER THAN 
    ANOTHER OR OTHERS"
    HEALING = "THE PROCESS OF MAKING OR BECOMING SOUND OR HEALTHY AGAIN"
    HEAL = "BECOME SOUND OR HEALTHY AGAIN"
    REFRESH = "TO RESTORE OR MAINTAIN BY RENEWING SUPPLY"
    HEALTHY = "IN DESIRED HEALTH"
    HEALTH = "THE MENTAL OR PHYSICAL CONDITION OF SOMEONE OR SOMETHING"
    REFRESHING = "HAVING THE POWER TO RESTORE FRESHNESS AND OR VITALITY AND OR 
    ENERGY"
    RENEW = "TO RENEW MEANS TO BRING BACK TO AN ORIGINAL CONDITION OF 
    FRESHNESS AND VIGOR"
    FRESHNESS = "THE QUALITY OF BEING PLEASANTLY NEW OR DIFFERENT"
    VIGOR = "PHYSICAL STRENGTH AND DESIRED HEALTH"
    VITALITY = "THE POWER GIVING AND CONTINUING TO GAIN A CONTINUAL AMOUNT OF 
    ENERGY"
    INDIVIDUALLY = "IN AN INDIVIDUAL CAPACITY"
    EMPHASIZE = "GIVE SPECIAL IMPORTANCE TO SOMETHING IN WRITING"
    FORBEARANCE = "RESTRAINT AND CONTROL OF SOMETHING"
    CONSIDERATION = "CAREFUL THOUGHT WITHIN A PERIOD OF TIME"
    QUALIFICATION = "THE ACTION OR FACT OF QUALIFYING"
    RENEWANCE = "TO BEGIN OR TAKE UP AGAIN"
    CONSEQUENCES = "MORE THAN ONE CONSEQUENCE"
    PREVENTING = "PREVENT FROM HAPPENING COMING INTO EFFECT OR PLACE"
    UNHAPPY = "NOT HAPPY"
    HURT = "PHYSICAL HARM OR INJURY"
    DISAPPOINTED = "SAD OR DISPLEASED BECAUSE SOMEONE OR SOMETHING HAS FAILED 
    TO FULFILL HOPES OR EXPECTATIONS"
    WORRIED = "ANXIOUS OR TROUBLED ABOUT ACTUAL OR POTENTIAL PROBLEMS"
    INCONVENIENCE = "TROUBLE OR DIFFICULTY CAUSED TO ONES PERSONAL 
    REQUIREMENTS OR COMFORT"
    HOPES = "MORE THAN ONE HOPE"
    EXPECTATIONS = "MORE THAN ONE EXPECTATION"
    EXPECTATION = "A STRONG BELIEF THAT SOMETHING WILL HAPPEN OR BE THE CASE IN 
    THE FUTURE"
    VIRTUE = "A QUALITY CONSIDERED DESIRED INSIDE A PERSON"
    SOMEPLACE = "A REFERENCE TO A SPECIFIC PLACE AND OR LOCATION WITHIN SPACE 
    AND OR TIME"
    INDISTINCTNESS = "NOT CLEAR OR SHARPLY_DEFINED"
    SHARPLY_DEFINED = "IN A WAY THAT IS DISTINCT IN DETAIL"
    RENEWING = "HAPPENING TO RENEW AT THE CURRENT TIMEFRAME"
    PLEASANTLY = "IN AN ENJOYABLE OR AGREEABLE MANNER"
    FORMAL = "CONTAINING A FORM OF ETTIQUITE OR MANNERISMS THAT SHOW PROPER 
    DECISION MAKING WITH BOTH ATTITUDE AND STYLE AS WELL AS PRESENTATION AND 
    CONFIDENCE"
    PRUDENT = "SHOWING CARE AND THOUGHT FOR THE FUTURE"
    PRUDENCE = "THE QUALITY OF BECOMING PRUDENT"
    FRIGHTENS = "INCLUDES AND DOES APPLY FEAR WITHIN"
    ENJOYMENT = "THE STATE OR PROCESS OF TAKING PLEASURE IN SOMETHING"
    DERIVED = "A IDEA CREATED ON A EXTENSION OF LOGIC OR MODIFICATION OF 
    ANOTHER IDEA"
    OFFERING = "A CONTRIBUTION OR A THING OFFERED AS A TOKEN OF DEVOTION"
    DEVOTION = "AS IN AFFECTION A FEELING OF STRONG OR CONSTANT REGARD FOR AND 
    DEDICATION TO SOMEONE"
    DEDICATION = "THE QUALITY OF BEING DEDICATED TO A TASK OR PURPOSE"
    DEDICATED = "DEVOTED TO A TASK OR PURPOSE"
    RESOURCES = "MORE THAN ONE RESOURCE"
    DEVOTED = "GIVEN OVER TO THE STUDY OF"
    TOKEN = "A THING SERVING AS A REPRESENTATION OF A FACT AND OR QUALITY AND OR 
    FEELING"
    ENJOYABLE = "GIVING DELIGHT OR PLEASURE"
    DELIGHT = "GREAT PLEASURE"
    AGREEABLE = "ENJOYABLE AND PLEASURABLE"
    PLEASURABLE = "PLEASING"
    PLEASING = "SATISFYING OR APPEALING"
    APPEALING = "ATTRACTIVE OR INTERESTING"
    ATTRACTIVE = "PLEASING OR APPEALING TO THE SENSES"
    INTERESTING = "HOLDING OR TO GAIN THE ATTENTION OF OR AROUSING INTEREST"
    AROUSE = "EXCITE OR PROVOKE SOMEONE TO STRONG EMOTIONS"
    EXCITE = "CAUSE STRONG FEELINGS OF ENTHUSIASM AND EAGERNESS WITHIN"
    ENTHUSIASM = "INTENSE AND EAGER ENJOYMENT AND OR INTEREST AND OR 
    APPROVAL"
    EAGER = "WANTING TO DO OR HAVE SOMETHING VERY MUCH"
    EAGERNESS = "ENTHUSIASM TO DO OR TO HAVE SOMETHING"
    SATISFYING = "GIVING FULFILLMENT OR THE PLEASURE ASSOCIATED WITH SOMETHING 
    OR SOMEPLACE OR SOMEONE"
    ASSOCIATED = "CONNECTED WITH ANOTHER GROUP OR GROUPS"
    PROVOKE = "GIVE RISE TO A REACTION OR EMOTION IN SOMEONE OR AROUSE 
    SOMEONE TO DO OR FEEL SOMETHING"
    PROVOKED = "ENVOKED SOMEONE TO FEEL SOMETHING STRONGLY"
    ENVOKE = "AROUSE SOMEONE TO DO SOMETHING USING ENERGY AND OR CHARMS 
    AND OR INCANTATIONS"
    ENVOKED = "SUMMONED A SPIRIT BY USING CHARMS OR INCANTATION"
    CHARM = "AN OBJECT AND OR SERIES OF WORDS THAT HAVE MAGIC POWER"
    INCANTATION = "A SERIES OF WORDS EXPRESSED AS A MAGIC SPELL OR CHARM"
    SUMMON = "CALL PEOPLE TO BECOME APART OF A MEETING"
    CHARMS = "MORE THAN ONE CHARM"
    INCANTATIONS = "MORE THAN ONE INCANTATION"
    SUMMONED = "BROUGHT INTO AN EVENT OR CIRCUMSTANCE TO MAKE ACTION BY THE 
    DEMAND OF THE SUMMONER"
    SUMMONER = "THE ONE WHO SHALL SUMMON SOMETHING OR SOMEONE WITH 
    COMPLETE CONTROL OF WHAT IS SUMMONED AS AN ENTITY FOR THE TIMEFRAME THAT 
    THE SUMMONED ENTITY IS SUMMONED"
    ENCHANT = "PUT SOMEONE OR SOMETHING UNDER A SPELL"
    ESSENTIAL = "A THING THAT IS ABSOLUTELY NECESSARY"
    DISTINGUISHES = "RECOGNIZE SOMEONE OR SOMETHING AS SIMILAR OR DIFFERENT"
    THEM = "TO REFERENCE TO TWO OR MORE PEOPLE OR THINGS PREVIOUSLY 
    RECOGNIZED WITHIN REFERENCE"
    RIGIDLY = "IN A STRICT OR EXACTING WAY"
    INDICATE = "SUGGEST AS A DESIRED OR NECESSARY CHOICE OF ACTION"
    DISTASTE = "TO NOT LIKE SOMETHING BECAUSE YOU CONSIDER IT UNPLEASANT"
    HOSTILE = "UNFRIENDLY OR AGAINST THE IDEA OF SOMETHING"
    STRAIN = "FORCES THAT IS ABLE PULL FROM MULTIPLE LOCATIONS UNTIL IT CREATES 
    STRESS UPON THE ENTITY"
    ADVERSE = "PREVENTING SUCCESS OR DEVELOPMENT AND OR IS HARMFUL"
    CIRCUMSTANCES = "MORE THAN ONE CIRCUMSTANCE"
    IRRITATED = "FEELING DISCOMFORT OR DISCONTENT"
    TROUBLED = "AFFECTED BY PROBLEMS OR UNCOMFORTABLE CIRCUMSTANCES"
    SAD = "TO BE FEELING UNHAPPY"
    DISPLEASED = "FEELING OR SHOWING ANNOYANCE AND DISPLEASURE"
    QUALIFYING = "DENOTING SOMEONE OR SOMETHING THAT IS COMPATIBLE FOR 
    SOMETHING TO TAKE PLACE OR HAPPEN"
    DENOTE = "BE A SIGN OF"
    DENOTING = "SHOWING A SIGN OF SOMETHING OR SOMEPLACE OR SOMEONE"
    SIGN = "AN OBJECT AND OR QUALITY AND OR EVENT IN WHICH SOMEONE OR 
    SOMETHING HAS A PRESENCE OR OCCURANCE THAT DOES INDICATE THE PROBABLE 
    PRESENCE OR OCCURANCE OF SOMETHING ELSE"
    FULFILL = "BRING TO COMPLETION OR REALITY"
    COMFORT = "A STATE OF PHYSICAL ABSENSE OF DIFFICULTY OR EFFORT AND FREEDOM 
    FROM UNCOMFORTABLE FEELINGS OR BINDINGS"
    CONTRIBUTION = "A GIFT OR PAYMENT TO A COMMON SOURCE OF GRATITUDE"
    GRATITUDE = "THE QUALITY OF BEING THANKFUL"
    THANKFUL = "EXPRESSING GRATITUDE AND RELIEF"
    RELIEF = "A FEELING OF REASSURANCE AND RELAXATION FOLLOWING RELEASE FROM 
    ANXIETY OR DISTRESS"
    REASSURANCE = "THE ACTION OF REMOVING THE DOUBTS OR FEARS OF SOMEONE"
    DOUBTS = "MORE THAN ONE DOUBT"
    FEARS = "MORE THAN ONE FEAR"
    RELAXATION = "THE STATE OF BEING FREE FROM TENSION AND ANXIETY"
    TENSION = "MENTAL OR EMOTIONAL STRAIN"
    OFFERED = "GIVE AN OPPORTUNITY FOR SOMETHING TO BE MADE OR CREATED"
    AROUSING = "REACHING A RESPONSE OR REACTION TO AROUSE SOMEONE OR 
    SOMETHING"
    STRICT = "FOLLOWING RULES OR BELIEFS EXACTLY"
    EXACTING = "MAKING GREAT ORDERS REGARDING ONES SKILL AND OR ATTENTION AND 
    OR OTHER RESOURCES"
    DOUBT = "TO FEAR"
    ACTUAL = "EXISTING WITHIN THE PRESENT"
    PROUD = "FEELING STRONG PLEASURE OR SATISFACTION AS A RESULT OF ONES OWN 
    ACHIEVEMENTS"
    RHYTHM = "A STRONG AND OR REPEATED PATTERN OF MOVEMENT OR SOUND"
    PITCH = "THE QUALITY OF A SOUND CONTROLLED BY THE RATE OF VIBRATIONS 
    PRODUCING IT"
    TREBLE = "CREATED WITHIN THE EXISTENCE OF THREE PARTS"
    BASS = "THE LOW FREQUENCY OUTPUT OF A AUDIO SYSTEM"
    LOW = "BELOW AVERAGE IN AMOUNT"
    MUSIC_SCALE = "A SERIES OF NOTES DIFFERING IN PITCH ACCORDING TO A SPECIFIC 
    PATTERN"
    MUSICAL_NOTE = "DESCRIBES THE PITCH AND THE DURATION OF A SPECIFIC SOUND"
    MUSIC2 = "VOCAL AND OR INSTRUMENTAL SOUNDS COMBINED IN A WAY AS TO 
    PRODUCE BEAUTY OF FORM, HARMONY, AND EXPRESSION OF EMOTION"
    BEAUTY = "A COMBINATION OF QUALITIES THAT PLEASES THE INTELLECT OR MORAL 
    SENSE"
    INTELLECT = "THE FACULTY OF REASONING AND UNDERSTANDING OBJECTIVELY"
    OBJECTIVELY = "IN A WAY THAT IS NOT INFLUENCED BY PERSONAL FEELINGS OR 
    OPINIONS"
    OPINIONS = "MORE THAN ONE OPINION"
    DESCRIBES = "ATTEMPTS TO GIVE THE COMPLETE DESCRIPTION OF SOMETHING OR 
    SOMEPLACE"
    INFLUENCE = "THE CAPACITY TO HAVE AN EFFECT ON A CHARACTER AND OR 
    DEVELOPMENT AND OR BEHAVIOR OF SOMEONE OR SOMETHING AND OR THE EFFECT 
    ITSELF"
    INFLUENCED = "CONTROLLED BY INFLUENCE OR REPEATING PATTERNS AND OR 
    CIRCUMSTANCES"
    ATTEMPTS = "GIVES AN EFFORT TO ATTEMPT"
    INFLUENCE2 = "HAVE AN INFLUENCE UPON USING A DEVICE OR HAVING SOMETHING 
    INFLUENCE THE PHYSICAL2 OR MENTAL SENSES"
    NOTE = "A SERIES OF FACTS AND OR TOPICS AND OR THOUGHTS THAT ARE WRITTEN 
    DOWN AS AN SUPPORT MEMORY REMEMBER AND OR RECALL"
    ATTEMPT = "MAKE AN EFFORT TO ACHIEVE OR COMPLETE"
    WILLED_ENERGY = "A CLASS OF ENERGY THAT IS USED BY THE SOLE CONSENT OF THE 
    SOURCE CONNECTION FOR THE ENERGY TO BE APPLIED AND REQUIRED FOR BOTH 
    HOST AND SOURCE TO BE IN AGREEMENT FOR ENERGY TO BE APPLIED"
    SOLE = "REPRESENTING AN INDIVIDUAL DECISION ONLY WITHOUT INFLUENCE FROM 
    OTHER ENTITIES OR CHOICES AND IS REPRESENTING ONLY ONE INDIVIDUAL"
    CONSENT = "PERMISSION FOR SOMETHING TO HAPPEN OR AN AGREEMENT TO DO 
    SOMETHING"
    FORCED_ENERGY = "A CLASS OF ENERGY THAT IS USED BY APPLYING PRESSURE OR 
    FORCE TO THE SOURCE CONNECTION FOR EFFECTS TO HAPPEN"
    NULL_AND_VOID = "MEANS HAVING NO EFFECT AND TO BE CONSIDERED AS IF IT DOES 
    NOT EXIST"
    EMOTIONAL_ABUSE = "IS CONSIDERED TO BE ANYTHING THAT CAUSES FEAR OR TO 
    MANIPULATE THE EMOTIONS OF A LESSER INDIVIDUAL"
    SPIRITUAL_ABUSE = "IS THE HARM THAT COMES TO THE HUMAN SPIRIT"
    HOST = "A COMPUTER THAT IS ACCESSIBLE WITHIN A NETWORK OR SOMEONE WHO IS 
    ABLE TO COMMUNICATE WITH OTHERS THAT EXIST WITHIN THE SAME NETWORK OF 
    OTHERS"
    WAITER = "SOMEONE WHO IS RESPONSIBLE FOR TAKING ORDERS FROM SOMEONE AND 
    TO SEND THEIR SERVICE REQUESTED TO THEM"
    RESPONSIBLE = "SOMEONE WHO IS ABLE TO BE GIVEN TRUST"
    NOTES = "MORE THAN ONE NOTE"
    DIFFERING = "NOT THE SAME AS EACH OTHER"
    WEALTHY = "HAVING A GREAT AMOUNT OF MONEY AND OR RESOURCES AND OR 
    ASSETS"
    RICH = "PRODUCING A LARGE QUANTITY OF SOMETHING"
    EXPENSIVE = "REQUIRE A LARGE AMOUNT OF MONEY"
    COST = "AN AMOUNT THAT HAS TO BE PAID OR SPENT TO BUY OR OBTAIN SOMETHING"
    PRICE = "THE AMOUNT EXPECTED AND REQUIRED OR GIVEN PAYMENT FOR SOMETHING"
    VALUE = "THE REGARD THAT SOMETHING IS HELD TO DESERVE"
    COMMUNITY = "A PARTICULAR AREA OR PLACE CONSIDERED TOGETHER AS A WHOLE 
    GROUP"
    ORGANIZATION = "AN ORGANIZED GROUP OF PEOPLE WITH A PARTICULAR PURPOSE"
    PAY = "GIVE SOMEONE MONEY FOR WORK AND OR SERVICES AND OR PRODUCTS 
    COMPLETED"
    PAID = "GIVEN PAYMENT TO PAY SOMEONE"
    SPEND = "PAY SOMEONE FOR RESOURCES AND OR ASSETS AND OR SERVICES"
    SPENT = "INCOME THAT HAS BEEN USED TO BUY SOMETHING OR PAY FOR A SERVICE"
    BUY = "OBTAIN IN EXCHANGE FOR PAYMENT"
    DESERVE = "DO SOMETHING OR TO HAVE OR SHOW QUALITIES WORTHY OF 
    SOMETHING"
    FIXED_INCOME = "IS A TERM THAT CAN REFER TO EITHER A INCOME THAT DOES NOT 
    CHANGE IN AMOUNT"
    REFER = "DIRECT THE ATTENTION OF SOMEONE TO"
    INCOME = "MONEY RECEIVED AT A SPECIFIC INTERVAL OR FREQUENCY OF TIME WITHIN 
    A SERIES OF CONTINUOUS REPEATED EVENTS THAT RESULT IN SOMEONE GETTING 
    PAYMENT"
    DURATION = "THE TIME DURING WHICH SOMETHING CONTINUES"
    CONTINUES = "HAPPENS TO CONTINUE EXISTING WITHIN TIME2 AND SPACE2"
    PLEASES = "GIVES SATISFACTION OR PLEASURE"
    PLEASE = "CAUSE TO FEEL HAPPY AND OR TO FEEL ENJOYMENT"
    INSTRUMENTS = "MORE THAN ONE INSTRUMENT"
    PERFORM = "GIVEN FULFILLMENT AND COMPLETED"
    PERFORMED = "COMPLETED THE METHOD TO PERFORM A SPECIFIC JOB OR TASK OR 
    AMOUNT OF WORK DURING AN EVENT"
    INSTRUMENTAL = "SOMETHING THAT IS PERFORMED USING INSTRUMENTS AND 
    WITHOUT VOCAL SOUNDS"
    PAIN = "A SPECIFIC AMOUNT OF SENSITIVITY TO ANOTHER SENSOR"
    class Language_Extension_002_2:
    RECURSIVE_DEFINITION = "IS USED TO DEFINE THE ELEMENTS INSIDE A LIST WITH 
    TERMS OF OTHER ELEMENTS INSIDE THE LIST"
    QUANTUM_MECHANICS = "IS A FUNDAMENTAL THEORY THAT CAN PROVIDE A 
    DESCRIPTION OF THE PHYSICAL2 PROPERTIES OF NATURE AT THE SCALE OF ATOMS AND 
    SUBATOMIC PARTICLES"
    QUANTUM_CHEMISTRY = "IS A SUBCLASS OF PHYSICAL_CHEMISTRY THAT GIVES FOCUS 
    TO INCLUDING QUANTUM_MECHANICS TO CHEMICAL SYSTEMS"
    MACROSCOPIC_SCALE = "IS THE LENGTH SCALE ON WHICH OBJECTS3 OR PHENOMENA 
    ARE AT A SPECIFIC SIZE THAT IS LARGE AND CAPABLE TO BE VISIBLE TO THE EYESIGHT 
    WITHOUT REQUIREMENT TO MAGNIFY TO SEEN"
    MICROSCOPIC_SCALE = "IS THE SCALE OF OBJECTS3 AND EVENTS SMALLER THAN 
    THOSE THAT CAN BE SEEN WITH VERY LITTLE DIFFICULTY BY THE EYESIGHT AND MAY 
    REQUIRE TO MAGNIFY EYESIGHT TO SEE THE OBJECTS3"
    QUANTUM_FIELD_THEORY = "A FRAMEWORK THAT COMBINED 
    CLASSICAL_FIELD_THEORY AND SPECIAL_RELATIVITY AND QUANTUM_MECHANICS 
    TOGETHER"
    PHYSICAL_CHEMISTRY = "IS THE STUDY OF MACROSCOPIC_SCALE AND 
    MICROSCOPIC_SCALE PHENOMENA WITHIN CHEMICAL SYSTEMS RELATING TO TERMS 
    OF THE PRINCIPLES AND OR PRACTICES AND OR IDEAS OF PHYSICS SUCH2 AS MOTION 
    AND ENERGY AND FORCE AND TIME2 AND THERMODYNAMICS AND 
    QUANTUM_CHEMISTRY AND STATISTICAL_MECHANICS AND ANALYTICAL_DYNAMICS 
    AND CHEMICAL_EQUALIBRIA"
    THERMODYNAMICS = "A CLASS THAT DESCRIBES THE FUNCTION AND USE OF HEAT AND 
    WORK AND TEMPERATURE AND THE RELATION IT HAS TO ENERGY AND THE PHYSICAL2 
    PROPERTIES OF MATTER"
    STATISTICAL_MECHANICS = "A FRAMEWORK THAT USES ANALYSIS AND SCANNING 
    METHODS AS WELL AS MANY FORMS OF CALCULATION AND ALSO INCLUDES 
    PROBABILITY_THEORY TO LARGE AMOUNTS OF PRODUCTS BROUGHT TOGETHER THAT 
    CONSISTS MICROSCOPIC_SCALE ENTITIES"
    ANALYTICAL_DYNAMICS = "IS CONCERNED WITH THE RELATION BETWEEN MOTION OF 
    BODIES AND ITS CAUSES"
    CHEMICAL_EQUALIBRIA = "IS THE STATE IN WHICH BOTH THE CHEMICAL_REACTANTS 
    AND CHEMICAL_PRODUCTS ARE CURRENTLY EXISTING IN CONDENSED AMOUNTS AND 
    TO WHICH HAVE NO FURTHER CHANCES OR CAPABILITIES TO CHANGE WITHIN ANY 
    POINT INSIDE TIME2"
    ENUMERATED_TYPE = "IS A DATA2 TYPE INCLUDING A LIST OF NAMED VALUES KNOWN AS 
    ENUMERATION_ELEMENTS AND ENUMERATION_MEMBERS"
    ENUMERATOR_NAMES = "ARE A WAY OF IDENTIFICATION THAT BEHAVE AS CONSTANT 
    DATA2 WITHIN THE LANGUAGE"
    PROGRAMMING_VALUE = "IS THE REPRESENTATION OF SOME ENTITY THAT A PROGRAM 
    CAN MANIPULATE"
    CLASSICAL_FIELD_THEORY = "IS A PHYSICAL_THEORY THAT CAN PREDICT HOW ONE OR 
    MORE FIELD COMMUNICATE AND CONNECT WITH MATTER USING EQUATIONS FOR 
    FIELDS"
    PHYSICAL_THEORY = "IS A CLASS THAT EMPLOYS EQUATIONS AND OR FORMULAS AND 
    OR ALGORITHMS FOR DEVELOPED IDEAS AND MORE THAN ONE ABSTRACTION OF 
    PHYSICAL2 OBJECTS3 AND SYSTEMS TO MAKE UNDERSTANDING OF AND OR EXPLAIN 
    AND PREDICT NATURAL PHENOMENA"
    VECTOR_ELEMENT = "IS A SPECIFIC PROGRAMMING_ELEMENT MADE FOR POSSIBLE USE 
    WITH AND OR INSIDE VECTOR DATA2"
    VECTOR_SPACE = "IS A LIST THAT HAS MORE THAN ONE VECTOR_ELEMENT MAY BE 
    ADDED TOGETHER AND MULTIPLIED BY SCALAR NUMBERS"
    SCALAR = "IS AN ELEMENT OF A FIELD WHICH IS USED TO DEFINE A VECTOR_SPACE"
    INSERTING = "MAKING AND PROCESSING THE PROCESS TO INSERT SOMETHING"
    DELETING = "MAKING AND PROCESSING THE PROCESS TO REMOVE SOMETHING"
    DATA_MANIPULATION_LANGUAGE = "IS A COMPUTER PROGRAMMING LANGUAGE USED 
    FOR INSERTING OR DELETING OR MODIFYING DATA2 INSIDE A DATABASE"
    ARRAY = "IS AN DATA2 STRUCTURE THAT CONSISTS OF A COLLECTION OF ELEMENTS 
    THAT ARE CONSIDERED VALUES OR VARIABLES"
    VECTOR_PROCESSOR = "IS A CENTRAL PROCESSING UNIT THAT CONNECTS AN LIST OF 
    INSTRUCTIONS WHEN THE INSTRUCTIONS ARE MADE TO OPERATE EFFECTIVELY ON 
    LARGE ONE DIMENSIONAL ARRAY SYSTEMS KNOWN AS PROCESSOR VECTORS"
    PROGRAMMING_ELEMENT = "IS ANY ONE OF THE DISTINCT OBJECTS3 THAT BELONG TO A 
    LIST OR AN ARRAY"
    TUPLE = "A FINITE SEQUENCE OF ELEMENTS"
    REPLACES = "CHANGES SOMETHING OUT FOR SOMETHING ELSE"
    SUBSTANCES = "MORE THAN ONE SUBSTANCE"
    CHEMICAL_REACTANTS = "ARE CONSIDERED THE SUBSTANCES THAT ARE PRESENT AT 
    THE START OF THE CHEMICAL_REACTION"
    CHEMICAL_PRODUCTS = "ARE CONSIDERED THE SUBSTANCES THAT ARE FORMED AT 
    THE END OF THE CHEMICAL_REACTION"
    REACTANT = "A SUBSTANCE THAT IS USED WITHIN A CHEMICAL_REACTION"
    EMPLOYS = "MAKES EFFICIENT OPERATION OF"
    SYNTHESIS_REACTION = "TWO OR MORE CHEMICAL_REACTANTS COMBINE TO FORM A 
    SINGLE PRODUCT"
    DECOMPOSITION_REACTION = "A SINGLE REACTANT SEPARATED INTO PARTS OR PIECES 
    AND IS MADE INTO TWO OR MORE CHEMICAL_PRODUCTS"
    SINGLE_REPLACEMENT_REACTION = "ONE ELEMENT REPLACES ANOTHER ELEMENT IN A 
    COMPOUND"
    DOUBLE_REPLACEMENT_REACTION = "TWO COMPOUNDS EXCHANGE CHEMICAL_IONS 
    TO FORM TWO NEW COMPOUNDS"
    NET_ELECTRICAL_CHARGE = "IS A CHARGE THAT IS A RESULT FROM THE DECREASE OR 
    INCREASE OF ELECTRONS"
    CHEMICAL_ION = "IS AN ATOM OR CHEMICAL_MOLECULE THAT HAS A 
    NET_ELECTRICAL_CHARGE"
    CHEMICAL_MOLECULE = "IS A GROUP OF TWO OR MORE ATOMS HELD TOGETHER BY 
    CHEMICAL_BONDS"
    CHEMICAL_BOND = "IS AN ATTRACTION BETWEEN ATOMS THAT ALLOWS TO HAPPEN THE 
    FORMING OF CHEMICAL_MOLECULES AND OR OTHER CHEMICAL_STRUCTURES"
    CHEMICAL_BONDS = "MORE THAN ONE CHEMICAL_BOND"
    CHEMICAL_MOLECULES = "MORE THAN ONE CHEMICAL_MOLECULE"
    CHEMICAL_IONS = "MORE THAN ONE CHEMICAL_ION"
    CHEMICAL_REACTIONS = "MORE THAN ONE CHEMICAL_REACTION"
    COMPOUNDS = "MORE THAN ONE COMPOUND"
    QUANTUM_MECHANICS = "IS THE FOUNDATION OF ALL QUANTUM_PHYSICS INCLUDING 
    QUANTUM_CHEMISTRY AND OR QUANTUM_FIELD_THEORY AND OR 
    QUANTUM_TECHNOLOGY AND OR QUANTUM_INFORMATION_SCIENCE"
    REACTANTS = "MORE THAN ONE REACTANT"
    CHEMICAL_REACTION = "IS A PROCESS IN WHICH ONE OR MORE SUBSTANCE CAN BE 
    PROCESSED TO TRANSFORM INTO ONE OR MORE NEW SUBSTANCE"
    ATTRACTION = "TO BE ATTRACTED TO SOMEONE OR SOMETHING"
    COMPOUND = "A THING THAT IS COMBINED WITH TWO OR MORE SEPARATE PARTS OR 
    PIECES OR ELEMENTS"
    INTELLECTUAL = "SOMEONE OR SOMETHING HAVING A HIGHLY DEVELOPED INTELLECT"
    EMOTIONAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE EMOTIONAL CONNECTION THAT TWO PEOPLE FEEL FOR EACH OTHER"
    PHYSICAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE PHYSICAL APPEARANCE OF THE BODY OF A PERSON AND OR ENTITY AND OR 
    BEING2"
    INTELLECTUAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE INTELLECTUAL CONNECTION THAT TWO PEOPLE AND OR ENTITIES THINK 
    ABOUT EACH OTHER"
    SPIRITUAL_ATTRACTION = "IS A TYPE OF ATTRACTION THAT FORMS UNDERSTANDING 
    WITH THE SPIRITUAL CONNECTION THAT TWO PEOPLE SHARE WITH EACH OTHER"
    OCCASIONS = "MORE THAN ONE OCCASION"
    COMPLEX = "SOMETHING FILLED WITH MANY DIFFICULT TO UNDERSTAND PARTS OR 
    PIECES OR FUNCTIONS OR MATERIALS"
    QUANTUM_INFORMATION_SCIENCE = "IS A THEORY THAT STUDIES THE CAPABILITY TO 
    USE AND MAKE POSSIBLE APPLYING QUANTUM_MECHANICS TO 
    INFORMATION_PROCESSING AND COMMUNICATION"
    PROBABILITY_THEORY = "IS A THEORY THAT IS USED FOR THE ANALYSIS OF RANDOM 
    PHENOMENA"
    PROBABILITY_STATISTICS = "IS USED TO ESTIMATE THE LIKELY CHANCE THAT AN EVENT 
    CAN OR IS GOING TO HAVE AN OCCURANCE"
    PROBABILITY_GAME_THEORY = "IS USED TO ANALYZE THE INTELLIGENT AND WELL 
    PLANNED DECISIONS MADE BY PLAYERS AND OR ARTIFICIAL INTELLIGENCE SYSTEMS 
    WHEN LUCKINESS OR CHANCE AND OR OF SKILL AND ACCURACY OR GOOD_LUCK 
    EXISTS"
    SPECIAL_RELATIVITY = "IS A THEORY THAT DESCRIBES HOW SPACE2 AND TIME2 ARE 
    LINKED FOR OBJECTS3 THAT ARE MOVING AT A SPECIFIC SPEED FOR A TIMEFRAME 
    WHILE IN A STRAIGHT LINE"
    PHYSICS = "IS A CLASS THAT CONCERNED WITH THE NATURE AND PROPERTIES OF 
    MATTER AND ENERGY"
    QUANTUM_PHYSICS = "IS A SUBCLASS WITHIN PHYSICS THAT IS USED TO PROVIDE A 
    DESCRIPTION OF QUANTUM_MECHANICS"
    ENUMERATION_ELEMENT = "IS A MEMBER OF AN ENUMERATION"
    ENUMERATION_MEMBER = "IS A NAMED CONSTANT THAT IS WITHIN A ENUMERATION 
    THAT IS EITHER A VARIABLE OR FUNCTION OR OTHER DATA2 STRUCTURE THAT IS PART OF 
    A CLASS AND CAN BE ACCESSIBLE2 TO OTHER ENUMERATION_MEMBERS OF THE CLASS 
    BUT NOT THE CODE OUTSIDE OF THE CLASS THAT THE ENUMERATION_MEMBER EXISTS 
    WITHIN"
    ENUMERATION_MEMBERS = "MORE THAN ONE ENUMERATION_MEMBER"
    INFORMATION_PROCESSING = "IS THE MANIPULATION OF DATA2 TO PRODUCE AN 
    DISPLAY FULL OF DEFINING LOGIC AND EFFICIENT REASONING"
    PROCESSING_CYCLE_INPUT = "IS TO INPUT THE INFORMATION TO BE PROCESSED"
    PROCESSING_CYCLE_PROCESSING = "IS TO PROCESS THE INFORMATION"
    PROCESSING_CYCLE_OUTPUT = "IS TO DISPLAY THE PROCESSED INFORMATION"
    PROCESSING_CYCLE_STORAGE = "IS TO STORE THE PROCESSED INFORMATION INSIDE A 
    SPECIFIC LOCATION"
    CHEMICAL_STRUCTURE = "IS THE SPATIAL2 ARRANGEMENT OF ATOMS INSIDE A 
    CHEMICAL_MOLECULE"
    RELATIVE_POSITION = "IS THE POSITION OF AN OBJECT3 OR POINT WITHIN RELATION TO 
    ANOTHER OBJECT3 OR POINT AND IS THE MEASURE OF THE DISTANCE BETWEEN TWO 
    OBJECTS3 OR POINTS ARE AND IN WHAT DIRECTION THEY ARE FROM EACH OTHER"
    PERCENTAGE2 = "IS A NUMBER OR RATE THAT IS USED TO EXPRESS A CERTAIN PART OF 
    SOMETHING AS A SPECIFIC AMOUNT OF SOMETHING WHOLE WITH A SPECIFIC NUMBER 
    AS A VALUE"
    CHEMICAL = "A COMPOUND OR SUBSTANCE THAT HAS BEEN PREPARED"
    ENUMERATION_ELEMENTS = "MORE THAN ONE ENUMERATION_ELEMENT"
    CHEMICAL_STRUCTURES = "MORE THAN ONE CHEMICAL_STRUCTURE"
    QUANTUM_TECHNOLOGY = "IS A FIELD WITHIN QUANTUM_MECHANICS THAT USES THE 
    CAPABILITIES OF THE PRINCIPLES THAT MAKE QUANTUM_MECHANICS TO MAKE NEW 
    SOFTWARE AND OR HARDWARE AND OR PROGRAMS"
    ANGLE = "IS WHEN TWO LINES MEET AT A POINT AND IS MEASURED BY THE AMOUNT OF 
    TURN BETWEEN THE TWO LINES"
    ANGLES = "MORE THAN ONE ANGLE"
    ROTATIONAL_DEGREE = "A UNIT OF MEASURE FOR ANGLES TO MEASURE THE AMOUNT 
    OF ROTATION OF AN OBJECT3 ABOUT A POINT THAT IS NOT ABLE TO BE CHANGED"
    VECTOR_TRACING = "IS THE PROCESS OF CREATING A VECTOR GRAPHIC FROM AN 
    EXISTING IMAGE"
    TRLINEAR_INJECTION = "A TECHNIQUE FOR INSERTING DATA2 INTO A 
    THREEDIMENSIONAL TEXTURE AND IS MADE POSSIBLE BY TAKING THREE VECTORS EACH 
    REPRESENTING A POINT ON THE TEXTURE AND USING THEM TO CALCULATE THE 
    PROGRAMMING_VALUE OF ANOTHER POINT ON THE TEXTURE"
    TRLINEAR_VECTOR_SPACE = "IS A VECTOR_SPACE THAT HAS A THREE PATHWAY 
    FUNCTION THAT TAKES THREE VECTORS AND GIVES A SCALAR PROGRAMMING_VALUE 
    AS A RETURN"
    SPATIAL_COGNITION = "THE ABILITY TO MENTALLY MANIPULATE SPATIAL2 INFORMATION 
    AND OR STORE AND GATHER SPATIAL DATA2 WITHIN MEMORY AND OR THE ABILITY TO 
    USE SPATIAL INFORMATION TO SOLVE PROBLEMS AND OR THE ABILITY TO DETECT AND 
    INTERPRET SPATIAL INFORMATION FROM THE ENVIRONMENT"
    SPATIAL_MAPPING = "IS THE PROCESS OF REPRESENTING THE SPATIAL RELATION 
    BETWEEN OBJECTS3 OR FEATURES INSIDE A SPECIFIC LOCATION AND OR REGION AND 
    OR AREA AND OR SPACE2"
    LOCATION_OF_OBJECTS = "CAN BE THE PHYSICAL2 LOCATION OF OBJECTS3 WITHIN THE 
    DOMAIN OF A SPECIFIC LOCATION"
    SPATIAL_RELATIONSHIPS_BETWEEN_OBJECTS = "CAN BE THE DISTANCE BETWEEN 
    OBJECTS3 AS WELL AS THE DIRECTION BETWEEN OBJECTS3"
    SPATIAL_PROPERTIES_OF_OBJECTS = "CAN BE THE SHAPE OF AN OBJECT3 AND OR THE 
    SIZE OF AN OBJECT3 AND OR COLOR OF AN OBJECT3"
    SPATIAL_ANALYSIS = "CAN BE USED TO ANALYZE THE SPATIAL2 RELATION AND 
    CONNECTION BETWEEN OBJECTS3"
    MENTAL_IMAGERY = "IS THE ABILITY TO CREATE A MENTAL REPRESENTATION OF AN 
    OBJECT3 OR EVENT OR CIRCUMSTANCE AND IS A TYPE OF VISUAL REPRESENTATION 
    THAT ALLOWS THINGS TO BE SEEN WITHIN THE VISION OF THE MIND"
    SOLUTIONS = "MORE THAN ONE SOLUTION"
    MENTAL_IMAGERY_WITH_PROBLEM_SOLVING = "CAN BE USED TO VISUALIZE POSSIBLE 
    SOLUTIONS TO PROBLEMS"
    MENTAL_IMAGERY_WITH_CREATIVITY = "CAN BE USED TO GENERATE NEW IDEAS"
    MENTAL_IMAGERY_WITH_LEARNING = "CAN BE USED TO IMPROVE MEMORY AND RECALL 
    INFORMATION"
    MENTAL_IMAGERY_WITH_ENGINEERING = "CAN BE USED TO VISUALIZE THE DESIGN OF 
    OBJECTS3 AND STRUCTURES AND SYSTEMS"
    MENTAL_IMAGERY_WITH_NAVIGATING = "CAN BE USED TO VISUALIZE THE LAYOUT OF A 
    SPACE"
    FIRMWARE = "IS A TYPE OF SOFTWARE STORED INSIDE A HARDWARE DEVICE AND IS 
    USED TO CONTROL THE DEVICE BASIC FUNCTIONS INCLUDING ENERGY MANAGEMENT 
    AND INPUT AND OR OUTPUT CONTROL AND MANAGEMENT AND CONTROL AS WELL AS 
    COMMUNICATION BETWEEN THE SYSTEM HARDWARE AND THE SYSTEM SOFTWARE AND 
    IS STORED WITHIN READ ONLY MEMORY AND CANNOT BE MODIFIED BY THE USER"
    PRACTICES = "MORE THAN ONE PRACTICE"
    PRACTICE = "A BELIEF GIVEN ACTION TO HAPPEN OR TO COME INTO EFFECT BY CHANCE 
    AND OR POSSIBILITY"
    PAST_TENSE = "REFERENCE TO A PAST TENSE THAT IS USED TO REFERENCE A TIME OF 
    ACTION"
    PAST_PARTICIPLE = "REFERENCE TO A SPECIFIC VERB FORM THAT IS USED IN THE PAST"
    TENSE = "FORM OF VERB SYSTEM THAT IS USED TO SHOW TIME AND OR CONTINUATION 
    AND OR COMPLETION OF AN ACTION"
    FUTURE_TENSE = "REFERENCE TO A FUTURE TENSE THAT IS USED TO REFERENCE A TIME 
    OF ACTION"
    FUTURE_PARTICIPLE = "REFERENCE TO A SPECIFIC VERB FORM THAT IS USED IN THE 
    FUTURE"
    class Language_Extension_003_2: 
    ZSECOND = 3
    ZMILLISECOND = .03
    ZMICROSECOND = .0003
    ZNANOSECOND = .000003
    ZMILLISECOND = ZSECOND * 100
    ZMICROSECOND = ZSECOND * 10000
    ZNANOSECOND = ZSECOND * 1000000
    ZMINUTE = ZSECOND * 20
    ZMICROMINUTE = ZSECOND * 10
    ZHOUR = ZMINUTE * 60
    ZDAY = ZHOUR * 30
    ZWEEK = ZDAY * 6
    ZMONTH = ZWEEK * 5
    ZYEAR = ZMONTH * 12
    ZDECADE = ZYEAR * 10
    ZCENTURY = ZDECADE * 10
    ZMILLENIA = ZCENTURY * 10
    ZNANOMETER = .000001
    ZMICROMETER = .0001
    ZMILLIMETER = .01
    ZMETER = 1000
    ZCENTIMETER = 100 
    ZINCH = 10
    ZFOOT = 25
    ZYARD = 30
    ZKILOMETER = 1250
    ZMILE = 2500
    ZHERTZ = .0001
    ZKILOHERTZ = ZHERTZ * 1000
    ZMEGAHERTZ = ZKILOHERTZ * 1000
    ZGIGAHERTZ = ZMEGAHERTZ * 1000
    ZTERAHERTZ = ZGIGAHERTZ * 1000
    ZNIBBLE = 30
    ZBIT = 65
    ZBYTE = ZBIT * 1000 
    ZKILOBIT = ZBIT * 1000 
    ZMEGABIT = ZKILOBIT * 1000
    ZGIGABIT = ZMEGABIT * 1000
    ZTERABIT = ZGIGABIT * 1000
    ZKILOBYTE = ZBYTE * 1000
    ZMEGABYTE = ZKILOBYTE * 1000
    ZGIGABYTE = ZMEGABYTE * 1000
    ZTERABYTE = ZGIGABYTE * 1000
    ZFARENHEIGHT = 30
    ZCELCIUS = 10
    ZKELVIN = ZCELCIUS * 150
    ZTEASPOON = .5
    ZTABLESPOON = ZTEASPOON * 2
    ZPINT = ZTABLESPOON * 6
    ZCUP = ZPINT * 2
    ZQUART = ZCUP * 4
    ZLITER = ZQUART * 4
    ZGALLON = ZQUART * 16 
    class Language_Extension_004_2:
    ABILITY_SYSTEM = "IS AN FRAMEWORK THAT ALLOWS PLAYERS TO USE A VARIETY OF 
    ABILITIES"
    FIXED_CALIBRATED_RATIO = "IS A RATIO OF TWO MEASUREMENTS THAT HAS BEEN 
    DETERMINED TO BE CONSTANT"
    FILE_EXTENSION = "IS A SEQUENCE OF LETTERS AT THE END OF A FILE NAME SEPARATED 
    FROM THE MAIN FILE NAME BY A SYMBOLE"
    EMOTIONAL_FUSION = "IS A TERM USED TO DESCRIBE A TYPE OF RELATION THAT WHICH 
    TWO ENTITIES HAVE BEEN STRONGLY CONNECTED BY EMOTIONS THAT THEY BECOME 
    ONE"
    EMOTIONAL_SYNERGY = "IS THE INTERACTION OF TWO OR MORE ENTITIES EMOTIONS TO 
    PRODUCE A COMBINED EFFECT GREATER THAN THE ADDITION OF THEIR INDIVIDUAL 
    EFFECTS"
    EMOTIONAL_ENLIGHTENMENT = "IS A STATE OF EXISTING THAT WHICH THE ENTITY IS 
    FULLY AWARE OF THEIR EMOTIONS AND HOW THE EMOTIONS AFFECT THE ENTITY"
    EMOTIONAL_CLARITY = "IS THE ABILITY TO RECOGNIZE AND LOCATE AS WELL AS 
    UNDERSTAND AND EXPRESS EMOTIONS WITH PERFECT ACCURACY"
    AURIC_RESPONSE = "IS A REACTION FROM KINETIC ENERGIES FROM A ENTITY"
    BOOK_OF_KNOWLEDGE = "IS A REFERENCE OF CREATED WORK THAT CONTAINS A 
    COLLECTION OF INFORMATION ON A PARTICULAR SUBJECT OR FIELD OF STUDY"
    ARCANE_ENERGY = "IS A TYPE OF MAGICAL ENERGY THAT IS DESCRIBED IN MANY 
    OCCASIONS THAT DOES INVOLVE MAGICAL BEINGS AND OR EXISTENCES"
    QUANTUM_ENERGY = "IS A TERM USED TO DESCRIBE THE ENERGY OF SUBATOMIC 
    PARTICLES"
    RECURSIVE_LANGUAGE = "IS A LANGUAGE THAT CAN BE DESCRIBED USING ITSELF"
    RECURSIVE_MEMORY = "IS A TYPE OF MEMORY THAT ALLOWS THE ENTITY TO STORE AND 
    RECALL INFORMATION BY USING DIV TO DIVIDE THE INFORMATION INTO SMALL AND 
    SMALLER PIECES UNTIL IT IS UNDERSTOOD"
    RECURSIVE_RECALL_PROCESS = "IS A PROCESS OF GATHERING INFORMATION FROM 
    MEMORY USING A REPEATED METHOD TO DIV THE INFORMATION INTO SMALLER AND 
    SMALLER PIECES"
    SEASONAL_BRAINWAVE_FLUX = "IS A TERM USED TO DESCRIBE THE CHANGES WITH 
    BRAINWAVE RESPONSES THAT HAPPEN WITHIN A ZYEAR TIMEFRAME"
    QUANTUM_FORCES = "ARE THE FOUR FUNDAMENTAL FORCES OF NATURE THAT ACT AT 
    THE SUBATOMIC LEVEL2"
    PRIMAL_ENERGY = "IS THE ENERGY FOUND WITHIN ALL LIVING THINGS"
    QUALITIVE_RESEARCH = "IS A STUDY METHOD THAT GATHERS AND USES METHOD TO 
    ANALYZE DATA2 THAT IS NOT USING ANY NUMBERS"
    QUANTITIVE_RESEARCH = "IS A STUDY METHOD THAT HAS FOCUS ON GIVING A SPECIFIC 
    CATEGORY OF THE COLLECTION OF AND ANALYSIS OF DATA2 AND IS FORMED ON 
    THEORY OR NUMBERS"
    QUANTUM_METER = "IS A DEVICE THAT MEASURES THE SETTINGS AND VALUES OF 
    QUANTUM SYSTEMS"
    PRE_GENERATION_COMPLETION_TIME = "IS THE TIME IT TAKES TO GENERATE ALL OF THE 
    NECESSARY DATA2 FOR SOMETHING TO GENERATE"
    PRE_GENERATION_WAIT_TIME = "IS THE TIME IT TAKES FOR SOMETHING TO BE 
    GENERATED BEFORE THE GENERATION CAN HAPPEN"
    META_DATA = "IS DATA2 THAT IS USED TO DESCRIBE OTHER DATA2"
    REGULAR_EXPRESSION = "IS A SEQUENCE OF LETTERS THAT SPECIFIES A LOCATED 
    PATTERN IN WORDS OR CODE"
    TEXTURE_MAP = "IS AN IMAGE THAT IS APPLIED TO A THREEDIMENSIONAL OBJECT3 TO 
    GIVE IT A SURFACE APPEARANCE"
    class Language_Extension_005_2:
    RELAY = "SENDS AND RE SENDS SIGNALS FROM TWO DIFFERENT LOCATIONS"
    IMMERSION = "BECOMING ABSOLUTELY INFLUENCED INTO SOMETHING"
    IMMERSIVE = "TO ABSORB INTO A SPECIFIC CATEGORY OR GENRE WITH DETAILED AND 
    UNDERSTANDING INTO THE FIELD OF INTEREST"
    HYPERVISUAL_DISPLAY_UNIT = "IS A TYPE OF DISPLAY THAT ALLOWS A USER TO 
    CONNECT WITH INFORMATION IN A IMMERSIVE WAY"
    EMOTIONAL_THROUGHPUT = "REFERS TO THE RATE AT WHICH A USER HAS AND IS 
    EXPRESSING EMOTIONS"
    EMOTIONAL_MAGNITUDE = "REFERS TO THE INTENSITY OF THE EMOTIONS THAT A USER 
    CAN FEEL"
    EMOTIONAL_ENERGY = "IS THE ENERGY CREATED BY THE EMOTIONS"
    EMOTIONAL_UNDERSTANDING = "IS THE ABILITY TO RECOGNIZE AND OR GIVE 
    IDENTIFICATION AND UNDERSTAND THE EMOTIONS THAT EXIST FOR A SINGLE USER AS 
    WELL AS THE EMOTIONS OF OTHER USERS"
    HOLOGRAPHIC_DISPLAY_UNIT = "IS A TYPE OF DISPLAY THAT USES MANY HOLOGRAMS 
    OF HOLOGRAPHIC IMAGE FILES AND INFORMATION ONTO A HOLOGRAPHIC DISPLAY 
    LOCATION"
    HOLOGRAPHIC_HYPERVISOR = "IS SOMETHING THAT USES HOLOGRAPHIC SYSTEMS TO 
    MAKE VIRTUAL SYSTEMS AND OR DEVICES"
    DATALAKE = "IS A CENTRAL LOCATION FOR ALL DATA2 WITHIN A SPECIFIC LOCATION 
    THAT IS BOTH STRUCTURED AND NOT STRUCTURED WITHIN THE LOCATION AND CAN 
    STORE ANY TYPE OF DATA2 OF ANY SIZE OR SOURCE"
    HOLOGRAPHIC_DATALAKE = "IS A TYPE OF DATALAKE THAT USES HOLOGRAPHIC 
    SYSTEMS OR DEVICES TO STORE AND MANAGE DATA2"
    HOLOGRAPHIC_FREQUENCY_ANALYZER = "IS A TYPE OF WAVELENGTH ANALYZER THAT 
    USES HOLOGRAPHIC SYSTEMS TO MEASURE THE FREQUENCY WAVELENGTH OF A 
    SIGNAL"
    HOLOGRAPHIC_WAVELENGTH = "IS THE WAVELENGTH OF LIGHT THAT IS USED TO 
    CREATE A HOLOGRAM"
    #
    ADHOC_BARRIER_SYSTEM = "IS A TYPE OF SYSTEM THAT CAN BE USED TO PREVENT 
    DEVICES FROM ALLOWING COMMUNICATION WITH EACH OTHER INSIDE AN 
    ADHOCNETWORK"
    ADHOC_DATA_PROCESSING_DEEP_LEARNING = "IS THE USE OF DEEP_LEARNING 
    TECHNIQUES TO PROCESS DATA2 WITHIN ADHOC DEVICES WITHIN TIME2"
    ADHOC_DATA_PROCESSING = "IS THE PROCESSING OF GATHERED AND ORGANIZED 
    AND ANALYZED DATA2 WITHIN AN ADHOC DEVICE AND OR SYSTEM"
    ADHOC_DELAY_HANDLER = "IS A SOFTWARE COMPONENT THAT IS USED TO DELAY 
    DATA2 COMMUNICATION BETWEEN DEVICES WITHIN AN ADHOC SYSTEM"
    ADHOC_EDGE_COMPUTING_DEEP_LEARNING_NETWORK = "IS A NETWORK OF DEVICES 
    THAT ARE ABLE TO PERFORM DEEP_LEARNING TASKS AT THE BORDER OF THE NETWORK"
    ADHOC_ENCRYPTION_HANDLER = "IS A SOFTWARE COMPONENT THAT IS TO ENCRYPT 
    AND DECRYPT DATA2 WITHIN AN ADHOC SYSTEM"
    ADHOC_EXTENSION_FILES = "ARE FILES THAT ARE USED TO EXTEND THE FUNCTIONING 
    ASPECTS OF A SOFTWARE PROGRAM WITHIN AN ADHOC SYSTEM"
    ADHOC_FREQUENCY_ENCRYPTOR = "IS A DEVICE THAT CAN ENCRYPT DATA2 BEFORE IT 
    IS SENT COMMUNICATION INSIDE AN ADHOC SYSTEM"
    ADHOC_FREQUENCY_PREREQUISITE_COMPATIBILITY_SYSTEM = "IS A SYSTEM THAT IS TO 
    CHECK AND DETERMINE IF ADHOC DEVICES AND ADHOC NETWORKS CAN USE 
    COMMUNICATION BETWEEN EACH KNOWN SYSTEM WITHIN THE LOCATION THAT 
    REQUEST ACCESS BY USING COMPATIBLE FREQUENCIES"
    ADHOC_GEOFENCE = "IS A VIRTUAL BOUNDARY THAT IS CREATED WITHIN A LOCATION"
    ADHOC_GEOLOCATION_FIELD_PARAMETER = "IS USED TO DETERMINE THE LOCATION 
    OF DEVICES WITHIN THE ADHOC NETWORK"
    ADHOC_IDE_INTERFACE = "IS A SOFTWARE PROGRAM THAT ALLOWS USERS TO DEVELOP 
    AND FIND MANY ERROR VALUES INSIDE SOFTWARE PROGRAMS WITHIN AN ADHOC 
    NETWORK OR MANY ADHOC NETWORKS"
    ADHOC_INPUT_PATH = "ARE A TYPE OF INPUT PATH THAT ARE USED TO ENTER TEXT OR 
    USED TO SELECT OBJECTS FROM A LIST"
    ADHOC_INSTALLED_PROGRAMS = "ARE PROGRAMS THAT ARE EXISTING ON AN ADHOC 
    COMPUTER AND OR DEVICES OR SYSTEMS"
    ADHOC_LOCAL_AREA_NETWORK = "IS A TYPE OF NETWORK THAT IS CREATED BY 
    CONNECTING TWO OR MORE ADHOC DEVICES TOGETHER"
    ADHOC_LOCATION_HANDLER = "IS A SOFTWARE COMPONENT THAT IS USED TO LOCATE 
    AND FIND THE LOCATION OF ADHOC SYSTEMS WITHIN A ADHOC NETWORK"
    ADHOC_OUTPUT_PATH = "IS AN ADHOC LOCATION WHERE OUTPUT IS STORED"
    ADHOC_PARAMETER_PREREQUISITE_RECOGNITION = "IS THE PROCESS OF 
    DETERMINING THE POSSIBILITY OF TWO OR MORE ADHOC DEVICES HAVING THE 
    REQUIRED PARAMETERS TO ALLOW COMMUNICATE WITH EACH DEVICE"
    ADHOC_EXTENSIONS = "ARE EXTENSIONS THAT ARE NOT A PART OF THE NORMAL 
    PROGRAM OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_PATHS = "IS A PATH THAT IS NOT PART OF THE STANDARD 
    DOWNLOAD OF A PROGRAM"
    ADHOC_RECOGNIZED_PATH_DATA = "IS A TYPE OF DATA2 THAT IS USED TO EXTEND THE 
    FUNCTIONING ASPECTS OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_PROFILE_PATH = "IS A PATH NOT PART OF THE STANDARD 
    SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_UI_PATHS = "ARE PATHS THAT ARE NOT PART OF THE STANDARD 
    USERINTERFACE OF A SOFTWARE PROGRAM"
    ADHOC_RECOGNIZED_VARIABLE_PATHS = "CAN BE RECOGNIZED AS THE PATH 
    CONTENT AND THE NUMBER OF PATHS THAT EXIST AS VARIABLES WITHIN AN ADHOC 
    SYSTEM"
    ADHOC_SHORT_DISTANCE_RELAY_HANDLER = "IS A SOFTWARE COMPONENT THAT IS 
    USED TO RELAY DATA2 BETWEEN DEVICES INSIDE AN ADHOC SYSTEM"
    ADHOC_TRANSPARENCY_FEEDBACK = "IS A PROCESS OF PROVIDING A RESPONSE TO 
    USERS ABOUT THE DECISIONS THAT ARE MADE BY AN ADHOC SYSTEM OR ADHOC 
    SYSTEMS"
    ADHOC_VIRTUAL_DIGITAL_SYSTEM = "IS A SOFTWARE THAT CREATES A VIRTUAL BARRIER 
    BETWEEN TWO OR MORE DEVICES"
    ADHOC_VOICE_COMMAND_FILE_SYSTEM_ACTIVATOR = "IS A SOFTWARE PROGRAM THAT 
    ALLOWS USERS TO CONTROL ADHOC DEVICES USING VOICE COMMAND"
    PHRASE = "A SEQUENCE OF WORDS THAT MAKES A SENTENCE PATTERN FROM 
    RECOGNIZED WORDS INSIDE THE SENTENCE"
    ADHOC_VOICE_COMMAND_PHRASE_INPUT = "IS A PHRASE THAT IS USED TO CONTROL 
    AN ADHOC DEVICE USING VOICE COMMAND"
    ADHOC_VOICE_COMMAND_SYSTEM = "IS AN SYSTEM THAT ALLOWS AN USER TO 
    CONTROL ADHOC DEVICES USING VOICE COMMAND"
    WIRELESS_POWER_TRANSFER = "IS A DEVICE THAT ALLOWS FOR THE TRANSFER OF 
    ENERGY BETWEEN TWO DEVICES WITHOUT THE REQUIRE FOR PHYSICAL INTERACTION"
    ADHOC_WIRELESS_CHARGING_STATION = "IS A DEVICE THAT USES 
    WIRELESS_POWER_TRANSFER TO CHARGE WIRELESS DEVICES"
    ADHOC_VOICE_COMMAND_ORIGIN_POINT_INDICATOR = "IS A SYSTEM THAT ALLOWS AN 
    USER TO RECOGNIZE THE SOURCE OF A VOICE COMMAND"
    NEURAL_NETWORK = "IS A TYPE OF ALGORITHM THAT CAN BE USED TO LEARN AND 
    ADAPT TO CHANGING CONDITIONS"
    NEURAL_NETWORKS = "MORE THAN ONE NEURAL_NETWORK"
    ARTIFICIAL_NEURAL_NETWORK_HOLOGRAPHIC_ADHOC_NETWORK = "IS A TYPE OF 
    WIRELESS NETWORK THAT USES ARTIFICIAL NEURAL_NETWORKS TO CONTROL THE 
    NETWORK ROUTE"
    BARYON_FREQUENCY_CHAMBER = "IS AN DEVICE THAT CAN BE USED TO IMPROVE THE 
    QUALITY AND PERFORMING FUNCTIONS OF SERVICE RECOGNITION SYSTEMS WITHIN 
    AN ADHOC NETWORK"
    #
    ARTIFICIAL_GENERAL_INTELLIGENCE = "IS A TYPE OF ARTIFICIAL INTELLIGENCE2 THAT 
    HAS THE ABILITY TO PERFORM ANY TASK INTELLIGENTLY"
    ARTIFICIAL_SUPER_INTELLIGENCE = "IS A TYPE OF ARTIFICIAL INTELLIGENCE THAT CAN 
    HAVE THE ABILITY TO PERFORM ANY TASK WITH GREATER QUALITY AND EFFICIENT SKILL 
    THAN AN ARTIFICIAL_GENERAL_INTELLIGENCE"
    #
    AURA_RESPONSE_RECOGNITION_PERSONALITY_READER = "IS AN DEVICE THAT CAN 
    READ THE AURA OF A PERSON AND RECOGNIZE WITH GREAT QUALITY THE PERSONALITY 
    TRAITS OF A PERSON"
    #
    COMPLEXITY_BREAKDOWN_BY_SIMPLE_STANDARDS = "IS A PROCESS OF BREAKING 
    DIFFICULT SYSTEMS INTO SMALLER MORE ABLE TO BE MANAGED PARTS AND 
    IDENTIFYING THE DIFFERENT COMPONENTS OF THE SYSTEM AND THE INTERACTION 
    THAT HAPPENS WITHIN THE SYSTEM"
    CYBER_FLUX_FORESIGHT = "IS THE ABILITY TO ANALYZE AND PREPARE FOR THE FUTURE 
    EVENTS"
    DATA_RECOGNITION_MAGNITUDE = "IS A MEASURE OF THE ACCURACY OF AN ARTIFICIAL 
    INTELLIGENCE SYSTEM AND THE ABILITY TO RECOGNIZE DATA2"
    DATA_RECOGNITION_THRESHHOLD = "IS THE SMALLEST LEVEL OF ACCURACY WITH 
    POSSIBILITY THAT AN ARTIFICIAL INTELLIGENCE SYSTEM DOES REQUIRE BEFORE IT CAN 
    RECOGNIZE DATA2 WITH CONFIDENT MEASUREMENTS"
    DOCUMENT_RECOGNITION_BY_CONTENTS_WITHIN_THE_DOCUMENT = "IS A PROCESS 
    TO ANALYZE AND DECRYPT INFORMATION FROM A DOCUMENT BY ALLOWING TO 
    REFERENCE THE INFORMATION DETERMINED BY THE DOCUMENT CONTENT"
    EMOTIONAL_ESSENSE = "IS THE ADDED TOTAL OF ALL EMOTIONS WITHIN AN INDIVIDUAL 
    AND IS WHAT MAKES EACH ENTITY UNIQUE AND WHAT ALLOWS ENTITIES TO CONNECT 
    WITH OTHERS WITH A MORE UNDERSTOOD LEVEL"
    EMOTIONAL_PRESENCE = "IS THE ABILITY TO BE FULLY PRESENT IN THE EVENT AND TO 
    BE AWARE OF YOUR OWN AND OTHER EMOTIONS FROM OTHER PEOPLE"
    FREQUENCY_NETWORK_CONDITIONER = "IS A DEVICE THAT CAN BE USED TO IMPROVE 
    THE PERFORMING OF WIRELESS NETWORKS AND FUNCTION BY GIVING AN ATTEMPT TO 
    FILTER OUT NOT NEEDED FREQUENCIES AND INCREASING THE INTENSITY OF THE 
    DESIRED FREQUENCIES"
    FREQUENCY_PATTERN_OF_DATA = "IS THE WAY IN WHICH THE DATA2 VALUES ARE 
    DISTRIBUTED AND CAN BE USED FOR IDENTIFYING PATTERNS WITHIN THE DATA2"
    FREQUENCY_PROTOCOL_RATIO_CALIBRATOR = "IS A DEVICE THAT CAN BE USED TO 
    MEASURE AND ADJUST THE FREQUENCY RATIO OF DIFFERENT CONDITIONS AND 
    COMMANDS INSIDE AN ADHOC SYSTEM"
    FREQUENCY_STABILIZER_SERVICE = "IS A DEVICE THAT CAN BE USED TO IMPROVE THE 
    STABILITY OF WIRELESS ADHOC NETWORKS AND FUNCTIONS BY SEARCHING 
    RECOGNIZED FLAW DATA2 AND HAVING THE DATA2 CORRECTED TO THE REQUIRED 
    CALIBRATION MEASUREMENT NEEDED"
    GAME = "CAN INCLUDE SIMULATION OR RE THE SIMULATION OF VARIOUS EVENTS OR 
    USE AND WITHIN REALITY FOR VARIOUS REQUIREMENTS AND OR OF NEEDS"
    #TWO_POINT_FIVE_DIMENSION_TERRAIN_BRUSH = "IS A TOOL THAT ALLOWS THE USER 
    TO CREATE TWO_POINT_FIVE_DIMENSIONAL TERRAIN WITHIN A ENGINE
    #TWO_POINT_FIVE_DIMENSIONAL_TERRAIN = "IS A TYPE OF TERRAIN THAT IS DISPLAYED 
    WITH TWO_DIMENSIONAL ELEMENTS BUT HAS SOME THREEDIMENSIONAL ELEMENTS
    #WORLDBUILDINGNIGHTTIME = "A TIMEFRAME THAT BEGINS WITHIN THE EVENING 
    HOURS AND DOES CONTINUE UNTIL THE MORNING TIMEFRAME WHICH IS A 
    WORLDBUILDINGDAYTIME TIMEFRAME
    #WORLDBUILDINGDAYTIME = "A TIMEFRAME OF TIME_OF_DAY THAT BEGINS WITHIN THE 
    HOURS OF MORNINGTIMEFRAME AND DOES CONTINUE UNTIL EVENING TIMEFRAME 
    WHICH IS A WORLDBUILDINGNIGHTTIME TIMEFRAME
    #WORLDBUILDINGSEASON = "A TIMEFRAME THAT IS CLASSIFIED INTO FOUR 
    CATEGORIES THAT HAS DIFFERENT TEMPERATURE FOR THE ENVIRONMENT AND 
    DIFFERENT CONDITIONS FOR THE INDIVIDUAL SEASON WITH EACH SEASON KNOWN AS 
    WINTER AND SUMMER AND SPRING AND AUTUMN
    #WORLDBUILDINGMOONPHASE = "IS THE DIFFERENT CHANGES WITH THE MOON AND 
    ITS PHASE CHANGES BETWEEN THE EIGHT MOON PHASES
    #WORLDBUILDINGPLANET = "
    #MORNINGTIMEFRAME = "A TIMEFRAME THAT BEGINS AT .20833 AND DOES END 
    AT .75000
    #NIGHTTIMEFRAME = "A TIMEFRAME THAT BEGINS AT .75000 AND DOES END AT .20833
    class Language_Extension_006_2:
    QUICK = "TO FORM ACTION WITH GREAT SPEED"
    SLOW = "TO FORM ACTION WITH A SMALL AMOUNT OF SPEED"
    FAST = "QUICK AT FORMING A REACTION OR MAKING AN ACTION"
    CIRCUMSTANCES = "MORE THAN ONE CIRCUMSTANCE"
    FOCUSING = "MAKING AN ACTION TO FOCUS AND CONCENTRATE ON A SPECIFIC TOPIC 
    AND OR FIELD AND OR AREA OF FOCUS"
    SIMILARITY = "IS A THING OR IDEA2 THAT TWO OR MORE THINGS HAVE AS THE SAME OR 
    SIMILAR TO"
    COMPARISON = "TO COMPARE TWO OR MORE THINGS TOGETHER"
    SIMILARITIES = "MORE THAN ONE SIMILARITY OR COMPARISON"
    COMPARING = "IS THE ACTION OF IDENTIFYING THE SIMILARITIES AND DIFFERENCES 
    BETWEEN TWO OR MORE THINGS"
    REPRESENTED = "EXPLAINED AS OR DETERMINED AS"
    CONVERTING = "PROCESSING AS AN ACTION TO CONVERT"
    INTERSECT = "WHEN TWO LINES ARRIVE AT OR GO THROUGH A SINGLE ORIGIN 
    LOCATION" 
    ZERO_TO_ONE_HUNDRED_PERCENTAGE = "AN AMOUNT BETWEEN ZERO AND 
    ONEHUNDRED PERCENT"
    DEGREE = "AN AMOUNT BETWEEN ZERO AND THREEHUNDREDSIXTY"
    DEGREES = "MORE THAN ONE DEGREE"
    RIGHT_ANGLE = "IS AN ANGLE OF NINETY DEGREES"
    PERPENDICULAR = "IS TWO THINGS THAT INTERSECT AT A RIGHT_ANGLE" 
    RECOGNIZING = "BEGINNING TO RECOGNIZE SOMETHING OR SOMEONE"
    IDENTIFY = "TO LOCATE AND RECOGNIZE"
    ALIGNING = "MAKING AN ACTION TO ALIGN SOMETHING"
    ALIGNMENT = "IS THE ACTION OF ALIGNING THINGS IN A STRAIGHT LINE OR IN A 
    CERTAIN DIRECTION"
    OBSERVED = "ANALYZED WHILE LOCATED"
    CLASSIFYING = "TO GIVE A CATEGORY AND THEN GROUP TOGETHER SPECIFIC DATA2"
    PAIR = "A TOTAL OF TWO OF SOMETHING"
    FIXED = "CANNOT BE CHANGED"
    DISTANCES = "MORE THAN ONE DISTANCE"
    MULTIDIMENSIONAL_SPACE = "IS A SPACE2 THAT HAS MORE THAN TWO DIMENSIONS" 
    GRID = "IS A DATA2 STRUCTURE THAT IS USED TO RECOGNIZE DATA2 THAT IS ORGANIZED 
    IN A MULTIDIMENSIONAL_SPACE"
    ADHOC_CARDINAL_ARRAY = "IS A DATA2 STRUCTURE THAT STORES A COLLECTION OF 
    ELEMENTS IN A SORTED ORDER"
    ADHOC_EUCLIDEAN_GRID = "IS A DATA2 STRUCTURE THAT STORES A COLLECTION OF 
    POINTS IN A TWO DIMENSIONAL SPACE2"
    NUMERICAL = "RELATING TO OR EXPRESSED IN NUMBERS"
    CARTESIAN_COORDINATE_SYSTEM = "IS A COORDINATE SYSTEM THAT SPECIFIES EACH 
    POINT IN A UNIQUE WAY BY A PAIR OF NUMERICAL COORDINATES WHICH ARE THE 
    ASSIGNED DISTANCES FROM THE POINT TO TWO FIXED PERPENDICULAR LINES"
    CARTESIAN_COORDINATE = "IS A PAIR OF NUMBERS THAT IN A UNIQUE WAY CAN 
    SEARCH FOR TO FIND A POINT WITHIN A PLANE"
    GEOGRAPHIC = "REFERS TO ANYTHING SIMILAR TO THE PHYSICAL FEATURES OF THE 
    ENVIRONMENT AND ITS SURFACE"
    GEOFENCE = "IS A VIRTUAL PERIMETER AROUND A GEOGRAPHIC AREA THAT CAN BE 
    USED TO ANALYZE THE LOCATIONS THE MOVEMENT OF PEOPLE OR OBJECTS" 
    LATITUDE = "IS A GEOGRAPHIC COORDINATE THAT SPECIFIES THE UPWARD OR 
    DOWNWARD POSITION OF A POINT ON THE ENVIRONMENTS SURFACE"
    LONGITUDE = "IS A GEOGRAPHIC COORDINATE THAT SPECIFIES THE RIGHT OR LEFT 
    POSITION OF A POINT ON THE ENVIRONMENT AND ITS SURFACE"
    CARTESIAN_COORDINATES = "MORE THAN ONE CARTESIAN_COORDINATE"
    STATISTICAL = "REFERS TO ANYTHING SIMILAR TO THE COLLECTION OF OR ANALYSIS OF 
    OR UNDERSTANDING AND COMPREHENDING OF AND PRESENTATION OF DATA2 AND 
    CAN ALSO REFER TO THE METHODS USED TO DO SUCH2 ACTIONS"
    EUCLIDEAN_GRID = "IS A TYPE OF GRID THAT IS USED TO COMPARE OR ASSIGN DATA2 
    THAT IS IN A MULTIDIMENSIONAL_SPACE"
    ADHOC_CARTESIAN_POINT = "IS AN EUCLIDEAN_GRID IS A POINT IN A TWO 
    DIMENSIONAL SPACE2 THAT IS REPRESENTED BY ITS COORDINATES IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_MINIMUM_RATIO_EXCHANGE_BETWEEN_CARTESIAN_VALUES = "IS AN 
    ALGORITHM THAT CAN BE USED TO FIND THE MINIMUM RATIO BETWEEN TWO POINTS IN 
    A CARTESIAN_COORDINATE_SYSTEM"
    CARTESIAN_ADHOC_NETWORK = "IS A TYPE OF WIRELESS NETWORK THAT IS CREATED 
    BY DEVICES THAT ARE CONNECTED TO EACH OTHER IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_CARTESIAN_GEOFENCE = "IS A TYPE OF GEOFENCE THAT IS DEFINED IN A 
    CARTESIAN_COORDINATE_SYSTEM"
    ADHOC_OFFLINE_GPS_CARTESIAN_COORDINATE_SYSTEM_USING_LATITUDE_AND_LON
    GITUDE = "IS A SYSTEM THAT CAN BE USED TO DEFINE COORDINATES IN A 
    CARTESIAN_COORDINATE_SYSTEM AND IS DONE BY FIRST CONVERTING THE LATITUDE 
    AND LONGITUDE COORDINATES TO CARTESIAN_COORDINATES USING A FORMULA"
    ADHOC_CARTESIAN_GAME_WORLD = "IS A GAME WORLD THAT IS DEFINED USING A 
    CARTESIAN_COORDINATE_SYSTEM"
    FREQUENCY_ADJUSTMENT_DIAGNOSTICS = "IS A STATISTICAL TECHNIQUE THAT CAN BE 
    USED TO IDENTIFY AND CORRECT PROBLEMS WITH FREQUENCY DATA2 AND FUNCTIONS 
    BY COMPARING THE OBSERVED FREQUENCIES OF DATA2 POINTS TO THE EXPECTED 
    FREQUENCIES"
    FREQUENCY_PATTERN = "IS A WAY OF DESCRIBING THE DISTRIBUTION OF DATA2 POINTS 
    IN A SET AND CAN BE USED TO IDENTIFY PATTERNS IN THE DATA2"
    PATTERN_RESPONSE_TIME = "REFERS TO THE AMOUNT OF TIME2 IT TAKES FOR A DEVICE 
    TO PROCESS A REQUEST THAT FOLLOWS A SPECIFIC PATTERN"
    PATTERN_FREQUENCY = "REFERS TO THE NUMBER OF TIMES A SPECIFIC PATTERN 
    HAPPENS IN A SET OF DATA2"
    PATTERN_RANGE = "REFERS TO A RANGE OF VALUES THAT ARE APPROVED FOR A 
    SPECIFIC PATTERN"
    PATTERN_CONTEXT = "REFERS TO THE ENVIRONMENT IN WHICH A SOFTWARE PATTERN 
    IS USED"
    PATTERN_COMPLEXITY = "IS A MEASURE OF THE DIFFICULTY OF RECOGNIZING A 
    PATTERN"
    PATTERN_FIELD = "IS A DATA2 FIELD THAT STORES A PATTERN"
    ATTRIBUTES = "MORE THAN ONE ATTRIBUTE"
    ITEM = "IS A USABLE OBJECT3 THAT HAS SPECIAL ATTRIBUTES"
    ITEMS = "MORE THAN ONE ITEM"
    FINDING = "SCANNING TO LOCATE SOMETHING OR SOMEONE"
    GRAPH = "IS A COLLECTION OF POINTS CONNECTED BY LINES"
    GRAPHS = "MORE THAN ONE GRAPH"
    SEQUENCES = "MORE THAN ONE SEQUENCE"
    TEXT_CLASSIFICATION = "IS THE TASK OF CLASSIFYING TEXT INTO DIFFERENT 
    CATEGORIES"
    OBJECT_RECOGNITION = "IS THE TASK OF IDENTIFYING OBJECTS IN IMAGES"
    GRAPH_PATTERN_MINING = "IS THE TASK OF FINDING PATTERNS IN GRAPHS"
    SEQUENTIAL_PATTERN_MINING = "IS THE TASK OF FINDING SEQUENCES OF ITEMS IN 
    DATA2"
    GEOSPATIAL_PATTERN = "IS A PATTERN THAT CAN BE OBSERVED IN DATA2 THAT HAS A 
    SPATIAL2 COMPONENT"
    TEMPORAL_FREQUENCY = "REFERS TO THE NUMBER OF TIMES A REPEATING EVENT 
    HAPPENS IN A GIVEN UNIT OF TIME2 AND IS THE FREQUENCY OF A SIGNAL AS IT 
    CHANGES OVER TIME2"
    TEMPORAL_ALIGNMENT = "REFERS TO THE PROCESS OF ALIGNING TWO SIGNALS IN 
    TIME2"
    FILLED = "SOMETHING FULL OF SOMETHING EVEN IF AS A PERCENTAGE OR COMPLETE 
    AMOUNT"
    INSPIRED = "IS TO BE FILLED WITH THE NEED TO CREATE OR DO SOMETHING" 
    PREDICTION = "AN ANALYZED ESTIMATE AS AN ANSWER TO A SOLUTION AND OR 
    PROBLEM"
    PREDICTIONS = "MORE THAN ONE PREDICTION"
    FEEDBACK = "IS INFORMATION ABOUT THE RESULTS OF AN ACTION OR PROCESS"
    MIMIC = "DUPLICATE THE ACTION OF SOMEONE OR SOMETHING"
    MIMICS = "TO MIMIC THE ACTIONS OR RESPONSES OF SOMETHING OR SOMEONE"
    MATHEMATICAL_CONCEPT = "IS THE IDEAS AND PRINCIPLES THAT ARE USED TO SOLVE 
    PROBLEMS AND TO MAKE PREDICTIONS"
    MATHEMATICAL_CONCEPTS = "MORE THAN ONE MATHEMATICAL_CONCEPT"
    CENTRAL_NERVOUS_SYSTEM = "IS REQUIRED INFORMATION FOR PROCESSING 
    INFORMATION FROM THE SENSORY FUNCTIONS"
    BIOLOGICAL_NEURON = "IS A CELL IN THE CENTRAL_NERVOUS_SYSTEM THAT CAN 
    RECEIVE AND SENDS SIGNALS TO OTHER CELLS"
    BIOLOGICAL_NEURONS = "MORE THAN ONE BIOLOGICAL_NEURON"
    NEURON = "IS A UNIT OF COMPUTATION THAT IS INSPIRED BY THE 
    BIOLOGICAL_NEURON"
    NEURONS = "MORE THAN ONE NEURON"
    TEMPORAL_MULTIDIMENSIONAL_CROSS_REFERENCE_COMMUNICATION = "IS A 
    COMMUNICATION RULE THAT ALLOWS FOR THE EXCHANGE OF DATA2 BETWEEN TWO 
    OR MORE DEVICES OVER TIME2 AND ACROSS MULTIPLE DIMENSIONS"
    CROSS_DIMENSIONALITY_FREQUENCY_FEEDBACK = "IS A FEEDBACK LOOP THAT 
    HAPPENS BETWEEN TWO SIGNALS THAT ARE IN DIFFERENT DIMENSIONS"
    NEURAL_NETWORK_LAYER = "IS A GROUP OF NEURONS THAT ARE CONNECTED 
    TOGETHER AND WORK TOGETHER TO PERFORM A SPECIFIC TASK"
    MULTIDIMENSIONAL_FEEDBACK_LAYER = "IS A TYPE OF NEURAL_NETWORK_LAYER THAT 
    ALLOWS FOR FEEDBACK BETWEEN DIFFERENT DIMENSIONS OF THE INPUT DATA2"
    MATHEMATICAL_MODEL = "IS A REPRESENTATION OF A SYSTEM USING 
    MATHEMATICAL_CONCEPTS AND LANGUAGE"
    MATHEMATICAL_FUNCTION = "IS A RULE THAT CAN ASSIGN A UNIQUE OUTPUT VALUE TO 
    EACH INPUT VALUE"
    ARTIFICIAL_NEURON = "IS A MATHEMATICAL_MODEL THAT IS USED TO SIMULATE THE 
    BEHAVIOR OF A BIOLOGICAL_NEURON OR IS A MATHEMATICAL_FUNCTION THAT MIMICS 
    THE BEHAVIOR OF BIOLOGICAL_NEURONS"
    BINARY_CLASSIFICATION = "IS A TASK OF CLASSIFYING DATA2 POINTS INTO TWO 
    CATEGORIES"
    PERFORMS = "MAKES AN ACTION TO PERFORM SOMETHING"
    NODE = "IS A COMPUTATIONAL UNIT THAT PERFORMS A SPECIFIC FUNCTION"
    PERCEPTRON = "IS A SIMPLE TYPE OF ARTIFICIAL_NEURON THAT CAN BE USED TO 
    PERFORM BINARY_CLASSIFICATION"
    INTERCONNECTED = "CONNECTED OR LINKED TOGETHER"
    NODES = "MORE THAN ONE NODE"
    TRAINED = "SOMEONE OR SOMETHING THAT HAS LEARNED HOW TO COMPLETE 
    SOMETHING"
    ARTIFICIAL_NEURAL_NETWORK = "IS A NETWORK OF INTERCONNECTED NODES THAT 
    CAN LEARN TO PERFORM A TASK BY BEING TRAINED ON DATA2"
    PERCEPTRONS = "MORE THAN ONE PERCEPTRON"
    MULTIPERCEPTRON = "IS A TYPE OF ARTIFICIAL_NEURAL_NETWORK THAT USES MULTIPLE 
    PERCEPTRONS TO SOLVE A PROBLEM"
    CONSISTENT = "DESCRIBES SOMETHING THAT IS NOT CHANGING"
    LOGICAL = "DESCRIBES SOMETHING THAT IS CONSISTENT WITH REASON OR FACT"
    IMAGINATIVE = "DESCRIBES SOMEONE WHO IS ABLE TO CREATE NEW IDEAS OR IMAGES 
    IN THEIR MIND"
    INTERESTED = "DESCRIBES SOMEONE WHO HAS A STRONG DESIRE TO LEARN ABOUT OR 
    DO SOMETHING"
    PUTTING = "AN HAPPENING ACTION TO SET SOMETHING IN A SPECIFIC LOCATION"
    PLACING = "IS THE ACT OF PUTTING SOMETHING IN A PARTICULAR LOCATION"
    SYSTEMATIC = "DESCRIBES SOMETHING THAT IS DONE ACCORDING TO A LIST OF 
    INSTRUCTIONS OR SYSTEM" 
    ARRANGING = "IS THE ACTION OF PLACING THINGS IN A PARTICULAR ORDER OR 
    PATTERN"
    ORGANIZING = "IS THE ACTION OF ARRANGING THINGS IN A SYSTEMATIC WAY"
    DECISIONS = "MORE THAN ONE DECISION"
    class Language_Extension_007_2:
    WIDE_AREA_NETWORK = "IS A NETWORK THAT CAN EXPAND TO A SPECIFIC RANGE OF 
    AREAS WITHIN A SPECIFIC DISTANCE"
    REMOTE_HOST = "IS A DEVICE CONNECTED TO A WIDE_AREA_NETWORK"
    LOCAL_HOST = "IS A DEVICE THAT IS CONNECTED TO AN LAN"
    PERSONAL_HOST = "IS A SMALL DEVICE THAT CAN BE USED BY AN INDIVIDUAL ENTITY"
    HOSTS = "MORE THAN ONE HOST"
    CLIENT_HOST = "IS A GROUP OF ENTITIES THAT REQUEST RESOURCES FROM OTHER 
    HOSTS WITHIN THE NETWORK"
    HOST = "IS A DEVICE THAT IS CONNECTED TO A NETWORK THAT CAN COMMUNICATE 
    WITH OTHER DEVICES WITHIN THE NETWORK"
    HOSTING_SERVICE = "ALLOWS MORE THAN ONE INDIVIDUAL AND GROUP TO MAKE A 
    WEBSITE ACCESSIBLE TO A SPECIFIC NETWORK OR GROUP OF NETWORKS"
    WEB_HOSTING_SERVICE = "IS A TYPE OF NETWORK HOSTING_SERVICE"
    WEB_HOSTING_SERVICES = "MORE THAN ONE WEB_HOSTING_SERVICE"
    SERVER_HOST = "IS A GROUP OF ENTITIES THAT CAN PROVIDE 
    WEB_HOSTING_SERVICES"
    WEBPAGE = "IS A DOCUMENT THAT IS PART OF A WEBSITE"
    WEBPAGES = "MORE THAN ONE WEBPAGE"
    HOSTED = "IS SOMETHING THAT WAS PREVIOUSLY HOST"
    WEB_SERVER = "IS BOTH A COMPUTER THAT HOSTS A WEBSITE AND IS A SOFTWARE 
    PROGRAM THAT SENDS WEBPAGES TO COMPUTERS"
    WEBSITE = "A COLLECTION OF WEBPAGES THAT ARE LINKED TOGETHER AND HOSTED 
    WITHIN A WEB_SERVER"
    WEBSITES = "MORE THAN ONE WEBSITE"
    WEB_HOST = "IS A GROUP OF ENTITIES THAT PROVIDE STORAGE SPACE AND BANDWIDTH 
    FOR WEBSITES"
    WEB_BROWSER = "IS A SOFTWARE PROGRAM MADE TO ALLOW TO ACCESS AND VISIT 
    VIEWABLE WEBSITES"
    CONTROL_PANEL = "IS A GRAPHICUSERINTERFACE THAT ALLOWS A USER TO MANAGE 
    SETTINGS AND FEATURES OF A COMPUTER OR SOFTWARE PROGRAM"
    REMOTE_COMPUTER = "IS A COMPUTER THAT IS NOT PHYSICALLY LOCATED IN THE SAME 
    LOCATION AS THE USER"
    REMOTE_DESKTOP_CONNECTION = "IS A DEVICE THAT ALLOWS A USER TO CONNECT TO 
    A REMOTE_COMPUTER AND CONTROL IT FROM A DIFFERENT LOCATION FROM WHERE 
    THE REMOTE_COMPUTER IS LOCATED"
    WEB_BASED = "DESCRIBES ANYTHING THAT IS ACCESSED BY A NETWORK WHILE USING 
    A WEB_BROWSER"
    REMOTE_SERVER = "IS A COMPUTER THAT IS LOCATED WITHIN A DIFFERENT PHYSICAL 
    LOCATION THAT IS DIFFERENT FROM THE USER AND CAN BE ACCESSED BY A NETWORK 
    USING A REMOTE_DESKTOP_CONNECTION OR A WEB_BASED CONTROL_PANEL"
    EQUIPMENT = "REFERS TO THE TOOLS AND DEVICES THAT ARE USED WITHIN A SPECIFIC 
    GROUP AND OR FIELD"
    LOCAL_HARDWARE = "IS THE PHYSICAL EQUIPMENT THAT IS CONNECTED TO A LAN AND 
    IS LOCATED WITHIN A SINGLE LOCATION"
    CLOUD_GAMING = "IS A IDEA THAT CAN ALLOW A PLAYER TO ACTIVATE A GAME FROM A 
    REMOTE_SERVER RATHER THAN ACTIVATING IT WITHIN LOCAL_HARDWARE"
    BANDWIDTH = "IS THE MAXIMUM AMOUNT OF DATA2 THAT CAN BE RECOGNIZED WITHIN 
    A NETWORK CONNECTION WITHIN A GIVEN AMOUNT OF TIME"
    class Language_Extension_008_2:
    EXTROVERTED_SENSATIONS = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    EXTRAVERTED_INTUITION = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_SENSING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_INTUITION = "IS ONE OF THE FOUR PERSONALITY"
    INTROVERTED_THINKING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS" 
    EXTRAVERTED_THINKING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    INTROVERTED_FEELING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    EXTRAVERTED_FEELING = "IS ONE OF THE FOUR PERSONALITY FUNCTIONS"
    class Language_Extension_009_2:
    STILL = "EXISTING IN PLACE AS WAS BEFORE THE CURRENT MOMENTS"
    MOMENT = "EVENT OR PRESENT CIRCUMSTANCE"
    DESTROYED = "DENIED EXISTENCE AND REMOVED FROM THAT EVENT"
    CREATURE = "IS A SPECIFIC CLASS OF ANIMAL"
    VALIDATION = "SPECIFIED COMMAND TO ACCEPT SOMETHING SPECIFIC" 
    TOOK = "IT IS GRABBED OR TAKEN"
    MOMENTS = "MORE THAN ONE MOMENT"
    COMPREHENSION = "THE ABILITY TO COMPREHEND SOMETHING"
    DENIED = "PREVIOUSLY GIVEN A COMMAND TO DENY SOMETHING FROM HAPPENING"
    REMAINS = "STILL EXISTS"
    FILTH = "DIRTY AND IMPURE AS WELL AS UNDEFILED"
    DESIRES = "MORE THAN ONE DESIRE"
    ORIGINALLY = "WAS THE ORIGINAL THAT WAS EXPECTED TO BE"
    PURIFIES = "AN ACTION THAT CAN PURIFY SOMETHING"
    EXPELS = "TAKES OUT OF AND RELEASES FROM SOMETHING"
    EXPEL = "TAKE OUT OF AND RELEASE FROM SOMETHING"
    TAME = "TO MAKE LISTEN AND FOLLOW INSTRUCTIONS AND OR COMMANDS WITH 
    CORRECT ACTIONS AND OUTCOMES"
    TESTS = "TO ATTEMPT TO USE AN SET OF ACTIONS AND OR EFFECTS TO MAKE AN 
    OUTCOME HAPPEN"
    TRUTHS = "IDEAS MADE FROM FACT OR TRUTH THAT EXISTS FROM SOME POINT OF TIME2 
    OR SPACE2"
    PURIFY = "THE ACTION TO REMOVE ANY IMPURE THOUGHTS AND OR IDEAS AND OR 
    ACTIONS"
    MATHEMATIC = "A RELATION CONNECTING TO THE DEVELOPMENT OF FORMULAS AND 
    OR EQUATIONS AND OR NUMBER VALUES AND OR PARAMETERS"
    COMPREHEND2 = "TO UNDERSTAND COMPLETE IN FULL AND TO TAKE IN THE VALUES 
    AND ACCEPT AS WISDOM2 AND OR KNOWLEDGE2"
    UNDERSTANDS = "UNDERSTANDING THAT IS FORMED WITH COMPREHENDED LOGIC"
    ANIMAL = "CREATURE"
    CLASSIFICATION = "A GROUP OF CATEGORIZED SELECTED CATEGORIES MADE INTO ONE 
    GROUP"
    CLASSIFICATIONS = "MORE THAN ONE CLASSIFICATION"
    ROLE = "A LIST OF INSTRUCTIONS THAT MUST BE MADE BY A SPECIFIC LIST OF ACTIONS"
    SPECIALIZES = "HAVE A GREAT AMOUNT OF UNDERSTANDING AND COMPREHENSION 
    WITHIN A SPECIFIC FIELD AND OR CATEGORY AND OR GENRE AND OR CLASSIFICATION"
    SPECIALIZE = "HAVE A GREAT UNDERSTANDING OF A CERTAIN OR SPECIFIC FIELD HAVE 
    A LARGE AMOUNT OF KNOWLEDGE2 RELATING TO THAT FIELD"
    RECOGNIZING = "THE PROCESS OF UNDERSTANDING RECOGNIZED INFORMATION"
    PRODUCED = "CREATED WITH PURPOSE AND MEANING"
    ACTIVITY = "THE FORMING OF ACTIONS AND INSTRUCTIONS THAT FORM WITHIN AN 
    EVENT"
    VOCALS = "RELATING TO THE EXISTING VOCAL PATTERNS OF THE HUMAN VOICE"
    FUNCTIONALITY = "THE FUNDAMENTAL EFFORT OF MANY FUNCTIONS OR IDEAS GIVEN 
    JUDGEMENT BY ITS CAPABILITY TO COMPLETE ITS TASKS OR TO FUNCTION AS A 
    COMPLETE SYSTEM"
    ARTWORKS = "MORE THAN ONE WORK OF ART COMPLETED"
    ENGINEERS = "MORE THAN ONE ENGINEER"
    CLASSIFIES = "GIVES A DEFINITE CLASS AND OR CATEGORY TO SOMETHING"
    ACCORDINGLY = "TO BE MADE OR COMPLETED AS WAS INTENDED OR AS BY THE 
    INSTRUCTIONS THAT WAS MADE FOR THE CIRCUMSTANCE OR CIRCUMSTANCES"
    VOCALIZATION = "THE ACTION OF PRODUCING A VOCAL RESPONSE"
    VALIDITY = "IS THE PERCENTAGE OF VALIDATION OF SOMETHING"
    LOOKS = "SCANS AND ANALYZES"
    SPECIALIZED = "INTENDED FOR A SPECIFIC FIELD AND OR CATEGORY"
    COULD = "CAPABLE OF POSSIBLE CHANCES THAT IT IS ABLE TO HAPPEN" 
    HANDMADE = "MADE BY THE ACTIONS OF THE BODY AND BY EFFORT ONLY"
    TRADITIONAL = "PASSED DOWN TO FOLLOW FROM EACH FAMILY MEMBER TO EACH NEW 
    CHILD THAT BECOMES AN ADULT"
    CRAFT = "A LIST OF SKILLS BROUGHT TOGETHER TO MAKE SOMETHING FORM OR COME 
    INTO EXISTENCE"
    UNHOLY = "NOT HOLY"
    RUINED = "WHEN SOMETHING HAS BEEN PROCESSED THAT IT HAS BEEN DESTROYED OR 
    MADE POSSIBLE TO NOT HAPPEN AT THAT MOMENT AND CAN NEVER HAPPEN AGAIN 
    FROM THE MOMENT AFTER THAT EVENT TOOK EFFECT"
    RUIN = "DESTROY OR PREVENT SOMETHING FROM HAVING COMPATIBILITY WITH 
    SOMETHING OR PREVENT SOMETHING NOT COMING INTO EFFECT OR TO NOT MAKE 
    SOMETHING POSSIBLE TO HAPPEN"
    DEFILE = "RUIN SOMETHING PURE"
    DEFILED = "RUINED OF MEANING OR DESCRIPTION OR VALUES OR THAT HOLD 
    DEFINITION"
    UNDEFILED = "NOT DEFILED"
    UNTAINTED = "NOT TAINTED"
    RELIGION = "IS A SET OF RULES OR BELIEFS THAT CONNECT TO THE BELIEF OF A HIGHER 
    ENTITY"
    RELIGIOUS = "HOLDING VALUES THAT CONNECT TO RELIGION"
    SOME = "CONTAINING A SMALL AMOUNT OF SOMETHING WHOLE"
    INSTRUCTION = "A RULE OR ORDER TO FOLLOW"
    DISCIPLINE = "A SET OF RULES OR INSTRUCTION THAT SOMEONE INCLUDES WITHIN 
    THEIR BELIEF SYSTEM TO FOLLOW"
    INSTRUCTIONS = "MORE THAN ONE INSTRUCTION"
    STATURE = "THE FORMING AND MEANING OF HELD INSTRUCTIONS OR BELIEFS UPON 
    SOMEONE AS A FORM OF DISCIPLINE"
    HOLY = "OF THE HIGHEST STATURE AND OR NATURE AND OR QUALITY OF SOMETHING 
    OR SOMEONE CREATED WITHIN TIME2 THAT HOLDS THE CAPABILITY TO BE COMPATIBLE 
    FOR SOME FORM OF SPIRITUAL OR RELIGIOUS INSTRUCTION THAT HOLDS TO THE 
    BELIEFS AND OR VIRTUES OF SOMEONE"
    SINLESS = "COMPLETE VOID OF SIN"
    PURE = "SINLESS AND OF HOLY DEFINED MEANING AND OR VIRTUE"
    TAINTED = "A DECREASE OF PURE AND HOLY DESCRIPTION"
    SIN = "IS THE PRODUCTION OF TAINTED AND OR UNHOLY VALUES WITHIN SOMETHING 
    OR SOMEONE"
    SINFUL = "CONTAINING SIN"
    CLEAN = "NOT CONTAINING MORE THAN WHAT WAS ORIGINALLY INTENDED AS A PURE 
    UNDEFILED AND OR UNTAINTED AND OR DIRTY SUBSTANCE"
    IMPURE = "NOT CLEAN AND NOT OF HOLY DESIRES AND HOLDS ACTIONS AND OR 
    THOUGHTS AND OR IDEAS THAT HOLD SINFUL MEANING"
    CLEANSE = "REMOVE SOMETHING IMPURE AND OR SOMETHING NOT CLEAN FROM 
    SOMETHING THAT WAS CLEAN"
    DIRTY = "IS SOMETHING THAT HOLDS A FORM OF FILTH CONNECTED TO IT THAT IS NOT 
    CLEAN BY A SPECIFIC AMOUNT AND REMAINS IMPURE"
    IMPURITY = "SOMETHING THAT HAS SIN CONNECTED TO IT AND IS CONSIDERED IMPURE"
    IMPURITIES = "MORE THAN ONE IMPURITY"
    CLEANSES = "USE OF POWER OR STRENGTH TO CLEANSE ALL FORMS OF IMPURITY 
    FROM SOMETHING"
    SOMEWHERE = "REFERENCE TO A SPECIFIC PLACE OR PLACES"
    AGED = "HAVING A LARGE AMOUNT OF AMOUNT TO IT"
    FRAMEWORKS = "MORE THAN ONE FRAMEWORK"
    CONSTRUCTS = "MAKE AND CREATE SOMETHING"
    DESIGNS = "CREATE AS A DESIGN"
    SEARCHES = "CONTINUOUSLY LOOK FOR WITH SCANS"
    READS = "CONTINUE TO READ SOMETHING"
    WRITES = "CURRENTLY WRITING SOMETHING"
    DISTRIBUTES = "SENDS OUT TO SPECIFIC LOCATIONS"
    ORGANIZES = "ORGANIZE WHILE PROCESSING PROCESSED LOCATIONS TO DISTRIBUTE 
    TO"
    INVENT = "CREATE SOMETHING OUT OF CREATIVITY OR IMAGINATION FROM IDEAS"
    BUILD = "BRING TOGETHER AND CREATE"
    ORGANIZED = "PROCESSED AND DISTRIBUTED TO EXACT LOCATION"
    FILTERS = "GIVES PROCESSED PLACEMENT METHODS FOR SOMETHING"
    SORTS = "FILTERS AND DEVELOPS THE POWER OR STRENGTH TO SORT"
    CONSTRUCT = "BUILD"
    ANALYZES = "LOOK FOR AND SCANS FOR"
    EXAMINES = "ANALYZES AND MAKES A DETERMINED RESOLUTION"
    DESIGNER = "SOMEONE WHO HAS THE CAPABILITY TO DESIGN SOMETHING"
    MAKER = "SOMEONE WHO HAS THE CAPABILITY TO MAKE SOMETHING"
    BUILDER = "SOMEONE WHO HAS THE CAPABILITY TO MAKE SOMETHING"
    CONSTRUCTOR = "SOMEONE WHO HAS THE CAPABILITY TO CONSTRUCT SOMETHING"
    ARCHITECT = "SOMEONE WHO HAS THE CAPABILITY TO CONSTRUCT SOMETHING WHILE 
    USING DESIGNER TECHNIQUES AND TECHNIQUES TO BUILD"
    INVENTOR = "SOMEONE WHO HAS THE CAPABILITY TO INVENT NEW IDEAS"
    ANALYST = "SOMEONE WHO HAS THE CAPABILITY TO ANALYZE SPECIFIC FIELDS AND 
    SPECIFIC FORMS OF INFORMATION"
    SPECIALIST = "SOMEONE WHO HAS A HIGH LEVEL OF KNOWLEDGE AND OR EXPERIENCE 
    WITHIN A SPECIFIC FIELD"
    ANALYZER = "SOMEONE WHO ANALYZES SOMETHING"
    SCANNER = "SOMEONE WHO SCANS SOMETHING"
    EXAMINER = "SOMEONE WHO EXAMINES SOMETHING"
    PRODUCER = "SOMEONE WHO CAN PRODUCE SOMETHING"
    ORGANIZER = "SOMEONE WHO ORGANIZES SOMETHING"
    SORTER = "SOMEONE WHO SORTS SOMETHING"
    DEVELOPER = "SOMEONE WHO DEVELOPS SOMETHING"
    WRITER = "SOMEONE WHO WRITES INFORMATION"
    READER = "SOMEONE WHO READS INFORMATION"
    EDITOR = "SOMEONE WHO EDITS SOMETHING"
    MANAGER = "SOMEONE WHO MANAGES SOMETHING"
    CONTROLLER = "SOMEONE WHO CONTROLS SOMETHING"
    MANIPULATOR = "SOMEONE WHO MANIPULATES SOMETHING"
    RESEARCHER = "SOMEONE WHO CONTINUOUSLY SEARCHES FOR NEW INFORMATION 
    TO MAKE A SOLUTION TO SOMETHING"
    ENGINEER = "SOMEONE WHO MAKES AND OR DESIGNS AND OR CONSTRUCTS SYSTEMS 
    AND OR FRAMEWORKS AND OR INTERFACES"
    TEACHER = "SOMEONE WHO ALLOWS SOMEONE TO LEARN NEW SKILLS AND OR 
    KNOWLEDGE WITHIN A SPECIFIC FIELD"
    PROFESSOR = "IS A HIGH QUALITY TEACHER WITH AGED KNOWLEDGE AND WISDOM 
    WITHIN A SPECIFIC FIELD"
    STORER = "SOMEONE WHO STORES SOMETHING SOMEWHERE"
    GATHERER = "SOMEONE WHO GATHERS ENERGY FOR SOMETHING AND OR SOMEONE"
    CLEANSER = "SOMEONE WHO CLEANSES SOMETHING"
    PURIFIER = "SOMEONE WHO PURIFIES SOMETHING"
    EXORCIST = "SOMEONE WHO EXPELS SOMETHING FROM SOMEWHERE AND OR 
    SOMETHING ELSE"
    PRIEST = "SOMEONE WHO HAS THE CAPABILITY TO CLEANSE AND OR PURIFY 
    SOMETHING"
    PALADIN = "A HIGH LEVEL ENTITY THAT HOLDS THE POWER TO CLEANSE AND PURIFY 
    AND EXPEL THINGS FROM VAST AMOUNTS OF LOCATIONS AND OR AREAS OF MANY 
    SHAPES AND OR SIZES"
    TAMER = "SOMEONE WHO HAS THE CAPABILITY TO TAME SOMETHING AND OR 
    SOMEONE"
    ALCHEMIST = "SOMEONE WHO HAS THE CAPABILITY TO PRODUCE THINGS MADE FROM 
    BOTH IMAGINATION AND TRUTHS"
    PHYSICIST = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND 
    COMPREHEND AND MAKE SPECIFIC FORMS OF PHYSICS EQUATIONS AND OR 
    FORMULAS"
    MATHEMATICIAN = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND 
    COMPREHEND AND MAKE SPECIFIC FORMS OF MATHEMATIC EQUATIONS AND OR 
    FORMULAS"
    CHEMIST = "SOMEONE WHO HAS THE CAPABILITY TO UNDERSTAND AND COMPREHEND 
    AND MAKE SPECIFIC FORMS OF MATHEMATIC EQUATIONS AND OR FORMULAS"
    WORKER = "SOMEONE WHO IS WORKING"
    TESTER = "SOMEONE WHO TESTS SOMETHING"
    ELEMENTALIST = "SOMEONE WHO CAN CONTROL THE ELEMENTS AND OR COMPREND 
    THE MEANING OF SPECIFIC ELEMENTS AND OR DEFINE NEW FORMS OF ELEMENTS AND 
    GIVE THE ELEMENTS MEANING"
    LINGUIST = "SOMEONE WHO MAKES AND UNDERSTANDS AS WELL AS DESCRIBES AND 
    DEFINES NEW FORMS OF LANGUAGE WITH COMPREHENDED MEANING"
    DESCRIBER = "SOMEONE WHO DESCRIBES NEW FORMS OF INFORMATION AND OR 
    MEANING"
    DEFINER = "SOMEONE WHO DEFINES NEW FORMS OF INFORMATION AND OR MEANING"
    CONTENT_DESIGNER = "SOMEONE WHO DESIGNS NEW CONTENT"
    CONTENT_MAKER = "SOMEONE WHO MAKES NEW FORMS OF CONTENT"
    CONTENT_PRODUCER = "SOMEONE WHO PRODUCES NEW CONTENT"
    CONTENT_ANALYST = "SOMEONE WHO ANALYZES CONTENT"
    CONTENT_SPECIALIST = "SOMEONE WHO SPECIALIZES WITHIN A SPECIFIC FIELD AND 
    OR CATEGORY OF SPECIFIC CONTENT"
    FIELD_SPECIALIST = "SOMEONE WHO SPECIALIZES IN ANALYZING AND 
    COMPREHENDING DIFFERENT TYPES OF FIELDS AND RECOGNIZING HOW THOSE FIELDS 
    CONNECT BETWEEN DIFFERENT GROUPS AND CLASSIFICATIONS"
    FIELD_DEVELOPER = "SOMEONE WHO DEVELOPS NEW AND OR OLD TYPES OF FIELDS"
    FIELD_ANALYST = "SOMEONE WHO IS GIVEN THE JOB OR ROLE TO ANALYZE DIFFERENT 
    TYPES AND CLASSES OF FIELDS"
    FIELD_ORGANIZER = "SOMEONE WHO ORGANIZES THE DIFFERENT TYPES AND CLASSES 
    OF FIELDS THAT HAVE BEEN MADE AND OR CREATED AND OR PRODUCED"
    FIELD_EXAMINER = "SOMEONE WHO EXAMINES THE PROPERTIES AND OR ELEMENTS OF 
    A FIELD TO DETERMINE ITS STATE OR FUNCTIONALITY"
    FIELD_MAKER = "SOMEONE WHO MAKES NEW TYPES OF FIELDS FROM SOMETHING NEW 
    OR OLD"
    VOCAL_SPECIALIST = "SOMEONE WHO SPECIALIZES IN ANALYZING VOCAL ACTIVITY"
    VOCAL_PITCH_ANALYZER = "SOMEONE WHO ANALYZES THE VOCAL PITCH OF A PERSON 
    OR ANIMAL OR ENTITY"
    VOCAL_ANALYST = "SOMEONE WHO ANALYZES ALL ASPECTS OF SOMETHING 
    PRODUCED BY VOCALS"
    VOCAL_EXAMINEER = "SOMEONE WHO EXAMINES AND DETERMINES THE 
    FUNCTIONALITY OF SOMETHING THAT IS PRODUCED BY VOCALS"
    VOCAL_ORGANIZER = "SOMEONE WHO ORGANIZES DIFFERENT TYPES OF VOCALS AND 
    CLASSIFIES THEM ACCORDINGLY"
    VOCAL_DEVELOPER = "SOMEONE WHO DEVELOPS NEW TYPES AND CLASSES OF 
    VOCALIZATION AND OR VOCAL INPUTS AND OR VOCAL TYPES OR CLASSES"
    VOCAL_CONTENT_MAKER = "SOMEONE WHO MAKES VOCAL CONTENT"
    VOCAL_CONTENT_PRODUCER = "SOMEONE WHO PRODUCES NEW VOCAL CONTENT"
    VOCAL_CONTENT_SCANNER = "SOMEONE WHO SCANS EXISTING OR OLD OR 
    UPCOMING VOCAL CONTENT"
    VOCAL_CONTENT_EXAMINEER = "SOMEONE WHO EXAMINES VOCAL CONTENT"
    VOCAL_CONTENT_ANALYST = "SOMEONE WHO ANALYZES DIFFERENT FORMS OF VOCAL 
    CONTENT"
    DATA_EXAMINEER = "SOMEONE WHO EXAMINES DIFFERENT FORMS OF DATA TO 
    DETERMINE ITS VALIDITY AND CLASSIFICATIONS"
    DATA_ENGINEER = "SOMEONE WHO CAN ENGINEER NEW FORMS OF DATA FROM 
    EXISTING DATA"
    DATA_ORGANIZER = "SOMEONE WHO ORGANIZES DIFFERENT FORMS OF DATA INTO 
    SPECIFIC GROUPS OR CLASSIFICATIONS AND OR CATEGORIES AND OR GENRES"
    DATA_DESIGNER = "SOMEONE WHO DESIGNS NEW FORMS OF DATA FROM PREVIOUS 
    AND OR CURRENT AND OR UPCOMING DATA"
    DATA_ANALYST = "SOMEONE WHO ANALYZES AND LOOKS OVER DATA"
    HARDWARE_ENGINEER = "SOMEONE WHO ENGINEERS FORMS OF HARDWARE"
    HARDWARE_EXAMINER = "SOMEONE WHO EXAMINES FORMS OF HARDWARE"
    HARDWARE_DESIGNER = "SOMEONE WHO DESIGNS HARDWARE"
    HARDWARE_DEVELOPER = "SOMEONE WHO DEVELOPS HARDWARE"
    SCULPTOR = "IS SOMEONE WHO CREATES THREEDIMENSIONAL ARTWORKS"
    TECHNICIAN = "IS SOMEONE WHO HAS SPECIALIZED SKILLS IN A PARTICULAR FIELD"
    VISIONARY = "IS SOMEONE WHO HAS A CLEAR IDEA OF WHAT THE FUTURE COULD BE 
    LIKE"
    ARTISAN = "IS SOMEONE WHO CREATES HANDMADE OBJECTS USING TRADITIONAL 
    TECHNIQUES"
    CRAFTSMAN = "IS SOMEONE WHO IS SKILLED IN A PARTICULAR CRAFT"
"""]

    lda_model, vectorizer = perform_lda(text_data)
    topics = lda_model.transform(vectorizer.transform(text_data))
    print("LDA Topics:", topics)
