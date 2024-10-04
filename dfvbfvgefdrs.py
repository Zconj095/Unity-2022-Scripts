import neat 
import tensorflow as tf
from tensorflow.keras import layers

# Create TensorFlow model with evolvable topology
input_layer = layers.Input(shape=(max_text_length,))
hidden_1 = layers.Dense(10, activation="relu")(input_layer)
output_layer = layers.Dense(num_classes, activation="softmax")(hidden_1)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Define custom mutation operator to add/remove layers
def evolve_topology(model, activation):
    if np.random.random() < add_layer_chance:
        # Add new hidden layer  
        new_layer = layers.Dense(50, activation=activation)(model.layers[-1].output) 
        model = tf.keras.Model(inputs=model.inputs, outputs=new_layer)

    if np.random.random() < delete_layer_chance:
        # Delete last layer
        model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

    return model

# Evaluate model on NLP task and generate fitness  
def evaluate_fitness(genome):
    model = tf.keras.models.load_model(genome.model_file)
    loss, accuracy = model.evaluate(x_test, y_test)   
    fitness = accuracy

    return fitness

# Run NEAT algorithm to evolve topology and weights
def train_model():
    population = neat.Population(config_file)
    for generation in range(100):
        genome_list = neat.get_current_genomes(population)
        
        for genome in genome_list:
            model = tf.keras.models.load_model(genome.model_file)
            model = evolve_topology(model, activation="relu") # Mutate 
            genome.set_model(model)  
            genome.fitness = evaluate_fitness(genome)
            
        population.evolve()
       
    best_genome = population.get_best_genome() 
    best_model = tf.keras.models.load_model(best_genome.model_file)
    return best_model

if __name__ == "__main__":
    best_model = train_model()
    best_model.save("evolved_nlp_model.h5")