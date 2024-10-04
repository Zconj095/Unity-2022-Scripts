class MyPerceptron:
    class Subclassone:
        inputs = [1, 2]
        weights = [1, 1, 1]

        def perceptron_predict(inputs, weights):
            activation = weights[0]
            for i in range(len(inputs)-1):
                activation += weights[1] * input
            return 1.0 if activation >= 0.0 else 0.0

        print(perceptron_predict(inputs,weights))
    
    class Subclasstwo:
        train = [[1,2],[2,3],[1,1],[2,2],[3,3],[4,2],[2,5],[5,5],[4,1],[4,4]]
        weights = [1,1,1]
        
        def perceptron_predict(inputs, weights):
            activation = weights[0]
            for i in range(len(inputs)-1):
                activation += weights[i+1] * inputs[i]
                return 1.0 if activation >= 0.0 else 0.0
            
        for inputs in train:
            print(perceptron_predict(inputs,weights))

    class Subclassthree:
        def perceptron_predict(inputs, weights):
            activation = weights[0]
            for i in range(len(inputs)-1):
                activation+= weights[i + 1] * inputs[i]
            return 1.0 if activation >= 0.0 else 0.0
        
        def train_weights(train, learning_rate, epochs):
            weights = [0.0 for i in range(len(train[0]))]
            for epoch in range(epochs):
                sum_error = 0.0
                for inputs in train:
                    prediction = MyPerceptron.Subclassthree.perceptron_predict(inputs, weights)
                    error = inputs[-1] - prediction
                    sum_error += error ** 2
                    weights[0] = weights[0] + learning_rate * error
                    for i in range(len(inputs) - 1):
                        weights[i + 1] = weights[i + 1] + learning_rate * error * inputs[i]
            print('>epoch=%d,  learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
            return weights
        
        train = [[1.5,2.5,0],[2.5,3.5,0],[1.0,11.0,1],[2.3,2.3,1],[3.6,3.6,1],[4.2,2.4,0],[2.4,5.4,0],[5.1,5.1,0],[4.3,1.3,0],[4.8,4.8,1]]
        learning_rate = 0.1
        epochs = 10
        weights = train_weights(train, learning_rate, epochs)
        print(weights)
        
        class Subclassfour:
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
            import tensorflow as tf
            # Parameters
            learning_rate = 0.001
            training_epochs = 15
            batch_size = 100
            display_step = 1
            # Network Parameters
            n_hidden_1 = 256 # 1st layer number of neurons
            n_hidden_2 = 256 # 2nd layer number of neurons
            n_input = 784 # MNIST data input (img shape: 28*28)
            n_classes = 10 # MNIST total classes (0-9 digits)
            
            X = tf.placeholder("float", [None, n_input])
            Y = tf.placeholder("float", [None, n_classes])
            
            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
            
            #Create model
            def multilayer_perceptron(x):
                import tensorflow as tf
                # Hidden fully connected layer with 256 neurons
                layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
                # Hidden fully connected layer with 256 neurons