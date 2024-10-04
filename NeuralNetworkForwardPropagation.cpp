#include <vector>
#include <cmath>
#include <iostream>

class NeuralNetworkForwardPropagation
{
public:
    NeuralNetworkForwardPropagation(int input_size, int hidden_size, int output_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->output_size = output_size;

        // Initialize weights and biases
        W1 = Matrix(hidden_size, input_size);
        b1 = Vector(hidden_size);
        W2 = Matrix(output_size, hidden_size);
        b2 = Vector(output_size);

        // Randomly initialize weights and biases (simplified)
        W1.Randomize();
        b1.Zero();
        W2.Randomize();
        b2.Zero();
    }

    // Forward pass through the network
    std::vector<float> Forward(const std::vector<float>& input)
    {
        // Layer 1 (Hidden Layer)
        z1 = (W1 * input) + b1;
        a1 = Sigmoid(z1);

        // Layer 2 (Output Layer)
        z2 = (W2 * a1) + b2;
        a2 = Sigmoid(z2);

        return a2;
    }

private:
    int input_size, hidden_size, output_size;

    Matrix W1, W2;
    Vector b1, b2;

    std::vector<float> z1, a1, z2, a2;

    // Sigmoid activation function
    std::vector<float> Sigmoid(const std::vector<float>& x)
    {
        std::vector<float> result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = 1.0f / (1.0f + exp(-x[i]));
        }
        return result;
    }
};
