#include <vector>
#include <cmath>
#include <iostream>

class NeuralNetworkBackpropagation
{
public:
    NeuralNetworkBackpropagation(int input_size, int hidden_size, int output_size)
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

    // Backward pass (Backpropagation)
    void Backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate)
    {
        // Calculate output error (dL/dz2)
        std::vector<float> delta2 = ElementwiseMultiply(a2 - target, SigmoidDerivative(z2));

        // Calculate gradient for W2 and b2
        Matrix dW2 = OuterProduct(delta2, a1);
        Vector db2 = delta2;

        // Calculate hidden layer error (dL/dz1)
        std::vector<float> delta1 = ElementwiseMultiply(MatrixTranspose(W2) * delta2, SigmoidDerivative(z1));

        // Calculate gradient for W1 and b1
        Matrix dW1 = OuterProduct(delta1, input);
        Vector db1 = delta1;

        // Update weights and biases
        W2 -= dW2 * learning_rate;
        b2 -= db2 * learning_rate;
        W1 -= dW1 * learning_rate;
        b1 -= db1 * learning_rate;
    }

    // Train the network on a single sample
    void Train(const std::vector<float>& input, const std::vector<float>& target, float learning_rate)
    {
        Forward(input);
        Backward(input, target, learning_rate);
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

    // Derivative of the sigmoid function
    std::vector<float> SigmoidDerivative(const std::vector<float>& x)
    {
        std::vector<float> result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
            result[i] = sigmoid_x * (1.0f - sigmoid_x);
        }
        return result;
    }

    // Utility function: Elementwise multiplication
    std::vector<float> ElementwiseMultiply(const std::vector<float>& a, const std::vector<float>& b)
    {
        std::vector<float> result(a.size());
        for (int i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    // Utility function: Outer product
    Matrix OuterProduct(const std::vector<float>& a, const std::vector<float>& b)
    {
        Matrix result(a.size(), b.size());
        for (int i = 0; i < a.size(); ++i)
        {
            for (int j = 0; j < b.size(); ++j)
            {
                result(i, j) = a[i] * b[j];
            }
        }
        return result;
    }
};
