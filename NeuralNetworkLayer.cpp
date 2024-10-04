class NeuralNetworkLayer
{
public:
    NeuralNetworkLayer(int input_size, int output_size)
    {
        this->input_size = input_size;
        this->output_size = output_size;

        // Initialize weights and biases
        W = Matrix(output_size, input_size);
        b = Vector(output_size);

        // Randomly initialize weights and biases
        W.Randomize();
        b.Zero();
    }

    Vector Forward(const Vector& input)
    {
        Vector z = (W * input) + b;
        return ReLU::Apply(z);  // Apply ReLU activation function
    }

    Vector Backward(const Vector& input, const Vector& grad_output)
    {
        Vector z = (W * input) + b;
        Vector grad_z = ReLU::Derivative(z) * grad_output;  // Apply ReLU derivative
        // Update weights and biases here based on grad_z...
        return grad_z;
    }

private:
    int input_size;
    int output_size;

    Matrix W;
    Vector b;
};
