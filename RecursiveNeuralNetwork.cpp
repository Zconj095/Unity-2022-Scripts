class RecursiveNeuralNetwork
{
public:
    RecursiveNeuralNetwork(int input_size, int hidden_size, int output_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->output_size = output_size;

        root = new RNNNode(input_size, hidden_size);

        // Initialize output layer weights and biases
        W_out = Matrix(output_size, hidden_size);
        b_out = Vector(output_size);
        W_out.Randomize();
        b_out.Zero();
    }

    ~RecursiveNeuralNetwork()
    {
        delete root;
    }

    Vector Forward(const Vector& input)
    {
        // Process the root node recursively
        Vector hidden_state = root->Forward(input, root->children);

        // Output layer
        Vector output = (W_out * hidden_state) + b_out;
        return output;
    }

    void AddChildToRoot(RNNNode* child)
    {
        root->AddChild(child);
    }

private:
    int input_size;
    int hidden_size;
    int output_size;

    RNNNode* root;

    Matrix W_out;
    Vector b_out;
};
