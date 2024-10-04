class RecurrentNeuralNetwork
{
public:
    RecurrentNeuralNetwork(int input_size, int hidden_size, int output_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->output_size = output_size;

        // Initialize the RNN cell
        rnn_cell = new RNNCell(input_size, hidden_size);

        // Initialize output layer weights and biases
        W_out = Matrix(output_size, hidden_size);
        b_out = Vector(output_size);
        W_out.Randomize();
        b_out.Zero();
    }

    ~RecurrentNeuralNetwork()
    {
        delete rnn_cell;
    }

    Vector Forward(const std::vector<Vector>& inputs)
    {
        Vector h = Vector(hidden_size);
        h.Zero();

        // Process each time step in the sequence
        for (int t = 0; t < inputs.size(); ++t)
        {
            h = rnn_cell->Forward(inputs[t], h);
        }

        // Output layer
        Vector output = (W_out * h) + b_out;
        return output;
    }

private:
    int input_size;
    int hidden_size;
    int output_size;

    RNNCell* rnn_cell;

    Matrix W_out;
    Vector b_out;
};
