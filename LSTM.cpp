class LSTM
{
public:
    LSTM(int input_size, int hidden_size, int output_size, int num_layers)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->output_size = output_size;
        this->num_layers = num_layers;

        // Initialize LSTM layers
        for (int i = 0; i < num_layers; ++i)
        {
            lstm_layers.push_back(LSTMCell(input_size, hidden_size));
        }

        // Initialize output layer weights and biases
        W_out = Matrix(output_size, hidden_size);
        b_out = Vector(output_size);
        W_out.Randomize();
        b_out.Zero();
    }

    Vector Forward(const std::vector<Vector>& inputs)
    {
        Vector h = Vector(hidden_size);
        Vector c = Vector(hidden_size);
        h.Zero();
        c.Zero();

        for (int t = 0; t < inputs.size(); ++t)
        {
            for (int layer = 0; layer < num_layers; ++layer)
            {
                lstm_layers[layer].Forward(inputs[t], h, c, h, c);
            }
        }

        // Output layer
        Vector output = (W_out * h) + b_out;
        return output;
    }

private:
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;

    std::vector<LSTMCell> lstm_layers;
    Matrix W_out;
    Vector b_out;
};
