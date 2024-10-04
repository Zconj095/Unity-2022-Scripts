class SupervisedRNN
{
public:
    SupervisedRNN(int input_size, int hidden_size, int output_size, int num_layers = 1)
    {
        this->num_layers = num_layers;
        for (int i = 0; i < num_layers; ++i)
        {
            rnn_layers.push_back(RNNCell(input_size, hidden_size, output_size));
        }
    }

    // Forward pass through the entire sequence
    std::vector<std::vector<float>> Forward(const std::vector<std::vector<float>>& inputs)
    {
        std::vector<float> h = std::vector<float>(rnn_layers[0].hidden_size, 0.0f);
        std::vector<std::vector<float>> outputs;

        for (const auto& input : inputs)
        {
            for (int i = 0; i < num_layers; ++i)
            {
                h = rnn_layers[i].Forward(input, h);
            }
            outputs.push_back(h);
        }

        return outputs;
    }

    // Training method (placeholder)
    void Train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate)
    {
        // Placeholder for training implementation
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            std::vector<std::vector<float>> predictions = Forward(inputs);

            // Compute loss and backpropagate through time (BPTT) - simplified here
            // Update weights using the gradients - this requires additional functions

            std::cout << "Epoch " << epoch << " completed." << std::endl;
        }
    }

private:
    int num_layers;
    std::vector<RNNCell> rnn_layers;
};
