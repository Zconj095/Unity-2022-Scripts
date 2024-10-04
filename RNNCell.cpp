class RNNCell
{
public:
    RNNCell(int input_size, int hidden_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;

        // Initialize weights and biases
        Wx = Matrix(hidden_size, input_size);  // Weight matrix for input
        Wh = Matrix(hidden_size, hidden_size); // Weight matrix for hidden state
        b = Vector(hidden_size);               // Bias vector

        // Randomly initialize weights and biases
        Wx.Randomize();
        Wh.Randomize();
        b.Zero();
    }

    // Forward pass for one time step
    Vector Forward(const Vector& input, const Vector& h_prev)
    {
        Vector h_next = Tanh((Wx * input) + (Wh * h_prev) + b);
        return h_next;
    }

private:
    int input_size;
    int hidden_size;

    Matrix Wx, Wh;
    Vector b;

    // Activation function
    Vector Tanh(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = tanh(x[i]);
        }
        return result;
    }
};
