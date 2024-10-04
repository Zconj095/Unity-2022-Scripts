class LSTMCell
{
public:
    LSTMCell(int input_size, int hidden_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;

        // Initialize weights and biases for gates
        Wf = Matrix(hidden_size, input_size + hidden_size);
        Wi = Matrix(hidden_size, input_size + hidden_size);
        Wo = Matrix(hidden_size, input_size + hidden_size);
        Wc = Matrix(hidden_size, input_size + hidden_size);

        bf = Vector(hidden_size);
        bi = Vector(hidden_size);
        bo = Vector(hidden_size);
        bc = Vector(hidden_size);

        // Randomly initialize weights and biases
        Wf.Randomize();
        Wi.Randomize();
        Wo.Randomize();
        Wc.Randomize();
        bf.Zero();
        bi.Zero();
        bo.Zero();
        bc.Zero();
    }

    void Forward(const Vector& input, const Vector& h_prev, const Vector& c_prev, Vector& h_next, Vector& c_next)
    {
        Vector combined = Concatenate(input, h_prev);

        // Forget gate
        Vector f_t = Sigmoid((Wf * combined) + bf);

        // Input gate
        Vector i_t = Sigmoid((Wi * combined) + bi);

        // Candidate cell state
        Vector c_tilde_t = Tanh((Wc * combined) + bc);

        // New cell state
        c_next = (f_t * c_prev) + (i_t * c_tilde_t);

        // Output gate
        Vector o_t = Sigmoid((Wo * combined) + bo);

        // New hidden state
        h_next = o_t * Tanh(c_next);
    }

private:
    int input_size;
    int hidden_size;

    Matrix Wf, Wi, Wo, Wc;
    Vector bf, bi, bo, bc;

    // Activation functions
    Vector Sigmoid(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result;
    }

    Vector Tanh(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = tanh(x[i]);
        }
        return result;
    }

    Vector Concatenate(const Vector& a, const Vector& b)
    {
        Vector result(a.size() + b.size());
        for (int i = 0; i < a.size(); ++i)
        {
            result[i] = a[i];
        }
        for (int i = 0; i < b.size(); ++i)
        {
            result[a.size() + i] = b[i];
        }
        return result;
    }
};
