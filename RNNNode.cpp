class RNNNode
{
public:
    RNNNode(int input_size, int hidden_size)
    {
        this->input_size = input_size;
        this->hidden_size = hidden_size;

        // Initialize weights and biases
        W = Matrix(hidden_size, input_size + hidden_size);
        b = Vector(hidden_size);

        // Randomly initialize weights and biases
        W.Randomize();
        b.Zero();
    }

    Vector Forward(const Vector& input, const std::vector<RNNNode*>& children)
    {
        Vector h = input;

        // Recursively process child nodes
        for (RNNNode* child : children)
        {
            h = h + child->Forward(input, child->children);
        }

        // Apply transformation
        Vector combined = Concatenate(input, h);
        Vector h_next = Tanh((W * combined) + b);

        return h_next;
    }

    void AddChild(RNNNode* child)
    {
        children.push_back(child);
    }

private:
    int input_size;
    int hidden_size;

    Matrix W;
    Vector b;

    std::vector<RNNNode*> children;

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
