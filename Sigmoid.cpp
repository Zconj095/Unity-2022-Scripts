class Sigmoid
{
public:
    static Vector Apply(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = 1.0f / (1.0f + exp(-x[i]));
        }
        return result;
    }

    static Vector Derivative(const Vector& x)
    {
        Vector sigmoid = Apply(x);
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = sigmoid[i] * (1.0f - sigmoid[i]);
        }
        return result;
    }
};
