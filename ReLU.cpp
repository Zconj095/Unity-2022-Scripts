class ReLU
{
public:
    static Vector Apply(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = std::max(0.0f, x[i]);
        }
        return result;
    }

    static Vector Derivative(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = (x[i] > 0) ? 1.0f : 0.0f;
        }
        return result;
    }
};
