class Tanh
{
public:
    static Vector Apply(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = tanh(x[i]);
        }
        return result;
    }

    static Vector Derivative(const Vector& x)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = 1.0f - pow(tanh(x[i]), 2);
        }
        return result;
    }
};
