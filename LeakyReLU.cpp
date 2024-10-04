class LeakyReLU
{
public:
    static Vector Apply(const Vector& x, float alpha = 0.01f)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = (x[i] > 0) ? x[i] : alpha * x[i];
        }
        return result;
    }

    static Vector Derivative(const Vector& x, float alpha = 0.01f)
    {
        Vector result(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = (x[i] > 0) ? 1.0f : alpha;
        }
        return result;
    }
};
