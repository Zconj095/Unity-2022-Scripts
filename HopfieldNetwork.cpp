#include <vector>
#include <cmath>

class HopfieldNetwork
{
public:
    HopfieldNetwork(int size)
    {
        this->size = size;
        weights = Matrix(size, size);
        weights.Zero();
    }

    void Train(const std::vector<Vector>& patterns)
    {
        // Hebbian learning rule to adjust weights based on patterns
        for (const auto& pattern : patterns)
        {
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    if (i != j)
                    {
                        weights(i, j) += pattern[i] * pattern[j];
                    }
                }
            }
        }

        // Ensure weights are symmetric
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                weights(i, j) = weights(j, i);
            }
        }
    }

    Vector Recall(const Vector& input, int max_iterations = 100)
    {
        Vector state = input;

        for (int iteration = 0; iteration < max_iterations; ++iteration)
        {
            Vector new_state = state;

            for (int i = 0; i < size; ++i)
            {
                double sum = 0;
                for (int j = 0; j < size; ++j)
                {
                    sum += weights(i, j) * state[j];
                }

                new_state[i] = (sum >= 0) ? 1 : -1;  // Binary threshold
            }

            if (new_state == state)  // If state stabilizes, stop
            {
                break;
            }

            state = new_state;
        }

        return state;
    }

private:
    int size;
    Matrix weights;
};
