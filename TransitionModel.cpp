#include <iostream>
#include <vector>

class TransitionModel
{
public:
    TransitionModel(const std::vector<std::vector<double>>& transitionMatrix)
        : transitionMatrix(transitionMatrix)
    {
        num_states = transitionMatrix.size();
    }

    // Compute the next state vector given the current state vector
    std::vector<double> NextState(const std::vector<double>& currentState)
    {
        std::vector<double> nextState(num_states, 0.0);

        for (int i = 0; i < num_states; ++i)
        {
            for (int j = 0; j < num_states; ++j)
            {
                nextState[i] += currentState[j] * transitionMatrix[j][i];
            }
        }

        return nextState;
    }

private:
    int num_states;
    std::vector<std::vector<double>> transitionMatrix;
};
