#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class HiddenMarkovModel
{
public:
    HiddenMarkovModel(const std::vector<std::vector<double>>& A,
                      const std::vector<std::vector<double>>& B,
                      const std::vector<double>& pi)
        : A(A), B(B), pi(pi)
    {
        num_states = A.size();
        num_observations = B[0].size();
    }

    // Forward algorithm: Computes the probability of the observation sequence given the model
    double ForwardAlgorithm(const std::vector<int>& observations)
    {
        int T = observations.size();
        std::vector<std::vector<double>> alpha(T, std::vector<double>(num_states, 0.0));

        // Initialize the forward probabilities
        for (int i = 0; i < num_states; ++i)
        {
            alpha[0][i] = pi[i] * B[i][observations[0]];
        }

        // Recursively compute the forward probabilities
        for (int t = 1; t < T; ++t)
        {
            for (int j = 0; j < num_states; ++j)
            {
                alpha[t][j] = 0.0;
                for (int i = 0; i < num_states; ++i)
                {
                    alpha[t][j] += alpha[t - 1][i] * A[i][j];
                }
                alpha[t][j] *= B[j][observations[t]];
            }
        }

        // Sum the probabilities of the final states
        double prob = 0.0;
        for (int i = 0; i < num_states; ++i)
        {
            prob += alpha[T - 1][i];
        }

        return prob;
    }

    // Viterbi algorithm: Finds the most likely state sequence given the observations
    std::vector<int> ViterbiAlgorithm(const std::vector<int>& observations)
    {
        int T = observations.size();
        std::vector<std::vector<double>> delta(T, std::vector<double>(num_states, 0.0));
        std::vector<std::vector<int>> psi(T, std::vector<int>(num_states, 0));

        // Initialize
        for (int i = 0; i < num_states; ++i)
        {
            delta[0][i] = pi[i] * B[i][observations[0]];
            psi[0][i] = 0;
        }

        // Recursion
        for (int t = 1; t < T; ++t)
        {
            for (int j = 0; j < num_states; ++j)
            {
                double max_prob = delta[t - 1][0] * A[0][j];
                int max_state = 0;
                for (int i = 1; i < num_states; ++i)
                {
                    double prob = delta[t - 1][i] * A[i][j];
                    if (prob > max_prob)
                    {
                        max_prob = prob;
                        max_state = i;
                    }
                }
                delta[t][j] = max_prob * B[j][observations[t]];
                psi[t][j] = max_state;
            }
        }

        // Termination
        double max_prob = delta[T - 1][0];
        int last_state = 0;
        for (int i = 1; i < num_states; ++i)
        {
            if (delta[T - 1][i] > max_prob)
            {
                max_prob = delta[T - 1][i];
                last_state = i;
            }
        }

        // Path backtracking
        std::vector<int> state_sequence(T);
        state_sequence[T - 1] = last_state;
        for (int t = T - 2; t >= 0; --t)
        {
            state_sequence[t] = psi[t + 1][state_sequence[t + 1]];
        }

        return state_sequence;
    }

private:
    int num_states;
    int num_observations;

    std::vector<std::vector<double>> A; // Transition matrix
    std::vector<std::vector<double>> B; // Emission matrix
    std::vector<double> pi;             // Initial state distribution
};
