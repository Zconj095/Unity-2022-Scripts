#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <ctime>

class MarkovChain
{
public:
    MarkovChain(const std::vector<std::string>& states)
    {
        this->states = states;
        state_count = states.size();
        transition_matrix.resize(state_count, std::vector<float>(state_count, 0.0f));
        current_state_index = 0;
        srand(static_cast<unsigned>(time(0)));
    }

    // Set transition probability from one state to another
    void SetTransitionProbability(const std::string& from_state, const std::string& to_state, float probability)
    {
        int from_index = GetStateIndex(from_state);
        int to_index = GetStateIndex(to_state);

        if (from_index != -1 && to_index != -1)
        {
            transition_matrix[from_index][to_index] = probability;
        }
        else
        {
            throw std::runtime_error("Invalid state provided.");
        }
    }

    // Set the initial state of the Markov Chain
    void SetInitialState(const std::string& initial_state)
    {
        int index = GetStateIndex(initial_state);
        if (index != -1)
        {
            current_state_index = index;
        }
        else
        {
            throw std::runtime_error("Invalid initial state provided.");
        }
    }

    // Get the next state based on the current state
    std::string GetNextState()
    {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float cumulative_probability = 0.0f;

        for (int i = 0; i < state_count; ++i)
        {
            cumulative_probability += transition_matrix[current_state_index][i];
            if (r <= cumulative_probability)
            {
                current_state_index = i;
                return states[i];
            }
        }

        return states[current_state_index];  // Fallback, should never hit this
    }

    // Simulate the Markov Chain for a number of steps
    void Simulate(int steps)
    {
        for (int i = 0; i < steps; ++i)
        {
            std::string next_state = GetNextState();
            std::cout << "Step " << i + 1 << ": " << next_state << std::endl;
        }
    }

private:
    std::vector<std::string> states;
    std::vector<std::vector<float>> transition_matrix;
    int current_state_index;
    int state_count;

    // Get the index of a state in the states vector
    int GetStateIndex(const std::string& state)
    {
        auto it = std::find(states.begin(), states.end(), state);
        if (it != states.end())
        {
            return std::distance(states.begin(), it);
        }
        return -1;  // State not found
    }
};
