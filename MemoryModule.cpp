#include <vector>
#include <cmath>
#include <algorithm>

class MemoryModule
{
public:
    MemoryModule(int num_slots, int slot_size)
    {
        this->num_slots = num_slots;
        this->slot_size = slot_size;

        // Initialize memory with zeros
        memory = std::vector<std::vector<float>>(num_slots, std::vector<float>(slot_size, 0.0f));
    }

    // Write to memory at a specific location
    void Write(int slot_index, const std::vector<float>& data)
    {
        if (slot_index < 0 || slot_index >= num_slots || data.size() != slot_size)
        {
            throw std::runtime_error("Invalid memory write operation");
        }
        memory[slot_index] = data;
    }

    // Read from memory at a specific location
    std::vector<float> Read(int slot_index)
    {
        if (slot_index < 0 || slot_index >= num_slots)
        {
            throw std::runtime_error("Invalid memory read operation");
        }
        return memory[slot_index];
    }

    // Content-based addressing (returns the index of the most similar memory slot)
    int Address(const std::vector<float>& key, float beta = 1.0f)
    {
        std::vector<float> similarities(num_slots, 0.0f);

        // Calculate cosine similarities
        for (int i = 0; i < num_slots; ++i)
        {
            similarities[i] = CosineSimilarity(memory[i], key);
        }

        // Apply softmax with the focus parameter beta
        std::vector<float> softmax_weights = Softmax(similarities, beta);

        // Return the index with the highest weight
        return std::distance(softmax_weights.begin(), std::max_element(softmax_weights.begin(), softmax_weights.end()));
    }

private:
    int num_slots;
    int slot_size;
    std::vector<std::vector<float>> memory;

    float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
    {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (int i = 0; i < a.size(); ++i)
        {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        return dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8f);  // Adding small value to avoid division by zero
    }

    std::vector<float> Softmax(const std::vector<float>& x, float beta)
    {
        std::vector<float> exp_x(x.size());
        float sum_exp_x = 0.0f;

        for (int i = 0; i < x.size(); ++i)
        {
            exp_x[i] = exp(beta * x[i]);
            sum_exp_x += exp_x[i];
        }

        for (int i = 0; i < x.size(); ++i)
        {
            exp_x[i] /= sum_exp_x;
        }

        return exp_x;
    }
};
