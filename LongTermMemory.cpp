#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

class LongTermMemory
{
public:
    LongTermMemory(int slot_size)
    {
        this->slot_size = slot_size;
    }

    // Write data to memory with a specific key (label)
    void Write(const std::string& key, const std::vector<float>& data)
    {
        if (data.size() != slot_size)
        {
            throw std::runtime_error("Data size does not match the slot size");
        }

        memory[key] = data;
    }

    // Read data from memory by key
    std::vector<float> Read(const std::string& key)
    {
        if (memory.find(key) != memory.end())
        {
            return memory[key];
        }
        else
        {
            throw std::runtime_error("Key not found in memory");
        }
    }

    // Content-based addressing to retrieve the most similar memory slot
    std::string Address(const std::vector<float>& key_vector)
    {
        std::string best_key;
        float best_similarity = -1.0f;

        for (const auto& entry : memory)
        {
            float similarity = CosineSimilarity(entry.second, key_vector);
            if (similarity > best_similarity)
            {
                best_similarity = similarity;
                best_key = entry.first;
            }
        }

        if (best_similarity == -1.0f)
        {
            throw std::runtime_error("No suitable match found in memory");
        }

        return best_key;
    }

    // Display all stored memory keys
    void DisplayMemoryKeys()
    {
        std::cout << "Memory Keys: " << std::endl;
        for (const auto& entry : memory)
        {
            std::cout << entry.first << std::endl;
        }
    }

private:
    int slot_size;
    std::unordered_map<std::string, std::vector<float>> memory;

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

        return dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8f);
    }
};
