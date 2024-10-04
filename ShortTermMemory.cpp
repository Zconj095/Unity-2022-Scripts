#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>

class ShortTermMemory
{
public:
    ShortTermMemory(int num_slots, int slot_size)
    {
        this->num_slots = num_slots;
        this->slot_size = slot_size;

        // Initialize the memory with zeros
        memory = std::vector<std::vector<float>>(num_slots, std::vector<float>(slot_size, 0.0f));
        recent_keys = std::queue<std::string>();
    }

    // Write data to memory using a specific key
    void Write(const std::string& key, const std::vector<float>& data)
    {
        if (data.size() != slot_size)
        {
            throw std::runtime_error("Data size does not match the slot size");
        }

        if (memory_map.find(key) != memory_map.end())
        {
            memory[memory_map[key]] = data;
        }
        else
        {
            int slot_index = recent_keys.size() < num_slots ? recent_keys.size() : recent_keys.front();
            if (recent_keys.size() >= num_slots)
            {
                memory_map.erase(recent_keys.front());
                recent_keys.pop();
            }
            recent_keys.push(key);
            memory[slot_index] = data;
            memory_map[key] = slot_index;
        }
    }

    // Read data from memory using a specific key
    std::vector<float> Read(const std::string& key)
    {
        if (memory_map.find(key) != memory_map.end())
        {
            return memory[memory_map[key]];
        }
        else
        {
            throw std::runtime_error("Key not found in memory");
        }
    }

    // Clear the short-term memory
    void Clear()
    {
        memory = std::vector<std::vector<float>>(num_slots, std::vector<float>(slot_size, 0.0f));
        while (!recent_keys.empty())
        {
            recent_keys.pop();
        }
        memory_map.clear();
    }

    // Display the current memory keys
    void DisplayMemoryKeys()
    {
        std::queue<std::string> temp = recent_keys;
        std::cout << "Current Memory Keys: ";
        while (!temp.empty())
        {
            std::cout << temp.front() << " ";
            temp.pop();
        }
        std::cout << std::endl;
    }

private:
    int num_slots;
    int slot_size;
    std::vector<std::vector<float>> memory;
    std::queue<std::string> recent_keys;
    std::unordered_map<std::string, int> memory_map;
};
