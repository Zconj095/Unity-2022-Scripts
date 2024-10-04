#include <unordered_map>
#include <vector>
#include <functional>
#include <iostream>

class MemoryCache
{
public:
    MemoryCache(size_t max_size) : max_size(max_size) {}

    // Generate a hash key from input data (sequence)
    std::string GenerateKey(const std::vector<float>& input_sequence)
    {
        std::hash<std::string> hash_fn;
        std::string key = "";
        for (float value : input_sequence)
        {
            key += std::to_string(value) + ",";
        }
        return std::to_string(hash_fn(key));
    }

    // Check if the result for a given input sequence is in the cache
    bool Contains(const std::string& key)
    {
        return cache.find(key) != cache.end();
    }

    // Retrieve cached result
    std::vector<float> Get(const std::string& key)
    {
        if (Contains(key))
        {
            return cache[key];
        }
        else
        {
            throw std::runtime_error("Key not found in cache");
        }
    }

    // Store result in the cache
    void Store(const std::string& key, const std::vector<float>& result)
    {
        if (cache.size() >= max_size)
        {
            Evict();
        }
        cache[key] = result;
        order_of_keys.push_back(key);
    }

    // Clear the cache
    void Clear()
    {
        cache.clear();
        order_of_keys.clear();
    }

    // Display the cache contents (for debugging)
    void DisplayCache()
    {
        std::cout << "Cache contents: " << std::endl;
        for (const auto& key : order_of_keys)
        {
            std::cout << key << ": ";
            for (float value : cache[key])
            {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    size_t max_size;
    std::unordered_map<std::string, std::vector<float>> cache;
    std::vector<std::string> order_of_keys;

    // Evict the oldest entry from the cache
    void Evict()
    {
        if (!order_of_keys.empty())
        {
            std::string oldest_key = order_of_keys.front();
            order_of_keys.erase(order_of_keys.begin());
            cache.erase(oldest_key);
        }
    }
};
