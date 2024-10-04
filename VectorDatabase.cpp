#include <vector>
#include <string>
#include <unordered_map>

class VectorDatabase
{
public:
    // Add a vector to the database with an associated identifier (e.g., file path)
    void AddVector(const std::string& id, const std::vector<float>& vector)
    {
        database[id] = vector;
    }

    // Query the database with a vector, returning the closest matching id (cosine similarity)
    std::string Query(const std::vector<float>& vector)
    {
        std::string best_match;
        float best_similarity = -1.0f;

        for (const auto& entry : database)
        {
            float similarity = CosineSimilarity(vector, entry.second);
            if (similarity > best_similarity)
            {
                best_similarity = similarity;
                best_match = entry.first;
            }
        }

        return best_match;
    }

    // Get the vector for a given ID
    std::vector<float> GetVector(const std::string& id)
    {
        return database[id];
    }

private:
    std::unordered_map<std::string, std::vector<float>> database;

    // Compute cosine similarity between two vectors
    float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
    {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (size_t i = 0; i < a.size(); ++i)
        {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        return dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8f);
    }
};
