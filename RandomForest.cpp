#include <vector>
#include <cstdlib>
#include <ctime>

class RandomForest
{
public:
    RandomForest(int num_trees, int max_depth) : num_trees(num_trees), max_depth(max_depth)
    {
        srand(static_cast<unsigned>(time(0)));
    }

    void Train(const std::vector<std::vector<float>>& data, const std::vector<int>& labels)
    {
        trees.resize(num_trees);
        for (int i = 0; i < num_trees; ++i)
        {
            std::vector<std::vector<float>> subset_data;
            std::vector<int> subset_labels;

            BootstrapSample(data, labels, subset_data, subset_labels);

            trees[i] = new DecisionTree(max_depth);
            trees[i]->Train(subset_data, subset_labels);
        }
    }

    int Predict(const std::vector<float>& sample)
    {
        std::vector<int> votes(2, 0);

        for (int i = 0; i < num_trees; ++i)
        {
            int prediction = trees[i]->Predict(sample);
            votes[prediction]++;
        }

        return (votes[1] > votes[0]) ? 1 : 0;
    }

    ~RandomForest()
    {
        for (int i = 0; i < num_trees; ++i)
        {
            delete trees[i];
        }
    }

private:
    int num_trees;
    int max_depth;
    std::vector<DecisionTree*> trees;

    void BootstrapSample(const std::vector<std::vector<float>>& data, const std::vector<int>& labels,
                         std::vector<std::vector<float>>& subset_data, std::vector<int>& subset_labels)
    {
        int n = data.size();
        for (int i = 0; i < n; ++i)
        {
            int index = rand() % n;
            subset_data.push_back(data[index]);
            subset_labels.push_back(labels[index]);
        }
    }
};
