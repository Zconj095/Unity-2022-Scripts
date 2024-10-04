#include <vector>
#include <limits>
#include <algorithm>

class DecisionTreeNode
{
public:
    DecisionTreeNode* left;
    DecisionTreeNode* right;
    int feature_index;
    float threshold;
    int label;
    bool is_leaf;

    DecisionTreeNode() : left(nullptr), right(nullptr), feature_index(-1), threshold(0), label(-1), is_leaf(false) {}

    ~DecisionTreeNode()
    {
        delete left;
        delete right;
    }
};

class DecisionTree
{
public:
    DecisionTree(int max_depth) : max_depth(max_depth), root(nullptr) {}

    ~DecisionTree()
    {
        delete root;
    }

    void Train(const std::vector<std::vector<float>>& data, const std::vector<int>& labels)
    {
        root = BuildTree(data, labels, 0);
    }

    int Predict(const std::vector<float>& sample)
    {
        return PredictRecursive(root, sample);
    }

private:
    DecisionTreeNode* root;
    int max_depth;

    DecisionTreeNode* BuildTree(const std::vector<std::vector<float>>& data, const std::vector<int>& labels, int depth)
    {
        if (depth == max_depth || data.empty())
        {
            return CreateLeafNode(labels);
        }

        int best_feature = -1;
        float best_threshold = std::numeric_limits<float>::max();
        float best_gini = std::numeric_limits<float>::max();

        std::vector<std::vector<float>> left_data, right_data;
        std::vector<int> left_labels, right_labels;

        for (int feature = 0; feature < data[0].size(); ++feature)
        {
            std::vector<float> feature_values(data.size());
            for (int i = 0; i < data.size(); ++i)
            {
                feature_values[i] = data[i][feature];
            }

            std::sort(feature_values.begin(), feature_values.end());
            for (int i = 1; i < feature_values.size(); ++i)
            {
                float threshold = (feature_values[i] + feature_values[i - 1]) / 2;

                std::vector<std::vector<float>> left_split, right_split;
                std::vector<int> left_split_labels, right_split_labels;

                for (int j = 0; j < data.size(); ++j)
                {
                    if (data[j][feature] <= threshold)
                    {
                        left_split.push_back(data[j]);
                        left_split_labels.push_back(labels[j]);
                    }
                    else
                    {
                        right_split.push_back(data[j]);
                        right_split_labels.push_back(labels[j]);
                    }
                }

                float gini = CalculateGini(left_split_labels, right_split_labels);
                if (gini < best_gini)
                {
                    best_gini = gini;
                    best_feature = feature;
                    best_threshold = threshold;
                    left_data = left_split;
                    right_data = right_split;
                    left_labels = left_split_labels;
                    right_labels = right_split_labels;
                }
            }
        }

        if (best_feature == -1)
        {
            return CreateLeafNode(labels);
        }

        DecisionTreeNode* node = new DecisionTreeNode();
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = BuildTree(left_data, left_labels, depth + 1);
        node->right = BuildTree(right_data, right_labels, depth + 1);

        return node;
    }

    float CalculateGini(const std::vector<int>& left_labels, const std::vector<int>& right_labels)
    {
        auto gini = [](const std::vector<int>& labels)
        {
            int total = labels.size();
            if (total == 0) return 0.0f;

            int count1 = std::count(labels.begin(), labels.end(), 1);
            int count0 = total - count1;

            float prob1 = static_cast<float>(count1) / total;
            float prob0 = static_cast<float>(count0) / total;

            return 1.0f - (prob1 * prob1 + prob0 * prob0);
        };

        int total = left_labels.size() + right_labels.size();
        return (left_labels.size() * gini(left_labels) + right_labels.size() * gini(right_labels)) / total;
    }

    DecisionTreeNode* CreateLeafNode(const std::vector<int>& labels)
    {
        DecisionTreeNode* leaf = new DecisionTreeNode();
        leaf->is_leaf = true;

        int count1 = std::count(labels.begin(), labels.end(), 1);
        leaf->label = (count1 > labels.size() / 2) ? 1 : 0;

        return leaf;
    }

    int PredictRecursive(DecisionTreeNode* node, const std::vector<float>& sample)
    {
        if (node->is_leaf)
        {
            return node->label;
        }

        if (sample[node->feature_index] <= node->threshold)
        {
            return PredictRecursive(node->left, sample);
        }
        else
        {
            return PredictRecursive(node->right, sample);
        }
    }
};
