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
