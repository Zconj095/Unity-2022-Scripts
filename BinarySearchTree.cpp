class BinarySearchTree
{
public:
    BinarySearchTree() : root(nullptr) {}

    // Insert a key into the BST
    void Insert(int key)
    {
        root = InsertRec(root, key);
    }

    // Search for a key in the BST
    bool Search(int key)
    {
        return SearchRec(root, key) != nullptr;
    }

    // Delete a key from the BST
    void Delete(int key)
    {
        root = DeleteRec(root, key);
    }

    // Inorder traversal of the BST (sorted order)
    void InorderTraversal()
    {
        InorderRec(root);
        std::cout << std::endl;
    }

private:
    TreeNode* root;

    // Helper function to insert a key recursively
    TreeNode* InsertRec(TreeNode* node, int key)
    {
        if (node == nullptr)
        {
            return new TreeNode(key);
        }

        if (key < node->key)
        {
            node->left = InsertRec(node->left, key);
        }
        else if (key > node->key)
        {
            node->right = InsertRec(node->right, key);
        }

        return node;
    }

    // Helper function to search for a key recursively
    TreeNode* SearchRec(TreeNode* node, int key)
    {
        if (node == nullptr || node->key == key)
        {
            return node;
        }

        if (key < node->key)
        {
            return SearchRec(node->left, key);
        }

        return SearchRec(node->right, key);
    }

    // Helper function to delete a key recursively
    TreeNode* DeleteRec(TreeNode* node, int key)
    {
        if (node == nullptr) return node;

        if (key < node->key)
        {
            node->left = DeleteRec(node->left, key);
        }
        else if (key > node->key)
        {
            node->right = DeleteRec(node->right, key);
        }
        else
        {
            // Node with only one child or no child
            if (node->left == nullptr)
            {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            }
            else if (node->right == nullptr)
            {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }

            // Node with two children: Get the inorder successor (smallest in the right subtree)
            TreeNode* temp = MinValueNode(node->right);

            // Copy the inorder successor's content to this node
            node->key = temp->key;

            // Delete the inorder successor
            node->right = DeleteRec(node->right, temp->key);
        }

        return node;
    }

    // Helper function to find the node with the minimum key value
    TreeNode* MinValueNode(TreeNode* node)
    {
        TreeNode* current = node;

        // Loop down to find the leftmost leaf
        while (current && current->left != nullptr)
        {
            current = current->left;
        }

        return current;
    }

    // Helper function for inorder traversal
    void InorderRec(TreeNode* node)
    {
        if (node != nullptr)
        {
            InorderRec(node->left);
            std::cout << node->key << " ";
            InorderRec(node->right);
        }
    }
};
