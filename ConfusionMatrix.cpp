#include <vector>
#include <iostream>

class ConfusionMatrix
{
public:
    ConfusionMatrix()
    {
        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;
    }

    void Update(int actual, int predicted)
    {
        if (actual == 1 && predicted == 1)
        {
            TP++;
        }
        else if (actual == 0 && predicted == 0)
        {
            TN++;
        }
        else if (actual == 0 && predicted == 1)
        {
            FP++;
        }
        else if (actual == 1 && predicted == 0)
        {
            FN++;
        }
    }

    void Display()
    {
        std::cout << "Confusion Matrix:" << std::endl;
        std::cout << "                Predicted Positive | Predicted Negative" << std::endl;
        std::cout << "Actual Positive     " << TP << "              | " << FN << std::endl;
        std::cout << "Actual Negative     " << FP << "              | " << TN << std::endl;
    }

    float Accuracy()
    {
        return float(TP + TN) / float(TP + TN + FP + FN);
    }

    float Precision()
    {
        return float(TP) / float(TP + FP);
    }

    float Recall()
    {
        return float(TP) / float(TP + FN);
    }

    float F1Score()
    {
        float precision = Precision();
        float recall = Recall();
        return 2 * (precision * recall) / (precision + recall);
    }

private:
    int TP, TN, FP, FN;
};
