#include <string>
#include <vector>

class LanguageModelInterface
{
public:
    LanguageModelInterface()
    {
        vectorDB = new VectorDatabase();
        fileIngestor = new FileIngestor();
        embeddingGenerator = new EmbeddingGenerator();
    }

    ~LanguageModelInterface()
    {
        delete vectorDB;
        delete fileIngestor;
        delete embeddingGenerator;
    }

    // Ingest and store vectors from various file types
    void IngestAndStoreFile(const std::string& filePath, const std::string& fileType)
    {
        std::string content;
        if (fileType == "pdf")
        {
            content = fileIngestor->IngestPDF(filePath);
        }
        else if (fileType == "docx")
        {
            content = fileIngestor->IngestDOCX(filePath);
        }
        else if (fileType == "py" || fileType == "cpp")
        {
            content = fileIngestor->IngestCode(filePath);
        }

        if (!content.empty())
        {
            std::vector<float> embedding = embeddingGenerator->GenerateEmbedding(content);
            vectorDB->AddVector(filePath, embedding);
        }
    }

    // Query the vector database
    std::string QueryDatabase(const std::string& queryText)
    {
        std::vector<float> queryEmbedding = embeddingGenerator->GenerateEmbedding(queryText);
        return vectorDB->Query(queryEmbedding);
    }

    // Optionally fine-tune the model based on user feedback or additional data
    void FineTuneModel(const std::string& newContent)
    {
        std::vector<float> newEmbedding = embeddingGenerator->GenerateEmbedding(newContent);
        // Process and integrate new embeddings to fine-tune the model
    }

private:
    VectorDatabase* vectorDB;
    FileIngestor* fileIngestor;
    EmbeddingGenerator* embeddingGenerator;
};
