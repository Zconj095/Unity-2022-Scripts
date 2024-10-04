#include <fstream>
#include <string>
#include <vector>
#include <poppler/cpp/poppler-document.h>  // Example for PDF parsing
#include <pugixml.hpp>  // Example for DOCX parsing

class FileIngestor
{
public:
    // Ingest content from a PDF file
    std::string IngestPDF(const std::string& filePath)
    {
        std::string content;
        poppler::document* doc = poppler::document::load_from_file(filePath);
        if (doc)
        {
            for (int i = 0; i < doc->pages(); ++i)
            {
                content += doc->create_page(i)->text().to_latin1();
            }
        }
        delete doc;
        return content;
    }

    // Ingest content from a DOCX file
    std::string IngestDOCX(const std::string& filePath)
    {
        std::string content;
        pugi::xml_document doc;
        if (doc.load_file(filePath.c_str()))
        {
            pugi::xml_node body = doc.child("w:document").child("w:body");
            for (pugi::xml_node para = body.child("w:p"); para; para = para.next_sibling("w:p"))
            {
                for (pugi::xml_node text = para.child("w:r").child("w:t"); text; text = text.next_sibling("w:t"))
                {
                    content += text.child_value();
                }
                content += "\n";
            }
        }
        return content;
    }

    // Ingest content from a Python or C++ file
    std::string IngestCode(const std::string& filePath)
    {
        std::ifstream file(filePath);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return content;
    }
};
