#ifndef ALYSSARAPIDTOKENIZER_LIBRARY_H
#define ALYSSARAPIDTOKENIZER_LIBRARY_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>

class AlyssaRapidTokenizer {
    public:
    AlyssaRapidTokenizer(int vocabularySize);

    void Train(const std::string& trainingShardsDirectory,const std::string& jsonlKey, const size_t& chunkSize,
        const size_t& ioBufferSize);

    std::vector<int> Encode(const std::string& inputText);
    std::string Decode(const std::vector<int>& inputTokenIndices);

    void Save(const std::string& vocabularySavePath);
    void Load(const std::string& vocabularyPath);

    private:
    std::string FetchTrainingChunk(const std::string& trainingShardsDirectory, const std::string& jsonlKey,
        const size_t& chunkSize, const size_t& ioBufferSize);

    int vocabularySize;

    std::map<std::string, int> vocabulary;
    std::map<int, std::string> inverseVocabulary;

    std::string padToken;
    std::string unknownToken;
    std::string startOfTextToken;
    std::string endOfTextToken;

    std::vector<std::string> initialSpecialTokens;
    std::vector<std::string> initialCharacterTokens;
    std::vector<std::string> initialTokens;

    std::ifstream trainingShardStream;
    std::filesystem::directory_iterator trainingShardIterator;
    std::vector<char> trainingIOBuffer;
    bool trainingShardIteratorIsInitialized;

    std::string currentTrainingShardDirectory;
    std::string fullPassSignal;
};

#endif //ALYSSARAPIDTOKENIZER_LIBRARY_H