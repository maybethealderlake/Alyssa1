#include "AlyssaRapidTokenizer.h"

#include <iostream>

using namespace std;

namespace fs = std::filesystem;
using json = nlohmann::json;

/**
 * @brief Alyssa Rapid Tokenizer constructor
 *
 * It initializes the vocabulary with special tokens and byte-level characters.
 * It also builds trie for rapid encoding
 *
 * @param vocabularySize The target vocabulary size
 */
AlyssaRapidTokenizer::AlyssaRapidTokenizer(const int vocabularySize) :
    vocabularySize(vocabularySize),
    trainingShardIteratorIsInitialized(false)
{
    // Initialize the tokenizer full training pass signal
    fullPassSignal = "tokenizer_training_pass_completed";

    // Initialize the special tokens
    padToken = "<|pad|>";
    unknownToken = "<|unk|>";
    startOfTextToken = "<|startoftext|>";
    endOfTextToken = "<|endoftext|>";

    initialSpecialTokens = {padToken, unknownToken, startOfTextToken, endOfTextToken};

    // Initialize the character tokens
    for (int i = 0; i < 256; ++i) {
        string byteToken(1, static_cast<unsigned char>(i));
        initialCharacterTokens.push_back(byteToken);
    }

    for (const string &token : initialSpecialTokens)
        initialTokens.push_back(token);

    for (const string &token : initialCharacterTokens)
        initialTokens.push_back(token);

    // Initialize the vocabulary with initial tokens
    for (size_t index = 0; index < initialTokens.size(); index++) {
        vocabulary[initialTokens[index]] = static_cast<int>(index);
        inverseVocabulary[index] = initialTokens[index];
    }
}

void AlyssaRapidTokenizer::Train(const string& trainingShardsDirectory, const string& jsonlKey, const size_t& chunkSize, const size_t& ioBufferSize) {

}

vector<int> AlyssaRapidTokenizer::Encode(const string& inputText) {

}

string AlyssaRapidTokenizer::Decode(const vector<int>& inputTokenIndices) {

}

void AlyssaRapidTokenizer::Save(const string& vocabularySavePath) {

}

void AlyssaRapidTokenizer::Load(const string& vocabularyPath) {

}

/**
 * @brief A generator that fetches a chunk from the training shard files
 *
 * Fetches a new chunk each time when called and returns a full pass signal
 * when all the shard files have been processed. Starts over again from the
 * first shard when all the shard files have been processed
 *
 * @param trainingShardsDirectory The directory which contains the training shards
 * @param jsonlKey The JSONL key which refers to the training sequence
 * @param chunkSize The chunk size for memory-efficient training in MBs(e.g. 1024MB)
 * @param ioBufferSize The IO buffer size for rapid training in MBs(e.g. 10MB)
 *
 * @return A text chunk string or signal indicating a full pass.
 */
string AlyssaRapidTokenizer::FetchTrainingChunk(const string& trainingShardsDirectory, const string& jsonlKey,
    const size_t& chunkSize, const size_t& ioBufferSize) {
    // Verify directory existence
    if (!fs::exists(trainingShardsDirectory) || !fs::is_directory(trainingShardsDirectory))
        return "";

    // Initialize iterator if directory changed or uninitialized
    if (!trainingShardIteratorIsInitialized || trainingShardsDirectory != currentTrainingShardDirectory) {
        currentTrainingShardDirectory = trainingShardsDirectory;
        trainingShardIterator = fs::directory_iterator(trainingShardsDirectory);
        trainingShardIteratorIsInitialized = true;
    }

    string chunk;

    // Move to next shard file if necessary
    if (!trainingShardStream.is_open() || trainingShardStream.eof()) {
        if (trainingShardStream.is_open()) trainingShardStream.close();

        if (trainingShardIterator != fs::end(trainingShardIterator)) {
            trainingShardStream.open(trainingShardIterator->path());
            size_t bufferSizeBytes = ioBufferSize * (1 << 20);
            trainingIOBuffer.resize(bufferSizeBytes);
            trainingShardStream.rdbuf()->pubsetbuf(trainingIOBuffer.data(), bufferSizeBytes);
            ++trainingShardIterator;
        } else {
            trainingShardIterator = fs::directory_iterator(currentTrainingShardDirectory);
            return fullPassSignal;
        }
    }

    // Read shard content until chunk limit reached
    string line;
    while (getline(trainingShardStream, line)) {
        json j = json::parse(line);
        string seq = j[jsonlKey];
        if (chunk.size() + seq.size() + 1 > chunkSize)
            return chunk;
        chunk += seq + "\n";
    }

    // If current shard ends, move to next
    while (chunk.empty()) {
        if (!trainingShardStream.is_open() || trainingShardStream.eof()) {
            if (trainingShardStream.is_open())
                trainingShardStream.close();

            if (trainingShardIterator == fs::end(trainingShardIterator)) {
                trainingShardIterator = fs::directory_iterator(currentTrainingShardDirectory);
                return fullPassSignal;
            }

            trainingShardStream.open(trainingShardIterator->path());
            size_t bufferSizeBytes = ioBufferSize * (1 << 20);
            trainingIOBuffer.resize(bufferSizeBytes);
            trainingShardStream.rdbuf()->pubsetbuf(trainingIOBuffer.data(), bufferSizeBytes);
            ++trainingShardIterator;
        }

        while (getline(trainingShardStream, line)) {
            json j = json::parse(line);
            string seq = j[jsonlKey];
            if (chunk.size() + seq.size() + 1 > chunkSize)
                return chunk;
            chunk += seq + "\n";
        }
    }

    return chunk;
}