import torch

class AlyssaTokenizer:
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.vocabulary = {}
        self.inverse_vocabulary = {}

        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"

        self.special_tokens = [
            self.pad_token, self.unk_token,
            self.bos_token, self.eos_token
        ]

        for index, token in enumerate(self.special_tokens):
            self.vocabulary[token] = index


    def train(self, dataset, min_pair_frequency):
        """Train BPE tokenizer on text data"""

        # Initialize the vocabulary with all the characters in the dataset
        dataset_list = list(dataset)

        initial_characters = list(set(dataset_list))
        initial_characters.sort()

        for token in initial_characters:
            # Prevent vocabulary overflow
            if len(self.vocabulary) < self.vocabulary_size:
                self.vocabulary[token] = max(self.vocabulary.values()) + 1
            else:
                break

        while len(self.vocabulary) < self.vocabulary_size:
            token_pair_frequencies = {}

            # Implement BPE tokenization
            for index in range(len(dataset_list) - 1):
                token_pair = dataset_list[index] + dataset_list[index + 1]
                token_pair_frequencies[token_pair] = token_pair_frequencies.get(token_pair, 0) + 1

            token_pair_frequencies = {pair: count for pair, count in token_pair_frequencies.items() if count > min_pair_frequency}

            if not token_pair_frequencies:
                break

            # Find the highest frequency token pair
            highest_frequency_pair = max(token_pair_frequencies, key=token_pair_frequencies.get)

            self.vocabulary[highest_frequency_pair] = max(self.vocabulary.values()) + 1

            updated_dataset_list = []

            index = 0
            while index < len(dataset_list):
                if index < len(dataset_list) - 1 and (dataset_list[index] + dataset_list[index + 1] == highest_frequency_pair):
                    updated_dataset_list.append(highest_frequency_pair)
                    index += 2
                else:
                    updated_dataset_list.append(dataset_list[index])
                    index += 1

            dataset_list = updated_dataset_list


    def encode(self, input_text):
        """Encode the text into token indices"""
        input_text_list = list(input_text)

        encoded_token_indices = []

        i = 0
        while i < len(input_text_list):
            j = i+1
            longest_token = ""
            while j <= len(input_text_list):
                piece = "".join(input_text_list[i:j])

                if piece in self.vocabulary and len(piece) > len(longest_token):
                    longest_token = piece

                j += 1

            if longest_token:
                encoded_token_indices.append(self.vocabulary.get(longest_token))
                i += len(longest_token)
            else:
                encoded_token_indices.append(self.vocabulary.get(self.unk_token))
                i += 1

        return encoded_token_indices

    def decode(self, input_token_indices):
        """Decode the token indices into text"""
        self.inverse_vocabulary = {index: token for token, index in self.vocabulary.items()}

        decoded_text = ""

        for token_index in input_token_indices:
            decoded_text += self.inverse_vocabulary.get(token_index, self.unk_token)

        return decoded_text

    # def load_tokenizer(self):