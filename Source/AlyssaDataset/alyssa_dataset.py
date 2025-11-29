import torch

from torch.utils.data import Dataset

class AlyssaPretrainDataset(Dataset):
    def __init__(self, dataset, context_window, stride):
        self.input_sequences = []
        self.output_sequences = []

        for i in range(0, len(dataset) - context_window, stride):
            input_sequence = dataset[i:i + context_window]
            output_sequence = dataset[i + 1:i + context_window + 1]

            self.input_sequences.append(input_sequence)
            self.output_sequences.append(output_sequence)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_sequences[index], dtype=torch.long),
            torch.tensor(self.output_sequences[index], dtype=torch.long)
        )
