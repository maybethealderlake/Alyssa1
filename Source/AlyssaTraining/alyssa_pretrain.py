import torch

from Source.AlyssaModel.alyssa import Alyssa
from Source.AlyssaDataset.alyssa_dataset import AlyssaPretrainDataset
from Source.AlyssaTokenizer.alyssa_rapid_tokenizer import AlyssaRapidTokenizer

alyssa = Alyssa(2048, 50142, 1536, 16,
                16, 0.3, 0.3, 0.3)

alyssa_rapid_tokenizer = AlyssaRapidTokenizer(50142)
alyssa_rapid_tokenizer.train("../../Resources/Data/Pretraining", "text", 100, 10)
alyssa_rapid_tokenizer.save("../../Resources/Data/Pretraining/alyssa_1_vocabulary.jsonl")

