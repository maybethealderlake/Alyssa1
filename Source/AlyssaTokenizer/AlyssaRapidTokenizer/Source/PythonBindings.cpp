#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "AlyssaRapidTokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(alyssa_rapid_tokenizer, m) {
     m.doc() = "Alyssa Rapid Tokenizer — super-fast byte-level BPE tokenizer for Alyssa";

     py::class_<AlyssaRapidTokenizer>(m, "AlyssaRapidTokenizer")
         .def(py::init<int>(), py::arg("vocabulary_size"),
              "Initialize the tokenizer with a target vocabulary size")

         .def("train", &AlyssaRapidTokenizer::Train,
              py::arg("training_shards_directory"),
              py::arg("json_key"),
              py::arg("chunk_size"),
              py::arg("io_buffer_size"),
              "Train the tokenizer using shard-based JSONL datasets")

         .def("encode", &AlyssaRapidTokenizer::Encode,
              py::arg("input_text"),
              "Encode a string into a list of token IDs")

         .def("decode", &AlyssaRapidTokenizer::Decode,
              py::arg("input_token_indices"),
              "Decode a list of token IDs back into text")

         .def("save", &AlyssaRapidTokenizer::Save,
              py::arg("vocabulary_save_path"),
              "Save the tokenizer vocabulary to a JSONL file")

         .def("load", &AlyssaRapidTokenizer::Load,
              py::arg("vocabulary_path"),
              "Load the tokenizer vocabulary from a JSONL file");
}
