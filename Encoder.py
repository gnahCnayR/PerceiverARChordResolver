from typing import List, Dict, Any
from pathlib import Path
import json

from torch import Tensor, LongTensor, stack, flip, cat, full, argmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtoolkit.data import create_subsets
from miditok import REMI, MIDITokenizer

class MIDIDataset(Dataset):
    r"""Dataset for generator training

    :param files_paths: list of paths to files to load.
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int, tokenizer: MIDITokenizer = None):
        samples = []

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0].ids
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        self.samples = samples

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        return {"input_ids": self.samples[idx], "labels": self.samples[idx]}

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


def _pad_batch(examples: List[Dict[str, LongTensor]], pad_token: int) -> LongTensor:
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    length_of_first = examples[0]["input_ids"].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x["input_ids"].size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return stack([e["input_ids"] for e in examples], dim=0).long()

    # Creating the full tensor and filling it with our data.
    return pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=pad_token).long()


class DataCollatorGen(Dataset): ## DataCollatorMixin
    def __init__(self, pad_token: int, return_tensors: str = "pt"):
        """Collator that simply pad the input sequences.
        Input_ids will be padded with the pad token given, while labels will be
        padded with -100.

        :param pad_token: pas token
        :param return_tensors:
        """
        self.pad_token = pad_token
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, LongTensor]:
        x, y = _pad_batch(batch, self.pad_token), _pad_batch(batch, -100)
        return {"input_ids": x, "labels": y}  # will be shifted in GPT2LMHead forward


from miditok import REMI, MIDITokenizer
from miditok.constants import CHORD_MAPS
from miditoolkit import MidiFile
from tqdm import tqdm


# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250),  # (min, max)
                     'Program': False,
                     "chord_maps": CHORD_MAPS,
                     "chord_tokens_with_root_note": True,
                     "chord_unknown": False}
special_tokens = ["PAD", "BOS", "EOS"]

# Creates the tokenizer convert MIDIs to tokens
tokens_path = Path('Maestro_tokens_no_bpe')
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens=special_tokens) # REMI
midi_paths = list(Path('Maestro').glob('**/*.mid')) + list(Path('Maestro').glob('**/*.midi'))
tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

# Learn and apply BPE to data we just tokenized
tokens_bpe_path = Path('Maestro_tokens_bpe')
tokens_bpe_path.mkdir(exist_ok=True, parents=True)
tokenizer.learn_bpe(
    vocab_size=1000,
    tokens_paths=list(tokens_path.glob("**/*.json")),
    start_from_empty_voc=False,
)
tokenizer.apply_bpe_to_dataset(
    tokens_path,
    tokens_bpe_path,
)

# Loads tokens and create data loaders for training
tokens_paths = list(Path('Maestro_tokens_bpe').glob("**/*.json"))
dataset = MIDIDataset(
    tokens_paths, max_seq_len=512, min_seq_len=384,
)
subset_train, subset_valid = create_subsets(dataset, [0.3])