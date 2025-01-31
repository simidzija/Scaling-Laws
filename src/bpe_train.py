# Third-party
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def train(files: list[str], tokenizer_path: str, vocab_size: int) -> None:
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialzie trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])

    # Train on corpus
    tokenizer.train(files, trainer)

    # Save
    tokenizer.save(tokenizer_path)






