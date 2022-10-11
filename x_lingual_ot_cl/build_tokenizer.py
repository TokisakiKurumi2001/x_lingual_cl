from dataset import dataset

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="<cls> $A <sep>",
    special_tokens=[
        ("<cls>", 2),
        ("<sep>", 3)
    ],
)

from collections.abc import Mapping
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        examples = dataset[i: i + batch_size]["translation"]
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        yield encoded_inputs["ba"]

from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(
    vocab_size=6000, special_tokens=["<unk>", "<pad>", "<cls>", "<sep>", "<mask>"]
)

tokenizer.train_from_iterator(batch_iterator(dataset['train']), trainer=trainer, length=len(dataset['train']))

tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
tokenizer.enable_truncation(max_length=256)
from tokenizers import decoders
tokenizer.decoder = decoders.WordPiece()

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    mask_token="<mask>",
    sep_token="<sep>"
)

wrapped_tokenizer.save_pretrained("ba_tokenizer")