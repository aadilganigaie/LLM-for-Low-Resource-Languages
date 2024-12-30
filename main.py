from transformers import AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("mistralai/mistral-7b-v0.1", use_auth_token=access_token)
import os

with open('kashmiri_corpus.txt', 'r') as f:
    lines = f.readlines()
  
options = dict(
    input="kashmiri_corpus.txt",
    input_format="text",
    model_prefix="mistral_tel_tokenizer",
    model_type="bpe",
    vocab_size=16000,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,
    max_sentence_length=1073741824,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=1.0,
    byte_fallback=True,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
    num_threads=os.cpu_count(),
)

import sentencepiece as spm

def main():
    spm.SentencePieceTrainer.train(**options)

main()

import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='/content/mistral_tel_tokenizer.model')
vocab = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
len(vocab)

tel_text = "اگر أس باقی اؠجِکؠشن سِسٹم وُچھو"

sp_encode_ids = sp.encode(tel_text)
sp_decode_text = sp.decode(sp_encode_ids)

mistral_encode_ids = tokenizer1.encode(tel_text)
mistral_decode_text = tokenizer1.decode(mistral_encode_ids)

print("Length of kashmiri text: ", len(tel_text))
print('--------------')
print('NEW TOKENIZER')
print('--------------')
print("Length of encoded IDs: ", len(sp_encode_ids))
print('---')
print("Compression ratio: ", f"{len(sp_encode_ids) / len(tel_text):.2f}")
print('---')
print("Encoded token IDs: ", sp_encode_ids)
print('---')
print("Decoded text: ", sp_decode_text)
print('--------------')
print('MISTRAL TOKENIZER')
print('--------------')
print("Length of encoded IDs: ", len(mistral_encode_ids))
print('---')
print("Compression ratio: ", f"{len(mistral_encode_ids) / len(tel_text):.2f}")
print('---')
print("Encoded token IDs: ", mistral_encode_ids)
print('---')
print("Decoded text: ", mistral_decode_text)
import os
import re
from transformers import LlamaTokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

original_tokenizer_path = hf_hub_download(repo_id="mistralai/mistral-7b-v0.1", filename="tokenizer.model", local_dir="original_tokenizer", token=access_token)
tokenizer = LlamaTokenizer.from_pretrained(original_tokenizer_path)
original_tokenizer_spm = sp_pb2_model.ModelProto()
original_tokenizer_spm.ParseFromString(open(original_tokenizer_path, "rb").read())

new_tokenizer_spm = sp_pb2_model.ModelProto()
new_tokenizer_spm.ParseFromString(open("/content/mistral_tel_tokenizer.model", "rb").read())

def contains_eng(text):
    eng_pattern = re.compile(r"[\u0020-\u007E]+")
    return True if eng_pattern.search(text) else False

original_tokenizer_tokenset = set(p.piece for p in original_tokenizer_spm.pieces)
print(f"Number of tokens before merge: {len(original_tokenizer_tokenset)}")
for p in new_tokenizer_spm.pieces:
    piece = p.piece
    if piece not in original_tokenizer_tokenset and not contains_eng(piece):
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        original_tokenizer_spm.pieces.append(new_p)
print(f"Number of tokens after merge: {len(original_tokenizer_spm.pieces)}")

extended_tokenizer_save_path = "/content/english-kashmiri-tokenizer"
os.makedirs(extended_tokenizer_save_path, exist_ok=True)
with open(os.path.join(extended_tokenizer_save_path, "tokenizer.model"), "wb") as f:
    f.write(original_tokenizer_spm.SerializeToString())
import sentencepiece as spm

sp_new = spm.SentencePieceProcessor(model_file='/content/english-kashmiri-tokenizer/tokenizer.model')
new_vocab = [sp_new.id_to_piece(id) for id in range(sp_new.get_piece_size())]
len(new_vocab)

tel_text = "اگر أس باقی اؠجِکؠشن سِسٹم وُچھو"
en_text = "I am fine. How are you?"

sp_new_encode_ids = sp_new.encode(tel_text)
sp_new_encode_ids_en = sp_new.encode(en_text)
sp_new_decode_text = sp_new.decode(sp_new_encode_ids)
sp_new_decode_text_en = sp_new.decode(sp_new_encode_ids_en)

mistral_encode_ids = tokenizer1.encode(en_text)
mistral_decode_text = tokenizer1.decode(mistral_encode_ids)

print("Length of kashmiri text: ", len(tel_text))
print('---')
print("Length of english text: ", len(en_text))
print('--------------')
print('EXTENDED MISTRAL TOKENIZER')
print('--------------')
print('Kashmiri Performance')
print('---')
print("Length of encoded IDs: ", len(sp_new_encode_ids))
print('---')
print("Compression ratio: ", f"{len(sp_new_encode_ids) / len(tel_text):.2f}")
print('---')
print("Encoded token IDs: ", sp_new_encode_ids)
print('---')
print("Decoded text: ", sp_new_decode_text)
print('--------------')
print('English Performance')
print('---')
print("Length of encoded IDs: ", len(sp_new_encode_ids_en))
print('---')
print("Compression ratio: ", f"{len(sp_new_encode_ids_en) / len(en_text):.2f}")
print('---')
print("Encoded token IDs: ", sp_new_encode_ids_en)
print('---')
print("Decoded text: ", sp_new_decode_text_en)
print('--------------')
print('MISTRAL TOKENIZER')
print('--------------')
print("Length of encoded IDs: ", len(mistral_encode_ids))
print('---')
print("Compression ratio: ", f"{len(mistral_encode_ids) / len(en_text):.2f}")
print('---')
print("Encoded token IDs: ", mistral_encode_ids)
print('---')
print("Decoded text: ", mistral_decode_text)

!pip uninstall protobuf --y
!pip install -q --no-binary=protobuf protobuf

from transformers import LlamaTokenizer
extended_tokenizer_save_path = "/content/english-kashmiri-tokenizer-hf"

hf_tokenizer = LlamaTokenizer(vocab_file="/content/english-kashmiri-tokenizer/tokenizer.model", legacy=False)
hf_tokenizer.save_pretrained(extended_tokenizer_save_path)
print(f"Tokenizer saved to {extended_tokenizer_save_path}")

tok1 = LlamaTokenizer.from_pretrained("mistralai/mistral-7b-v0.1")
tok2 = LlamaTokenizer.from_pretrained("/content/english-kashmiri-tokenizer-hf")

mismatch_found = False
for i in range(len(tok1)):
    if tok1.convert_ids_to_tokens(i) != tok2.convert_ids_to_tokens(i):
        print(f"Token mismatch at index {i}.")
        mismatch_found = True
        break

if not mismatch_found:
    print("No mismatch found. The english vocabularies of the two tokenizers match.")

tel_text = "اگر أس باقی اؠجِکؠشن سِسٹم وُچھو"
en_text = "I am fine. How are you?"

sp_new_encode_ids = tok2.encode(tel_text)
sp_new_encode_ids_en = tok2.encode(en_text)
sp_new_decode_text = tok2.decode(sp_new_encode_ids)
sp_new_decode_text_en = tok2.decode(sp_new_encode_ids_en)

mistral_encode_ids = tok1.encode(en_text)
mistral_decode_text = tok1.decode(mistral_encode_ids)

print("Length of kashmiri text: ", len(tel_text))
print('---')
print("Length of english text: ", len(en_text))
print('--------------')
print('EXTENDED MISTRAL TOKENIZER')
print('--------------')
print('Kashmiri Performance')
print('---')
print("Length of encoded IDs: ", len(sp_new_encode_ids))
print('---')
print("Compression ratio: ", f"{len(sp_new_encode_ids) / len(tel_text):.2f}")
print('---')
print("Encoded token IDs: ", sp_new_encode_ids)
print('---')
print("Decoded text: ", sp_new_decode_text)
print('--------------')
print('English Performance')
print('---')
print("Length of encoded IDs: ", len(sp_new_encode_ids_en))
print('---')
print("Compression ratio: ", f"{len(sp_new_encode_ids_en) / len(en_text):.2f}")
print('---')
print("Encoded token IDs: ", sp_new_encode_ids_en)
print('---')
print("Decoded text: ", sp_new_decode_text_en)
print('--------------')
print('MISTRAL TOKENIZER')
print('--------------')
print("Length of encoded IDs: ", len(mistral_encode_ids))
print('---')
print("Compression ratio: ", f"{len(mistral_encode_ids) / len(en_text):.2f}")
print('---')
print("Encoded token IDs: ", mistral_encode_ids)
print('---')
print("Decoded text: ", mistral_decode_text)
from datasets import Dataset

with open('kashmiri_corpus.txt', 'r') as f:
    lines = f.readlines()

dataset = Dataset.from_dict({'text': lines})
print(dataset['text'][:3])

tokenizer = AutoTokenizer.from_pretrained("/content/english-kashmiri-tokenizer-hf")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to {tokenizer.pad_token}")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset.train_test_split(test_size=0.2)['train']
val_dataset = tokenized_dataset.train_test_split(test_size=0.2)['test']

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/mistral-7b-v0.1", use_auth_token=access_token, device_map='auto')

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=5e-5,
    fp16=True,
    gradient_accumulation_steps=16,
    load_best_model_at_end=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
