from x_lingual_ot_cl import dataset, student_tokenizer, teacher_tokenizer
import numpy as np
from torch.utils.data import DataLoader
from collections.abc import Mapping
from transformers import DataCollatorForLanguageModeling

mlm_collate_fn = DataCollatorForLanguageModeling(student_tokenizer)

def collate_fn(examples):
    if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
        encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
    
    teacher_input = encoded_inputs['vi']
    teacher_tok = teacher_tokenizer(teacher_input, padding='max_length', truncation=True, max_length=70, return_tensors='pt')
    
    student_input = encoded_inputs['ba']
    student_tok = student_tokenizer(student_input, padding='max_length', truncation=True, max_length=70, return_tensors='pt')
    format_student_tok = [{'input_ids': toks} for toks in student_tok['input_ids']]
    student_p_tok = mlm_collate_fn(format_student_tok)

    np.random.shuffle(student_input)
    student_tok = student_tokenizer(student_input, padding='max_length', truncation=True, max_length=70, return_tensors='pt')

    return teacher_tok, student_p_tok, student_tok

train_valid_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_valid_dataset.pop('train')
valid_dataset = train_valid_dataset.pop('test')
train_dataloader = DataLoader(train_dataset['translation'], batch_size=32, collate_fn=collate_fn, num_workers=24)
valid_dataloader = DataLoader(valid_dataset['translation'], batch_size=32, collate_fn=collate_fn, num_workers=24)