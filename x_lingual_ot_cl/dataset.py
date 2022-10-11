from datasets import load_dataset
dataset = load_dataset(
    'json',
    data_files=[
        'nmt_bana_data/clean_typo/train.json',
        'nmt_bana_data/clean_typo/test.json',
        'nmt_bana_data/clean_typo/valid.json'
    ])