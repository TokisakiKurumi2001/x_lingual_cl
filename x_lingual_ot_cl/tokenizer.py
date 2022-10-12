from transformers import DistilBertTokenizerFast, BertTokenizerFast

# teacher tokenizer
teacher_tokenizer = DistilBertTokenizerFast.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased")

# student tokenizer
student_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")