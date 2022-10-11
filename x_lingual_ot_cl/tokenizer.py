from transformers import DistilBertTokenizerFast, AutoTokenizer

# teacher tokenizer
teacher_tokenizer = DistilBertTokenizerFast.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased")

# student tokenizer
student_tokenizer = AutoTokenizer.from_pretrained("ba_tokenizer")