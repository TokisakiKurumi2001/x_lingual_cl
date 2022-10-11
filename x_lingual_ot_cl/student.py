from transformers import AutoTokenizer
from x_lingual_ot_cl import BertTwins, teacher_model
student_tokenizer = AutoTokenizer.from_pretrained("ba_tokenizer")
student_model = BertTwins(teacher_model, student_tokenizer)