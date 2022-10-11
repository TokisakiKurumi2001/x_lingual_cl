from transformers import DistilBertModel, DistilBertTokenizerFast

teacher_ck = "sentence-transformers/distiluse-base-multilingual-cased"
teacher_model = DistilBertModel.from_pretrained(teacher_ck)
teacher_tokenizer = DistilBertTokenizerFast.from_pretrained(teacher_ck)
for p in teacher_model.parameters():
    p.requires_grad = False