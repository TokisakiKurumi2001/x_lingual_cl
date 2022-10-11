import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class BertTwins(nn.Module):
    def __init__(self, teacher_model, student_tokenizer):
        super(BertTwins, self).__init__()
        self.config = BertConfig(vocab_size = student_tokenizer.vocab_size, num_hidden_layers=12, num_attention_heads=8, intermediate_size=1024)
        self.student_model = BertModel(self.config)
        self.teacher_model = teacher_model
        self.mlm_head = BertOnlyMLMHead(self.config)

    def export_model(self, path):
        self.student_model.save_pretrained(path)

    def forward_student(self, input):
        out = self.student_model(**input)
        student_linear = out.last_hidden_state
        cls_res = self.mlm_classfier(out)
        return student_linear, cls_res

    def forward_teacher(self, input):
        out = self.teacher_model(**input)
        return out.last_hidden_state

    def mlm_classfier(self, model_output):
        return self.mlm_head(model_output[0])

    def forward(self, input1, input_teacher, negative_inp):
        student_output, mlm_output = self.forward_student(input1)
        student_negative, mlm_negative = self.forward_student(negative_inp)
        teacher_output = self.forward_teacher(input_teacher)
        return teacher_output, student_output, mlm_output, student_negative, mlm_negative