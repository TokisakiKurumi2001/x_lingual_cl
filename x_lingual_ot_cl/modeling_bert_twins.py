import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, DistilBertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class BertTwins(nn.Module):
    def __init__(
        self, teacher_model: str, vocab_size: int, num_hidden_layers: int,
        num_attention_heads: int, intermediate_size: int
    ):
        super(BertTwins, self).__init__()
        self.config = BertConfig(
            vocab_size = vocab_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size
        )
        self.student_model = BertModel(self.config)
        self.teacher_model = DistilBertModel.from_pretrained(teacher_model)
        for p in self.teacher_model.parameters():
            p.requires_grad = False

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