import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class PretrainedBertTwins(nn.Module):
    def __init__(self, teacher_ckpt: str, student_ckpt: str):
        super(PretrainedBertTwins, self).__init__()
        self.student_model = BertModel.from_pretrained(student_ckpt)
        self.config = self.student_model.config
        self.teacher_model = DistilBertModel.from_pretrained(teacher_ckpt)
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