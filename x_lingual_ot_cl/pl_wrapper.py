import torch
import torch.nn as nn
from geomloss import SamplesLoss
import pytorch_lightning as pl
from x_lingual_ot_cl import BertTwins

class LitBertTwins(pl.LightningModule):
    def __init__(
        self, teacher_model: str, vocab_size: int, num_hidden_layers: int,
        num_attention_heads: int, intermediate_size: int
    ):
        super().__init__()
        self.bert_twins = BertTwins(
            teacher_model, vocab_size, num_hidden_layers,
            num_attention_heads, intermediate_size
        )
        self.ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.main_loss = nn.CrossEntropyLoss()
        self.consine = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.save_hyperparameters()

    def export_model(self, path):
        self.bert_twins.export_model(path)

    def training_step(self, batch, batch_idx):
        teacher, student, negative = batch
        student_label = student.pop('labels')

        teacher_output, student_output, mlm_output, negative_output, mlm_negative = self.bert_twins(student, teacher, negative)
        # Classifier loss
        loss_cls = self.main_loss(mlm_output.view(-1, self.bert_twins.config.vocab_size), student_label.view(-1))
        self.log("train/loss_cls", loss_cls, sync_dist=True)

        # OT loss
        ot_p = self.ot_loss(teacher_output, student_output).mean()
        self.log("train/ot_p", ot_p, sync_dist=True)

        # contrastive loss
        s_p = torch.div(self.consine(teacher_output, student_output), 0.05)
        s_n = torch.div(self.consine(student_output, negative_output), 0.05)
        cl_loss = (-1) * torch.log(torch.div(torch.exp(s_p), torch.exp(s_p) + torch.exp(s_n))).mean()

        self.log("train/cl_loss", cl_loss, sync_dist=True)

        # final loss
        loss = loss_cls + 0.1*ot_p + 0.5*cl_loss
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        teacher, student, negative = batch
        student_label = student.pop('labels')

        teacher_output, student_output, mlm_output, negative_output, mlm_negative = self.bert_twins(student, teacher, negative)
        # Classifier loss
        loss_cls = self.main_loss(mlm_output.view(-1, self.bert_twins.config.vocab_size), student_label.view(-1))
        self.log("valid/loss_cls", loss_cls, sync_dist=True)

        # OT loss
        ot_p = self.ot_loss(teacher_output, student_output).mean()
        self.log("valid/ot_p", ot_p, sync_dist=True)

        # contrastive loss
        s_p = torch.div(self.consine(teacher_output, student_output), 0.05)
        s_n = torch.div(self.consine(student_output, negative_output), 0.05)
        cl_loss = (-1) * torch.log(torch.div(torch.exp(s_p), torch.exp(s_p) + torch.exp(s_n))).mean()

        self.log("valid/cl_loss", cl_loss, sync_dist=True)

        # final loss
        loss = loss_cls + 0.1*ot_p + 0.5*cl_loss
        self.log("valid/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer