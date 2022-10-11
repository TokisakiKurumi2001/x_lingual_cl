from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from x_lingual_ot_cl import LitBertTwins, student_model, train_dataloader, valid_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_x_lingual_ot_cl")

    # model
    lit_bert_twins = LitBertTwins(student_model)

    # train model
    trainer = pl.Trainer(max_epochs=30, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp")
    trainer.fit(model=lit_bert_twins, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_bert_twins.export_model("banabert_ot_cl")
