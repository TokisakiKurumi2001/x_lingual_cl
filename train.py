from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from x_lingual_ot_cl import LitBertTwins, student_tokenizer, train_dataloader, valid_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_x_lingual_ot_cl")

    # model
    teacher_ckpt = "sentence-transformers/distiluse-base-multilingual-cased"
    lit_bert_twins = LitBertTwins(
        teacher_ckpt, student_tokenizer.vocab_size, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=2048
    )

    # train model
    trainer = pl.Trainer(max_epochs=30, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp")
    trainer.fit(model=lit_bert_twins, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_bert_twins.export_model("banabert_ot_cl")
