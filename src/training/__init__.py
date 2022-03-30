

if __name__ == "__main__":
    trainer = pl.Trainer(gpus=1)
    trainer.fit(bert_finetuner)