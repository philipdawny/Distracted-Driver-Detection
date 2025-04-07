from transformers import TrainingArguments, Trainer
import evaluate

def compute_metrics(eval_pred):
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }

def train_vit(model, train_dataset, val_dataset, processor, config):
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy="steps",
        num_train_epochs=config["num_epochs"],
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=config["learning_rate"],
        save_total_limit=2,
        remove_unused_columns=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
