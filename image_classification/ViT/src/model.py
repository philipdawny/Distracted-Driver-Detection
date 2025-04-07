from transformers import ViTForImageClassification

def get_vit_model(config, class_to_idx):
    
    id2label = {id: label for label, id in class_to_idx.items()}
    label2id = {label: id for id, label in id2label.items()}
    
    return ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=len(class_to_idx),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
