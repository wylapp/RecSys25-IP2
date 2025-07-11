import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


def build_optimizer(model, total_steps, args, stage="train"):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # split into two groups, bert related or not related, preparing for separate lr settings.
    bert_related_params = [(n, p) for (n, p) in model.named_parameters() if 'bert' in n]
    nobert_params = [(n, p) for (n, p) in model.named_parameters() if 'bert' not in n]


    optimizer_grouped_parameters = [
        # bert params
        {
            "params": [
                p for n, p in bert_related_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": float(args.bert_lr)
        },
        {
            "params": [p for n, p in bert_related_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": float(args.bert_lr)
        },
        # no-bert params
        {
            "params": [
                p for n, p in nobert_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": float(args.lr) if stage == "train" else float(args.pt_lr)
        },
        {
            "params": [p for n, p in nobert_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": float(args.lr) if stage == "train" else float(args.pt_lr)
        },
    ]
    if stage == "train":
        
        optimizer = optim.AdamW(optimizer_grouped_parameters, eps=float(args.adam_epsilon))
    else:
        # pretrain needs small lr
        optimizer = optim.AdamW(optimizer_grouped_parameters, eps=float(args.adam_epsilon))
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(args.warmup_proportion * total_steps),
    #     num_training_steps=total_steps,
    # )
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
    )

    return optimizer, scheduler