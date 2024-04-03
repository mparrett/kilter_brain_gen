import pprint
from datasets import Features, Value, load_dataset
from transformers import (
    TrainingArguments,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
)


out_dir = "name-model"

features = Features(
    {
        "name": Value("string"),
    }
)

dataset = load_dataset(
    "csv", data_files="climbs.csv", delimiter=",", features=features, split="train"
)
dataset = dataset.filter(lambda example: example["name"] is not None)

def wrap(example):
    example["name"] = example["name"] + "<|endoftext|>"
    return example

dataset = dataset.map(wrap)

datasets = dataset.train_test_split()

#config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.encode("TESTING", add_special_tokens=False, return_tensors="pt"))

config =  AutoConfig.from_pretrained("gpt2", n_layer=3)

model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
model.half()

tokenized_train = datasets["train"].map(
    lambda examples: tokenizer(examples["name"]), batched=True
)
tokenized_test = datasets["test"].map(
    lambda examples: tokenizer(examples["name"]), batched=True
)

pprint.pprint(datasets["train"][0])
pprint.pprint(tokenized_train[0])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir=out_dir,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=12,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=12,  # evaluation batch size
    logging_steps=200,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=200,
    report_to="tensorboard",
    remove_unused_columns=True,
    load_best_model_at_end=False,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,  # whether you don't have much space so you let only 3 model weights saved in the disk
    weight_decay=0.01, # ??? people seem to do this stuff when fine-tuning?
    learning_rate=1e-5 # ??? people seem to do this stuff when fine-tuning?
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

model.save_pretrained(out_dir)
