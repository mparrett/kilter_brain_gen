import pprint
import time
import sys
from pathlib import Path
from rich.console import Console
console = Console()
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, AutoConfig, pipeline, Pipeline

def get_newest_checkpoint(dir):
    checkpoints = Path(dir).glob("zcheckpoint-*")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)\
        .name.split("-")[1] if any(checkpoints) else None

token_dir = "name-model"
checkpoint = get_newest_checkpoint(token_dir)

model_dir = Path(token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint)
if not model_dir.exists():
    print(f"Checkpoint not found: {model_dir}")
    sys.exit(1)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="left")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
#config = GPT2Config.from_pretrained("gpt2")
context_length = 128
config =  AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    n_layer=3)

model = GPT2LMHeadModel.from_pretrained(model_dir, config=config)


generator: Pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

params = [
    {
        "do_sample": True,
    },
    {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.8
    },
    {
        "do_sample": True,
        "top_k": 100,
        "temperature": 0.8
    },
    {
        "do_sample": True,
        "num_beams": 1
    },
]

bad_params = [
    {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.2
    },
    {
        "do_sample": True,
        "top_k": 25,
        "temperature": 0.2
    },
    {
        "do_sample": False,
        "num_beams": 1
    },
    {
        "do_sample": True,
        "num_beams": 4
    },
 
]

#   The Big Big Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy Daddy
for p in params:
    #p['pad_token_id'] = tokenizer.eos_token_id
    console.print(p, style="bold orange1")
    start = time.time()
    for _ in range(5):
        out = generator("", **p, prefix=tokenizer.eos_token)
        #print(out)
        out = out[0]['generated_text']
        console.print("  " + out, style="bold green")
    end = time.time()
    print("%.2fms" % ((end - start) * 1000))
    print()