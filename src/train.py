from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf" 
model_path = "./models/llama-7b"
data_folder = "./data/alpaca_small"
download=False



if download:
    # # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    # Save locally
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # Just 100 samples for fast test
    dataset.save_to_disk(data_folder)
    

model = AutoModelForCausalLM.from_pretrained(
   model_path,
    device_map="auto",
    load_in_4bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

data = load_from_disk(data_folder)

def tokenize(batch):
    inputs = batch["input"]
    outputs = batch["output"]

    # Ensure all elements are strings
    combined_texts = []
    for inp, out in zip(inputs, outputs):
        if isinstance(inp, list):
            inp = " ".join(inp)
        if isinstance(out, list):
            out = " ".join(out)
        combined_texts.append(inp + "\n" + out)

    return tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized = data.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="./output/results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./output/checkpoint/qlora-checkpoint")
