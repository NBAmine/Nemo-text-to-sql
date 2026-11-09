import math

import torch
import wandb
from transformers import TrainerCallback


class SQLSampleCallback(TrainerCallback):
    def __init__(
        self,
        run,
        eval_dataset,
        tokenizer,
        model,
        samples_num,
        seed,
        is_main_process: bool = True,
    ):
        self.run = run
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.is_main_process = is_main_process
        self.samples = eval_dataset.shuffle(seed=seed).select(range(samples_num))

    def on_evaluate(self, args, state, control, **kwargs):
        # Only rank-0 should run inference and push W&B tables; non-zero ranks
        # skip this entirely to avoid duplicate logs and unnecessary GPU work.
        if not self.is_main_process:
            return

        epoch = state.epoch if state.epoch is not None else -1
        global_step = state.global_step
        device = self.model.device

        # average performance of the model during the period since the last N logging_steps
        train_loss = 0.0
        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log:
                    train_loss = log["loss"]
                    break

        columns = [
            "Epoch",
            "Step",
            "Train Loss",
            "Context",
            "Question",
            "True SQL",
            "Generated SQL",
        ]
        data = []

        self.model.eval()

        for sample in self.samples:
            text = sample["text"]

            context = text.split("Question:")[0].replace("Context: ", "").strip()
            question = text.split("Question:")[1].split("Answer:")[0].strip()
            true_sql = text.split("Answer:")[1].strip()

            inputs = self.tokenizer(
                f"Context: {context}\nQuestion: {question}\nAnswer:",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            gen_sql = (
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                .split("Answer:")[-1]
                .strip()
            )

            data.append(
                [epoch, global_step, train_loss, context, question, true_sql, gen_sql]
            )

        self.model.train()

        if self.run is not None:
            self.run.log({
                "train/global_step": global_step,
                "validation_samples": wandb.Table(data=data, columns=columns),
            })


def calculate_perplexity(model, tokenizer, test_dataset, device="cuda"):

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for sample in test_dataset:
        prompt = (
            f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer: "
        )
        target = sample["query"]
        full_text = prompt + target

        # Tokenize the prompt and the full text
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)
        full_ids = tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)

        prompt_len = prompt_ids.size(1)
        labels = full_ids.clone()

        # mask the prompt tokens in the labels to ignore them in loss calculation
        labels[:, :prompt_len] = -100

        target_tokens_len = full_ids.size(1) - prompt_len
        if target_tokens_len <= 0:
            continue

        with torch.no_grad():
            outputs = model(full_ids, labels=labels)
            loss = outputs.loss  # average loss over non masked tokens

            total_loss += loss.item() * target_tokens_len
            total_tokens += target_tokens_len

    # Calculate global perplexity
    ppl = math.exp(total_loss / total_tokens)
    return ppl
