import csv
import argparse
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
import torch
import datasets
import pandas as pd

DEFAULT_SYS_PROMPT = "Assume you are a pretend retrieval system and you are expected generate user queries that are purely procedural/how-to in nature that might lead users who are python programmers who need help programming to the given text and nothing more (no jokes, no comments, no conversations, no percieved responses,just start and your answer with a list of 5 queries)"


def prompt(prompt) -> str:
    prompt = f"<s>[INST] <<SYS>>\n{DEFAULT_SYS_PROMPT}\n<</SYS>>\n\n :for {prompt}, queries are : [/INST]"
    return prompt


parser = argparse.ArgumentParser(description="Make questions from a csv file")

parser.add_argument(
    "--csv_file",
    required=True,
    metavar="csv_file",
    type=str,
    help="csv file to read from",
)
# read two columns from the csv file, one func name, one description
parser.add_argument(
    "--func_name",
    default=0,
    required=False,
    metavar="func_name",
    type=int,
    help="column number of function name",
)
parser.add_argument(
    "--description",
    default=1,
    required=False,
    metavar="description",
    type=int,
    help="column number of description",
)
parser.add_argument(
    "--output_csv",
    required=True,
    metavar="output_csv",
    type=str,
    help="output csv file",
)

args = parser.parse_args()

model = "meta-llama/Llama-2-7b-chat-hf"

# config = LlamaConfig(rope_scaling={"type": "dynamic", "factor":2.0})
# config = LlamaConfig(max_length=500)
#
# model_ = LlamaForCausalLM(config)
# model_.from_pretrained(model)

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device="cuda",
    tokenizer=tokenizer,
)


def csv_collator(batch):
    keys = batch[0].keys()
    batch = {key: [d[key] for d in batch] for key in keys}
    return batch


# make csv file into a torch dataset
class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i = self.data.iloc[idx]
        i["prompt"] = prompt(i["doc"])
        return i


# make csv file into a torch dataloader
def csv_dataloader(csv_file, batch_size=8):
    dataset = CSVDataset(csv_file)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=csv_collator
    )
    return dataloader


# make csv file into a torch dataloader
dataloader = csv_dataloader(args.csv_file, batch_size=8)

with open(args.output_csv, "w") as f:
    output = csv.writer(f)
    output.writerow(["function_name", "question", "doc", "function"])

    for batch in dataloader:
        sequences = pipeline(
            batch["prompt"],
            do_sample=True,
            top_k=2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=450,
        )
        first = batch["function_name"]
        docs = batch["doc"]
        functions = batch["function"]

        for i, sequence in enumerate(sequences):
            gen_text = sequence[0]["generated_text"]

            gen_text = [
                line
                for line in gen_text.split("\n")
                if len(line) > 10 and line.strip()[0].isdigit()
            ]
            for line in gen_text:
                try:
                    output.writerow([first[i], line[2:], docs[i], functions[i]])
                except:
                    continue
