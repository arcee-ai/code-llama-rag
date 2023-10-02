import os
import csv
import argparse
import transformers
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
)
from transformers.pipelines.pt_utils import KeyDataset
import torch
import pandas as pd
from torch.utils.data import Subset
import pickle

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
parser.add_argument(
    "--output_csv",
    required=True,
    metavar="output_csv",
    type=str,
    help="output csv file",
)
# model name as an argument
parser.add_argument(
    "--model",
    required=False,
    metavar="model",
    default="meta-llama/Llama-2-7b-chat-hf",
    type=str,
    help="model name",
)
parser.add_argument(
    "--use_bnb",
    action="store_true",
    help="boolean flag to use bnb",
)
parser.add_argument(
    "--batch_size",
    required=False,
    default=8,
    metavar="batch_size",
    type=int,
    help="batch size",
)
# max length
parser.add_argument(
    "--max_length",
    required=False,
    default=500,
    metavar="max_length",
    type=int,
    help="max length",
)
parser.add_argument(
    "--split_into",
    required=False,
    default=1,
    metavar="split_into",
    type=int,
    help="split into",
)

args = parser.parse_args()

# config = LlamaConfig(rope_scaling={"type": "dynamic", "factor":2.0})
# config = LlamaConfig(max_length=500)
#
# model_ = LlamaForCausalLM(config)
# model_.from_pretrained(model)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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
        i["prompt"] = prompt(i["function"])
        return i


# make csv file into a torch dataloader
def csv_dataloader(csv_file, batch_size=8, state=None):
    dataset = CSVDataset(csv_file)

    parts = state["parts"] if state is not None else 1

    if parts == 1:
        print("The whole training data will be processed at once")
        state["index"] = 1
        dataset = dataset
    elif parts > 1:
        if state["index"] == 0:
            print(
                "The training data will be split into ",
                parts,
                " parts and each part will be around the size of ",
                int(len(dataset) / parts),
            )
        else:
            print(
                "Generation is picking up from where it left off. Starting at the ",
                state["index"],
                "th part of the data",
            )
        dataset = Subset(
            dataset,
            range(
                int(state["index"] * len(dataset) / parts),
                int((state["index"] + 1) * len(dataset) / parts),
            ),
        )
        state["index"] += 1

    return (
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=csv_collator
        ),
        state,
    )


tokenizer = AutoTokenizer.from_pretrained(
    args.model, quantization_config=bnb_config if args.use_bnb else None
)

pipeline = transformers.pipeline(
    "text-generation",
    model=args.model,
    torch_dtype=torch.float16,
    device="cuda",
    tokenizer=tokenizer,
)


if __name__ == "__main__":
    # =============== State creation ===============
    #
    # The state is a dictionary that keeps track of the progress of the generation.
    # It is saved as a pickle file and loaded at the start of the generation.
    #

    pickle_file = args.output_csv.split(".")[0] + ".pkl"
    try:
        state = pickle.load(open(pickle_file, "rb"))
    except:
        state = {"index": 0, "parts": args.split_into}
        pickle.dump(state, open(pickle_file, "wb"))

    r = state["parts"] - state["index"]

    # =============== Generation ===============
    #
    # depending on the state, the generation will either start from the beginning of the data or from where it left off
    # r is the number of parts left to generate

    for i in range(r):
        dataloader, state = csv_dataloader(
            args.csv_file, batch_size=args.batch_size, state=state
        )

        if state["parts"] == 1:
            filename = args.output_csv
        else:
            filename = (
                args.output_csv.split(".")[0] + "_" + str(state["index"] - 1) + ".csv"
            )

        with open(filename, "w") as f:
            output = csv.writer(f)
            output.writerow(["function_name", "question", "function"])

            for batch in dataloader:
                sequences = pipeline(
                    batch["prompt"],
                    do_sample=True,
                    top_k=2,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=args.max_length,
                )
                first = batch["function_name"]
                functions = batch["function"]

                for i, sequence in enumerate(sequences):
                    gen_text = sequence[0]["generated_text"]

                    gen_text = [
                        line
                        for line in gen_text.split("\n")
                        if len(line.strip()) > 10 and line.strip()[0].isdigit()
                    ]
                    for line in gen_text:
                        try:
                            output.writerow([first[i], line[2:], functions[i]])
                        except:
                            continue

            pickle.dump(state, open(pickle_file, "wb"))

    print("Deleting state")
    os.remove(pickle_file)
