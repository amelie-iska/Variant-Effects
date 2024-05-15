import argparse
import pathlib
import string
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple

# Create and configure the argument parser for command line interface
def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from DNA language model."
    )
    # Define the arguments the script can accept
    parser.add_argument("--sequence", type=str, help="Base sequence to which mutations were applied")
    parser.add_argument("--dms-input", type=pathlib.Path, default="./data/dna_dms.csv", help="CSV file containing the deep mutational scan")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="Column in the deep mutational scan labeling the mutation")
    parser.add_argument("--dms-output", type=pathlib.Path, help="Output file containing the deep mutational scan along with predictions")
    parser.add_argument("--offset-idx", type=int, default=0, help="Offset of the mutation positions")
    parser.add_argument("--scoring-strategy", type=str, default="wt-marginals", choices=["wt-marginals", "masked-marginals", "pseudo-ppl"], help="Scoring strategy to use")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

# Function to label a row in the DataFrame based on the scoring of mutations
def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    # Extract wild type, index, and mutated type from the row
    wt, idx, mt = row[0], int(row[1:-1]) - 1 - offset_idx, row[-1]  # Subtract 1 for 0-based indexing and apply offset

    # Print detailed sequence information with indexing
    # print(f"Full sequence: {sequence}")
    print(f"Checking mutation {row}: wt={wt}, idx={idx}, mt={mt}")
    # for i, base in enumerate(sequence):
    #     print(f"Position {i}: {base}")

    print(f"Expected {wt} at position {idx} in the sequence, found {sequence[idx]}")
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # Encode the wild type and mutated type
    wt_encoded, mt_encoded = tokenizer.encode(wt, add_special_tokens=False)[0], tokenizer.encode(mt, add_special_tokens=False)[0]
    # Calculate the score as the difference in log probabilities
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

# Function to compute pseudo-perplexity for a row, used in language model evaluation
def compute_pppl(row, sequence, model, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # Modify the sequence with the mutation
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
    # Tokenize the modified sequence
    data = [("dna1", sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"]
    # Calculate log probabilities for each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.to(device)).logits, dim=-1)
        log_probs.append(token_probs[0, i, batch_tokens[0, i]].item())
    return sum(log_probs)

# Main function to orchestrate mutation scoring process
def main(args):
    # Load deep mutational scan data from a CSV file
    df = pd.read_csv(args.dms_input)

    # Determine to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    # Load the DNA language model and tokenizer
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)

    # Preprocess and encode the base sequence
    data = [("dna1", args.sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"].to(device)

    # Apply selected scoring strategy
    if args.scoring_strategy == "wt-marginals":
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens).logits, dim=-1)
        df[model_name] = df.apply(
            lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, tokenizer, args.offset_idx),
            axis=1
        )
    elif args.scoring_strategy == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1))):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = tokenizer.mask_token_id
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens_masked).logits, dim=-1)
            all_token_probs.append(token_probs[:, i])
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        df[model_name] = df.apply(
            lambda row: label_row(row[args.mutation_col], args.sequence, token_probs, tokenizer, args.offset_idx),
            axis=1
        )
    elif args.scoring_strategy == "pseudo-ppl":
        tqdm.pandas()
        df[model_name] = df.progress_apply(
            lambda row: compute_pppl(row[args.mutation_col], args.sequence, model, tokenizer, args.offset_idx),
            axis=1
        )

    # Set default output file if not provided
    if args.dms_output is None:
        args.dms_output = pathlib.Path(f"./outputs/dna-dms_{args.scoring_strategy}.csv")
    
    # Save the scored mutations to a CSV file
    df.to_csv(args.dms_output)

# Script entry point for command-line interaction
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)


# Example usage:
# python3 ./scripts/dna_mutation_scoring.py --sequence "AGTCAGTCAAATCGCAGCTGACTATCGATC" --dms-input ./data/dna_dms.csv --mutation-col mutant --scoring-strategy masked-marginals
