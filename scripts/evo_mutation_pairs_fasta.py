from stripedhyena.tokenizer import CharLevelTokenizer
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

# Add command-line arguments
parser = argparse.ArgumentParser(description="Process DNA sequences using Evo model.")
parser.add_argument("--interval", type=str, required=True, help="Interval range for residues in the format [A-B]")
parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file")
parser.add_argument("--mutation_effects_output", type=str, default="mutation_effects.csv", help="Output file name for mutation effects (default: mutation_effects.csv)")
parser.add_argument("--pair_effects_output", type=str, default="mutation_pair_effects.csv", help="Output file name for pair effects (default: mutation_pair_effects.csv)")
parser.add_argument("--model", type=str, required=True, choices=["togethercomputer/evo-1-131k-base", "togethercomputer/evo-1-8k-base"], help="Evo model checkpoint to use")

args = parser.parse_args()

# Parse the interval argument
interval = args.interval.strip("[]").split("-")
start_residue = int(interval[0])
end_residue = int(interval[1])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = args.model
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True, revision="1.1_fix")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
model.to(device)

def process_sequence(dna_sequence, sequence_id, start_residue, end_residue):
    """
    Process a DNA sequence using the Evo model.

    Args:
        dna_sequence (str): The DNA sequence to process.
        sequence_id (str): The ID of the sequence.
        start_residue (int): The starting residue of the segment.
        end_residue (int): The ending residue of the segment.

    Returns:
        tuple: A tuple containing the sequence length, heatmap, DNA bases, and the original DNA sequence.
    """
    # Handle sequence length constraint based on the interval range
    sequence_length = len(dna_sequence)
    if end_residue > sequence_length:
        print(f"Sequence {sequence_id} length {sequence_length} is shorter than the specified end residue {end_residue}. Adjusting to sequence length.")
        end_residue = sequence_length

    dna_sequence = dna_sequence[start_residue-1:end_residue]
    
    # Tokenize the input sequence
    input_ids = tokenizer.encode(dna_sequence, return_tensors="pt").to(device)
    sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens

    # List of DNA bases
    dna_bases = list("ATCG")

    # Initialize heatmap
    heatmap = np.zeros((4, sequence_length))

    # Calculate LLRs for each position and DNA base
    for position in range(1, sequence_length + 1):
        # Get logits for the target position
        with torch.no_grad():
            logits = model(input_ids[:, :position + 1]).logits
        
        # Calculate log probabilities
        probabilities = torch.nn.functional.softmax(logits[0, -1], dim=0)
        log_probabilities = torch.log(probabilities)
        
        # Get the log probability of the wild-type base
        wt_base = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_base].item()
        
        # Calculate LLR for each variant
        for i, base in enumerate(dna_bases):
            log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(base)].item()
            heatmap[i, position - 1] = log_prob_mt - log_prob_wt

    return sequence_length, heatmap, dna_bases, dna_sequence

def calculate_pair_effects(sequence_length, heatmap, dna_bases, dna_sequence):
    """
    Calculate the pair effects for a DNA sequence.

    Args:
        sequence_length (int): The length of the DNA sequence.
        heatmap (numpy.ndarray): The heatmap containing the LLR values.
        dna_bases (list): The list of DNA bases.
        dna_sequence (str): The original DNA sequence.

    Returns:
        tuple: A tuple containing the pair positions, pair bases, pair sums, and sorted indices.
    """
    # Calculate the number of pairs
    num_pairs = (sequence_length * (sequence_length - 1) * 4 * 4) // 2
    pair_positions = np.zeros((num_pairs, 2), dtype=int)
    pair_bases = np.zeros((num_pairs, 2), dtype=int)
    pair_sums = np.zeros(num_pairs)

    # Compute the sums of pairs of LLR values for distinct pairs of bases
    index = 0
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            for base1 in range(4):
                for base2 in range(4):
                    pair_positions[index] = [i, j]
                    pair_bases[index] = [base1, base2]
                    pair_sums[index] = heatmap[base1, i] + heatmap[base2, j]
                    index += 1

    # Return the sorted indices, pairs information, and LLR values
    return pair_positions, pair_bases, pair_sums, np.argsort(pair_sums)

# Load sequences from the specified FASTA file
fasta_file = args.fasta_file
all_mutation_effects = []
all_pairs_info = []

for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq).upper()
    seq_length, heatmap, bases, seq = process_sequence(sequence, record.id, start_residue, end_residue)
    pair_positions, pair_bases, pair_sums, sorted_indices = calculate_pair_effects(seq_length, heatmap, bases, seq)
    
    # Store mutation effects
    for pos in range(seq_length):
        for base_idx, base in enumerate(bases):
            all_mutation_effects.append({
                "Sequence_ID": record.id,
                "Position": pos + 1 + start_residue - 1,
                "Base": base,
                "LLR": heatmap[base_idx, pos]
            })
    
    # Store top 10 most beneficial and deleterious pairs
    top_beneficial_indices = sorted_indices[-10:]
    top_deleterious_indices = sorted_indices[:10]

    # Extract pair information for reporting
    for idx_array, kind in [(top_beneficial_indices, "Beneficial"), (top_deleterious_indices, "Deleterious")]:
        for index in idx_array:
            pos1, pos2 = pair_positions[index]
            base1, base2 = pair_bases[index]
            llr1, llr2 = heatmap[base1, pos1], heatmap[base2, pos2]
            all_pairs_info.append({
                "Sequence_ID": record.id,
                "Pair": f"{seq[pos1]}{pos1+1+start_residue-1}{bases[base1]} - {seq[pos2]}{pos2+1+start_residue-1}{bases[base2]}",
                "Sum LLR": llr1 + llr2,
                "Type": kind
            })

# Convert results to DataFrame and save to CSV
df_effects = pd.DataFrame(all_mutation_effects)
df_effects.to_csv(args.mutation_effects_output, index=False)

df_pairs = pd.DataFrame(all_pairs_info)
df_pairs.to_csv(args.pair_effects_output, index=False)

print(f"Processing complete. Results saved to '{args.mutation_effects_output}' and '{args.pair_effects_output}'.")

# Example usage:
# python ./scripts/evo_mutation_pairs_fasta.py --interval [1-100] --fasta_file "./data/Homo_sapiens_BRAF_sequence.fa" --mutation_effects_output "./outputs/evo_mutation_effects.csv" --pair_effects_output "./outputs/evo_pair_effects.csv" --model togethercomputer/evo-1-131k-base
