import argparse
from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

# Add command-line arguments
parser = argparse.ArgumentParser(description="Process protein sequences using ESM2 model.")
parser.add_argument("--interval", type=str, required=True, help="Interval range for residues in the format [A-B]")
parser.add_argument("--model_name", type=str, required=True, choices=["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D", 
                                                                      "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D", 
                                                                      "facebook/esm2_t36_3B_UR50D", "facebook/esm2_t48_15B_UR50D"],
                    help="Name of the model to use")
parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file")
parser.add_argument("--mutation_effects_output", type=str, default="mutation_effects.csv", help="Output file name for mutation effects (default: mutation_effects.csv)")
parser.add_argument("--pair_effects_output", type=str, default="mutation_pair_effects.csv", help="Output file name for pair effects (default: mutation_pair_effects.csv)")

args = parser.parse_args()

# Parse the interval argument
interval = args.interval.strip("[]").split("-")
start_residue = int(interval[0])
end_residue = int(interval[1])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer based on user input
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)
model.to(device)

def process_sequence(protein_sequence, sequence_id, start_residue, end_residue):
    """
    Process a protein sequence using the ESM2 model.

    Args:
        protein_sequence (str): The protein sequence to process.
        sequence_id (str): The ID of the sequence.
        start_residue (int): The starting residue of the segment.
        end_residue (int): The ending residue of the segment.

    Returns:
        tuple: A tuple containing the sequence length, heatmap, amino acids, and the original protein sequence.
    """
    # Handle sequence length constraint based on the interval range
    sequence_length = len(protein_sequence)
    if end_residue > sequence_length:
        print(f"Sequence {sequence_id} length {sequence_length} is shorter than the specified end residue {end_residue}. Adjusting to sequence length.")
        end_residue = sequence_length

    protein_sequence = protein_sequence[start_residue-1:end_residue]
    
    # Tokenize the input sequence
    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt").to(device)
    sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens

    # List of amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # Initialize heatmap
    heatmap = np.zeros((len(amino_acids), sequence_length))

    # Calculate LLRs for each position and amino acid
    for position in range(1, sequence_length + 1):
        # Mask the target position
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, position] = tokenizer.mask_token_id

        # Get logits for the masked token
        with torch.no_grad():
            logits = model(masked_input_ids).logits

        # Calculate log probabilities
        probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
        log_probabilities = torch.log(probabilities)

        # Get the log probability of the wild-type amino acid
        wt_aa = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_aa].item()

        # Calculate LLR for each variant
        for i, aa in enumerate(amino_acids):
            log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(aa)].item()
            heatmap[i, position - 1] = log_prob_mt - log_prob_wt

    return sequence_length, heatmap, amino_acids, protein_sequence

def calculate_pair_effects(sequence_length, heatmap, amino_acids, protein_sequence):
    """
    Calculate the pair effects for a protein sequence.

    Args:
        sequence_length (int): The length of the protein sequence.
        heatmap (numpy.ndarray): The heatmap containing the LLR values.
        amino_acids (list): The list of amino acids.
        protein_sequence (str): The original protein sequence.

    Returns:
        tuple: A tuple containing the pair positions, pair amino acids, pair sums, and sorted indices.
    """
    # Calculate the number of pairs
    num_pairs = (sequence_length * (sequence_length - 1) * len(amino_acids) * len(amino_acids)) // 2
    pair_positions = np.zeros((num_pairs, 2), dtype=int)
    pair_aas = np.zeros((num_pairs, 2), dtype=int)
    pair_sums = np.zeros(num_pairs)

    # Compute the sums of pairs of LLR values for distinct pairs of amino acids
    index = 0
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            for aa1 in range(len(amino_acids)):
                for aa2 in range(len(amino_acids)):
                    pair_positions[index] = [i, j]
                    pair_aas[index] = [aa1, aa2]
                    pair_sums[index] = heatmap[aa1, i] + heatmap[aa2, j]
                    index += 1

    # Return the sorted indices, pairs information, and LLR values
    return pair_positions, pair_aas, pair_sums, np.argsort(pair_sums)

# Load sequences from the specified FASTA file
fasta_file = args.fasta_file
all_mutation_effects = []
all_pairs_info = []

for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq).upper()
    seq_length, heatmap, aas, seq = process_sequence(sequence, record.id, start_residue, end_residue)
    pair_positions, pair_aas, pair_sums, sorted_indices = calculate_pair_effects(seq_length, heatmap, aas, seq)
    
    # Store mutation effects
    for pos in range(seq_length):
        for aa_idx, aa in enumerate(aas):
            all_mutation_effects.append({
                "Sequence_ID": record.id,
                "Position": pos + 1 + start_residue - 1,
                "Amino_Acid": aa,
                "LLR": heatmap[aa_idx, pos]
            })
    
    # Store top 10 most beneficial and deleterious pairs
    top_beneficial_indices = sorted_indices[-10:]
    top_deleterious_indices = sorted_indices[:10]

    # Extract pair information for reporting
    for idx_array, kind in [(top_beneficial_indices, "Beneficial"), (top_deleterious_indices, "Deleterious")]:
        for index in idx_array:
            pos1, pos2 = pair_positions[index]
            aa1, aa2 = pair_aas[index]
            llr1, llr2 = heatmap[aa1, pos1], heatmap[aa2, pos2]
            all_pairs_info.append({
                "Sequence_ID": record.id,
                "Pair": f"{seq[pos1]}{pos1+1+start_residue-1}{aas[aa1]} - {seq[pos2]}{pos2+1+start_residue-1}{aas[aa2]}",
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
# python protein_mutation_pairs_fasta.py --interval [1-100] --model_name facebook/esm2_t6_8M_UR50D --fasta_file O95905-ecd_human.fa --mutation_effects_output protein_mutation_effects.csv --pair_effects_output protein_pair_effects.csv
