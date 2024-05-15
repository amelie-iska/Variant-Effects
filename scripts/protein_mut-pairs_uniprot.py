from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import requests

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)
model.to(device)

# Function to fetch protein sequence from UniProt
def fetch_protein_sequence(protein_name):
    base_url = "https://rest.uniprot.org/uniprotkb/search?query=protein_name:{}&format=fasta"
    response = requests.get(base_url.format(protein_name))
    
    if response.status_code == 200:
        fasta_data = response.text
        lines = fasta_data.split("\n")
        protein_sequence = "".join(lines[1:])
        return protein_sequence
    else:
        print(f"Error fetching protein sequence for {protein_name}")
        return None

# Input protein name
protein_name = input("Enter protein name (e.g., BRCA1): ")

# Fetch protein sequence from UniProt
protein_sequence = fetch_protein_sequence(protein_name)

if protein_sequence:
    # Truncate input sequence if longer than 1022
    if len(protein_sequence) > 1022:
        protein_sequence = protein_sequence[:1022]
    
    # Tokenize the input sequence
    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt").to(device)
    sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens
    
    # List of amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Initialize heatmap
    heatmap = np.zeros((20, sequence_length))
    
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
        
        # Get the log probability of the wild-type residue
        wt_residue = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()
        
        # Calculate LLR for each variant
        for i, amino_acid in enumerate(amino_acids):
            log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(amino_acid)].item()
            heatmap[i, position - 1] = log_prob_mt - log_prob_wt
    
    
    # Visualize the heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(heatmap, cmap="coolwarm", aspect="auto")
    plt.xticks(range(sequence_length), list(protein_sequence))
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid")
    plt.title(f"Predicted Effects of Mutations on {protein_name} Protein Sequence (LLR)")
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    plt.savefig(f"{protein_name}_mutation_heatmap.png")
    plt.close()
    
# Compute the number of pairs
num_pairs = (sequence_length * (sequence_length - 1) * 20 * 20) // 2
print(f"Number of distinct LLR value pairs: {num_pairs}")

# Initialize arrays to store the pairs and their sums
pair_positions = np.zeros((num_pairs, 2), dtype=int)
pair_amino_acids = np.zeros((num_pairs, 2), dtype=int)
pair_sums = np.zeros(num_pairs)

# Compute the sums of pairs of LLR values for distinct pairs of residues
index = 0
for i in range(sequence_length):
    for j in range(i + 1, sequence_length):
        for aa1 in range(20):
            for aa2 in range(20):
                pair_positions[index] = [i, j]
                pair_amino_acids[index] = [aa1, aa2]
                pair_sums[index] = heatmap[aa1, i] + heatmap[aa2, j]
                index += 1

# Sort the pairs based on their sums
sorted_indices = np.argsort(pair_sums)

# Get the top 10 most beneficial and deleterious pairs
top_beneficial_indices = sorted_indices[-10:]
top_deleterious_indices = sorted_indices[:10]

# Print the results
print("Top 10 most beneficial pairs:")
for index in top_beneficial_indices:
    pos1, pos2 = pair_positions[index]
    aa1, aa2 = pair_amino_acids[index]
    llr1, llr2 = heatmap[aa1, pos1], heatmap[aa2, pos2]
    print(f"({protein_sequence[pos1]}{pos1+1}{amino_acids[aa1]}, {llr1:.2f}), ({protein_sequence[pos2]}{pos2+1}{amino_acids[aa2]}, {llr2:.2f}), {llr1+llr2:.2f}")

print("\nTop 10 most deleterious pairs:")
for index in top_deleterious_indices:
    pos1, pos2 = pair_positions[index]
    aa1, aa2 = pair_amino_acids[index]
    llr1, llr2 = heatmap[aa1, pos1], heatmap[aa2, pos2]
    print(f"({protein_sequence[pos1]}{pos1+1}{amino_acids[aa1]}, {llr1:.2f}), ({protein_sequence[pos2]}{pos2+1}{amino_acids[aa2]}, {llr2:.2f}), {llr1+llr2:.2f}")