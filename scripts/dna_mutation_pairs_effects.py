from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys; print(sys.executable)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

# Input DNA sequence
dna_sequence = "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"

# Tokenize the input sequence
input_ids = tokenizer.encode(dna_sequence, return_tensors="pt").to(device)
sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens

# List of DNA bases
dna_bases = list("ATCG")

# Initialize heatmap
heatmap = np.zeros((4, sequence_length))

# Calculate LLRs for each position and DNA base
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
    
    # Get the log probability of the wild-type base
    wt_base = input_ids[0, position].item()
    log_prob_wt = log_probabilities[wt_base].item()
    
    # Calculate LLR for each variant
    for i, base in enumerate(dna_bases):
        log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(base)].item()
        heatmap[i, position - 1] = log_prob_mt - log_prob_wt

# Visualize the heatmap
plt.figure(figsize=(15, 5))
plt.imshow(heatmap, cmap="coolwarm", aspect="auto")
plt.xticks(range(len(dna_sequence)), list(dna_sequence))
plt.yticks(range(4), dna_bases)
plt.xlabel("Position in DNA Sequence")
plt.ylabel("DNA Base")
plt.title("Predicted Effects of Mutations on DNA Sequence (LLR)")
plt.colorbar(label="Log Likelihood Ratio (LLR)")
plt.tight_layout()
plt.savefig("dna_mutation_heatmap.png")  # Save the plot as an image file
plt.close()  # Close the plot to free up memory

# Compute the number of pairs
num_pairs = (sequence_length * (sequence_length - 1) * 4 * 4) // 2
print(f"Number of distinct LLR value pairs: {num_pairs}")

# Initialize arrays to store the pairs and their sums
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

# Sort the pairs based on their sums
sorted_indices = np.argsort(pair_sums)

# Get the top 10 most beneficial and deleterious pairs
top_beneficial_indices = sorted_indices[-10:]
top_deleterious_indices = sorted_indices[:10]

# Print the results
print("Top 10 most beneficial pairs:")
for index in top_beneficial_indices:
    pos1, pos2 = pair_positions[index]
    base1, base2 = pair_bases[index]
    llr1, llr2 = heatmap[base1, pos1], heatmap[base2, pos2]
    print(f"({dna_sequence[pos1]}{pos1+1}{dna_bases[base1]}, {llr1:.2f}), ({dna_sequence[pos2]}{pos2+1}{dna_bases[base2]}, {llr2:.2f}), Sum: {llr1+llr2:.2f}")

print("\nTop 10 most deleterious pairs:")
for index in top_deleterious_indices:
    pos1, pos2 = pair_positions[index]
    base1, base2 = pair_bases[index]
    llr1, llr2 = heatmap[base1, pos1], heatmap[base2, pos2]
    print(f"({dna_sequence[pos1]}{pos1+1}{dna_bases[base1]}, {llr1:.2f}), ({dna_sequence[pos2]}{pos2+1}{dna_bases[base2]}, {llr2:.2f}), Sum: {llr1+llr2:.2f}")
