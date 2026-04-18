import torch
from flashchrombert.model import BertConfig, BertForMaskedLM
from flashchrombert.data import KmerCStateTokenizer

tokenizer = KmerCStateTokenizer(k=4, num_states=15)
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=1536,
    max_position_embeddings=512
)
model = BertForMaskedLM(config)
ckpt = torch.load("checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")})
model.eval()

# Let's find a sequence that has an interesting transition (not just OOOO OOOO...)
interesting_seq = None
with open("data/ch_temp/promoter_pretrain_data/val_split.txt", "r") as f:
    for line in f:
        if "A" in line and "B" in line and "E" in line:
            interesting_seq = line.strip()
            break

tokens = interesting_seq.split()

# Find an interesting place to mask. E.g., where there's a transition.
mask_idx = 10
for i in range(10, len(tokens)-3):
    if tokens[i] != tokens[i+1]:
        mask_idx = i
        break

original_tokens = [tokens[mask_idx], tokens[mask_idx+1], tokens[mask_idx+2]]
tokens[mask_idx] = "[MASK]"
tokens[mask_idx+1] = "[MASK]"
tokens[mask_idx+2] = "[MASK]"

masked_seq = " ".join(tokens)
print(f"Masked 3 tokens at idx {mask_idx}-{mask_idx+2}: {original_tokens}")

input_ids = torch.tensor([tokenizer.encode(masked_seq, add_special=True)])

with torch.no_grad():
    logits = model(input_ids).logits

for offset in range(3):
    pos = mask_idx + 1 + offset
    masked_position_logits = logits[0, pos, :]
    top_3_idx = torch.topk(masked_position_logits, 3).indices
    print(f"\n--- Position {mask_idx + offset} (Original: {original_tokens[offset]}) ---")
    for idx in top_3_idx:
        token_id = idx.item()
        prob = torch.softmax(masked_position_logits, dim=0)[token_id].item()
        print(f"Pred: {tokenizer.id_to_token[token_id]:<8} Prob: {prob:.4f}")

