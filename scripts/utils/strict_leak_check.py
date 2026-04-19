import torch
from flashchrombert.data import KmerCStateTokenizer, KmerMaskListMaskingStrategy

tokenizer = KmerCStateTokenizer(k=4, num_states=15)
# Set probability to 1.0 just to force a mask, then we'll see the span
strat = KmerMaskListMaskingStrategy(k=4, mlm_probability=1.0)

seq_text = " ".join(["ABCD", "BCDE", "CDEF", "DEFG", "EFGH", "FGHI", "GHIJ", "HIJK", "IJKL", "JKLM"])
input_ids = torch.tensor([tokenizer.encode(seq_text, add_special=True)])

# Overwrite strat.mask's internal random choice to ONLY pick index 5 ('EFGH') as the center
# This requires a tiny monkey patch for the test
original_mask = strat.mask
def mock_mask(input_ids, tokenizer):
    inputs = input_ids.clone()
    labels = input_ids.clone()
    
    # Force only index 5 to be the center
    masked_indices = torch.zeros(labels.shape, dtype=torch.bool)
    masked_indices[0, 5] = True 
    
    # Paste the expansion logic from dataset.py
    mask_list = [-1, 1, 2] # MASK_LIST[4]
    for i in range(masked_indices.shape[0]):
        centers = set(torch.nonzero(masked_indices[i], as_tuple=False).view(-1).tolist())
        new_centers = set(centers)
        for c in centers:
            for off in mask_list:
                j = c + off
                if 1 <= j <= 10: # seq len
                    new_centers.add(j)
        if new_centers:
            idx = torch.tensor(sorted(new_centers), dtype=torch.long)
            masked_indices[i, idx] = True

    labels[~masked_indices] = -100
    replace_mask = masked_indices # Force everything to [MASK] for clear printing
    inputs[replace_mask] = tokenizer.mask_token_id
    return inputs, labels

strat.mask = mock_mask
inputs, labels = strat.mask(input_ids, tokenizer)

print("Original sequence:", ['CLS'] + seq_text.split() + ['SEP'])
print("Inputs received by model:")
input_str = []
for idx in inputs[0]:
    if idx == tokenizer.mask_token_id:
        input_str.append("[MASK]")
    elif idx == tokenizer.cls_token_id:
        input_str.append("[CLS]")
    elif idx == tokenizer.sep_token_id:
        input_str.append("[SEP]")
    else:
        input_str.append(tokenizer.id_to_token[idx.item()])
print(input_str)
