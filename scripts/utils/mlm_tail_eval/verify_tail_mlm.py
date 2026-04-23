import torch
import re
import random
from flashchrombert.model import BertConfig, BertForMaskedLM
from flashchrombert.data import KmerCStateTokenizer

def load_fcb_model(ckpt_path: str, vocab_size: int, device: str) -> BertForMaskedLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(cfg)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model

def find_interesting_sequences(data_path, chunk_size=10*1024*1024):
    with open(data_path, "r") as f:
        f.seek(0, 2)
        end_pos = f.tell()
        f.seek(max(0, end_pos - chunk_size))
        data = f.read()
    
    lines = [l for l in data.splitlines() if len(l.split()) > 400]
    print(f"Found {len(lines)} candidate long sequences in tail chunk.")
    
    selected = []
    # 策略 1: 寻找包含状态转换最剧烈的序列 (Entropy/Transitions)
    def count_transitions(seq):
        tokens = seq.split()
        count = 0
        for i in range(len(tokens)-1):
            if tokens[i][0] != tokens[i+1][0]: # 4-mer的首个状态发生变化
                count += 1
        return count

    lines.sort(key=count_transitions, reverse=True)
    selected.extend(lines[:3]) # 选前3个最复杂的
    
    # 策略 2: 使用正则寻找特定的模式，例如从 A 连续区转换到非 A 区
    # A 通常代表 Active TSS
    pattern = re.compile(r"(AAAA\s){10,}(?![AAAA])")
    for l in lines:
        if pattern.search(l) and l not in selected:
            selected.append(l)
            if len(selected) >= 5: break
            
    return selected[:5]

def verify_sequence(model, tokenizer, seq_str, device, title):
    tokens = seq_str.split()[:510]
    
    # 自动寻找遮盖点：在状态变化剧烈的地方遮盖
    mask_indices = []
    for i in range(10, len(tokens)-10):
        if tokens[i][0] != tokens[i+1][0]:
            # 连续遮盖 3 个
            mask_indices = [i-1, i, i+1]
            break
    
    if not mask_indices:
        mask_indices = [200, 201, 202]

    original_tokens = [tokens[idx] for idx in mask_indices]
    test_tokens = list(tokens)
    for idx in mask_indices:
        test_tokens[idx] = "[MASK]"
    
    print(f"\n>>> Case: {title}")
    print(f"Context (before/after): {' '.join(tokens[mask_indices[0]-3 : mask_indices[0]])} [MASK]x3 {' '.join(tokens[mask_indices[-1]+1 : mask_indices[-1]+4])}")
    
    input_ids = torch.tensor([tokenizer.encode(" ".join(test_tokens), add_special=True)]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
    
    for i, mask_idx in enumerate(mask_indices):
        pos = mask_idx + 1
        probs = torch.softmax(logits[0, pos, :], dim=0)
        top_v, top_i = torch.topk(probs, 3)
        
        print(f"  Pos {mask_idx} (True: {original_tokens[i]}):", end="")
        for v, idx in zip(top_v, top_i):
            pred = tokenizer.id_to_token[idx]
            print(f" {pred}({v:.3f})", end="")
        print()

def main():
    ckpt_path = "checkpoints/ch_ihec_whole_genome_tuned/step=42000-train_loss=0.147.ckpt"
    data_path = "/work/Users/lee/dev/IHEC/pretrain/IHEC_all_genome_cut_4mer_wo_R_all_concatenated_css.txt"
    vocab_path = "/work/Users/lee/ch_temp_IHEC/genome_IHEC18_pretrain_result/vocab.txt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KmerCStateTokenizer.from_vocab_file(vocab_path, k=4, num_states=18)
    model = load_fcb_model(ckpt_path, tokenizer.vocab_size, device)
    
    interesting_seqs = find_interesting_sequences(data_path)
    
    for i, seq in enumerate(interesting_seqs):
        verify_sequence(model, tokenizer, seq, device, f"Selected Sequence {i+1}")

if __name__ == "__main__":
    main()
