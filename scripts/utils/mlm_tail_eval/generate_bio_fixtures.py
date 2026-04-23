import torch
import re
import os
import random
import tqdm
from flashchrombert.data import KmerCStateTokenizer

def get_interesting_score(line):
    tokens = line.split()
    if len(tokens) < 510:
        return 0
    
    # 1. 计算状态转换频率 (越高说明越不是单调重复区)
    transitions = 0
    for i in range(len(tokens)-1):
        if tokens[i][0] != tokens[i+1][0]:
            transitions += 1
    
    # 2. 特征模式加分
    score = transitions * 1.0 
    
    # 状态 A-R 对应 1-18
    active_tokens = sum(1 for t in tokens if t[0] in "ABCDEGIJ")
    score += (active_tokens / len(tokens)) * 50
    
    biv_tokens = sum(1 for t in tokens if t[0] in "NO")
    score += (biv_tokens / len(tokens)) * 100
    
    return score

def main():
    data_path = "/work/Users/lee/dev/IHEC/pretrain/IHEC_all_genome_cut_4mer_wo_R_all_concatenated_css.txt"
    out_path = "fixtures/bio_eval_10k.pt"
    vocab_path = "/work/Users/lee/ch_temp_IHEC/genome_IHEC18_pretrain_result/vocab.txt"
    
    tokenizer = KmerCStateTokenizer.from_vocab_file(vocab_path, k=4, num_states=18)
    
    file_size = os.path.getsize(data_path)
    chunk_size = 500 * 1024 * 1024 # 500MB
    
    print(f"Reading last 500MB of {data_path}...")
    with open(data_path, "r") as f:
        f.seek(max(0, file_size - chunk_size))
        f.readline()
        lines = f.readlines()
    
    print(f"Total lines read: {len(lines)}")
    print("Scoring sequences...")
    scored_lines = []
    for l in tqdm.tqdm(lines):
        l = l.strip()
        if not l: continue
        score = get_interesting_score(l)
        if score > 0:
            scored_lines.append((score, l))
    
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    top_lines = [x[1] for x in scored_lines[:10000]]
    
    print(f"Selected {len(top_lines)} sequences. Generating masks...")
    
    all_input_ids = []
    all_labels = []
    
    for line in tqdm.tqdm(top_lines):
        # 裁剪到 510，加上 CLS 和 SEP 正好 512
        tokens = line.split()[:510]
        ids = tokenizer.encode(" ".join(tokens), add_special=True)
        # 确保正好 512
        if len(ids) > 512:
            ids = ids[:512]
        elif len(ids) < 512:
            ids = ids + [tokenizer.pad_token_id] * (512 - len(ids))
            
        input_ids = torch.tensor(ids)
        labels = input_ids.clone()
        
        mask_prob = 0.15
        probability_matrix = torch.full(labels.shape, mask_prob)
        
        special_tokens_mask = [
            val in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] 
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.mask_token_id
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    
    os.makedirs("fixtures", exist_ok=True)
    torch.save({
        "input_ids": torch.stack(all_input_ids),
        "labels": torch.stack(all_labels)
    }, out_path)
    print(f"Saved 10,000 cases to {out_path}")

if __name__ == "__main__":
    main()
