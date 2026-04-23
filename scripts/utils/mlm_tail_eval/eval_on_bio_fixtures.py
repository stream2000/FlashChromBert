import torch
import tqdm
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

def main():
    ckpt_path = "checkpoints/ch_ihec_whole_genome_tuned/step=42000-train_loss=0.147.ckpt"
    fixture_path = "fixtures/bio_eval_10k.pt"
    vocab_path = "/work/Users/lee/ch_temp_IHEC/genome_IHEC18_pretrain_result/vocab.txt"
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = KmerCStateTokenizer.from_vocab_file(vocab_path, k=4, num_states=18)
    model = load_fcb_model(ckpt_path, tokenizer.vocab_size, device)
    
    data = torch.load(fixture_path)
    input_ids = data["input_ids"]
    labels = data["labels"]
    
    num_samples = input_ids.size(0)
    total_correct = 0
    total_masked = 0
    
    print(f"Evaluating {num_samples} bio-meaningful samples...")
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, num_samples, batch_size)):
            batch_input = input_ids[i : i + batch_size].to(device)
            batch_labels = labels[i : i + batch_size].to(device)
            
            logits = model(batch_input).logits
            preds = torch.argmax(logits, dim=-1)
            
            mask = batch_labels != -100
            correct = (preds[mask] == batch_labels[mask]).sum().item()
            total_correct += correct
            total_masked += mask.sum().item()
            
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Total Masked Tokens: {total_masked}")
    print(f"  Correct Predictions: {total_correct}")
    print(f"  Overall Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
    main()
