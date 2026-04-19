import torch

from flashchrombert.model import BertConfig, BertForMaskedLM, BertModel
from flashchrombert.model.attention import MultiHeadAttention


def tiny_config():
    return BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )


def test_attention_shape():
    cfg = tiny_config()
    attn = MultiHeadAttention(cfg)
    x = torch.randn(2, 16, cfg.hidden_size)
    out, weights = attn(x)
    assert out.shape == x.shape
    assert weights is None


def test_attention_return_attn():
    cfg = tiny_config()
    attn = MultiHeadAttention(cfg).eval()
    x = torch.randn(2, 16, cfg.hidden_size)
    out, weights = attn(x, return_attn=True)
    assert out.shape == x.shape
    assert weights.shape == (2, cfg.num_attention_heads, 16, 16)


def test_sdpa_matches_eager():
    """Eager path and SDPA path should be numerically close in eval mode (no dropout)."""
    cfg = tiny_config()
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    attn = MultiHeadAttention(cfg).eval()
    x = torch.randn(2, 16, cfg.hidden_size)
    with torch.no_grad():
        out_eager, _ = attn(x, return_attn=True)
        out_sdpa, _ = attn(x, return_attn=False)
    assert torch.allclose(out_eager, out_sdpa, atol=1e-5)


def test_sdpa_matches_eager_with_mask():
    """With a padding mask, the two paths must agree (bool mask vs additive -inf)."""
    cfg = tiny_config()
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    attn = MultiHeadAttention(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, cfg.hidden_size)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[0, 12:] = 0
    mask[1, 8:] = 0
    with torch.no_grad():
        out_eager, _ = attn(x, attention_mask=mask, return_attn=True)
        out_sdpa, _ = attn(x, attention_mask=mask, return_attn=False)
    assert torch.allclose(out_eager, out_sdpa, atol=1e-5)


def test_head_init_std_matches_config():
    """MLM and classification heads must use config.initializer_range, not PyTorch defaults."""
    from flashchrombert.model import BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=64, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=256,
        max_position_embeddings=32, initializer_range=0.02,
    )
    mlm = BertForMaskedLM(cfg)
    clf = BertForSequenceClassification(cfg, num_labels=2)

    # Linear weights should have std ≈ 0.02 (tolerate finite-sample noise).
    for name, w in [
        ("mlm.transform[0]", mlm.mlm_head.transform[0].weight),
        ("clf.pooler_dense", clf.head.pooler_dense.weight),
        ("clf.classifier", clf.head.classifier.weight),
    ]:
        assert abs(w.std().item() - cfg.initializer_range) < 0.005, (
            f"{name} std={w.std().item():.4f}, expected ≈ {cfg.initializer_range}"
        )

    # Biases must be zero.
    assert torch.all(mlm.mlm_head.transform[0].bias == 0)
    assert torch.all(mlm.mlm_head.decoder.bias == 0)
    assert torch.all(clf.head.pooler_dense.bias == 0)
    assert torch.all(clf.head.classifier.bias == 0)

    # decoder.weight must remain tied to token embedding after init.
    assert mlm.mlm_head.decoder.weight.data_ptr() == mlm.bert.embeddings.token.weight.data_ptr()


def test_bert_forward():
    cfg = tiny_config()
    model = BertModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids)
    assert out.last_hidden_state.shape == (2, 16, cfg.hidden_size)


def test_mlm_loss_and_gradient():
    cfg = tiny_config()
    model = BertForMaskedLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.loss is not None and out.loss.requires_grad
    out.loss.backward()
    # sanity: at least one parameter should receive non-zero gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_overfit_single_batch():
    """Model must be able to drive loss near zero on a single batch."""
    torch.manual_seed(0)
    cfg = tiny_config()
    model = BertForMaskedLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    input_ids = torch.randint(5, cfg.vocab_size, (2, 8))  # avoid specials
    labels = input_ids.clone()

    final_loss = None
    for _ in range(100):
        opt.zero_grad()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        opt.step()
        final_loss = out.loss.item()

    assert final_loss < 0.5, f"Model failed to overfit, final loss={final_loss}"
