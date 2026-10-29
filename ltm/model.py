"""LTM: in-context tabular transformer (batched).

One table is an episode: context rows (with labels) followed by query rows.
Every cell becomes a token. Blocks alternate attention across columns (within a
row) and across rows (within a column). A target token is appended as an extra
column: it carries the label for context rows and a learned mask for query rows.
Query rows attend only to context rows, so no query label can leak.

Inputs may be one table (X: (N, F)) or a batch of tables with the same shape
(X: (B, N, F)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_COLS = 128
MAX_CAT = 12       # categorical codes 0..MAX_CAT-1; index MAX_CAT = missing
MAX_CLASSES = 10


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        mk = lambda: nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.col, self.row = mk(), mk()

    def forward(self, h, mask):  # h: (B, N, C, d); mask: (N, N) bool, True = not allowed
        B, N, C, d = h.shape
        h = self.col(h.reshape(B * N, C, d)).reshape(B, N, C, d)   # across columns, per row
        h = h.permute(0, 2, 1, 3).reshape(B * C, N, d)
        h = self.row(h, src_mask=mask)                              # across rows, per column
        return h.reshape(B, C, N, d).permute(0, 2, 1, 3)


class LTM(nn.Module):
    def __init__(self, d=64, heads=4, layers=4):
        super().__init__()
        self.num = nn.Linear(2, d)                        # (value, missing flag)
        self.cat = nn.Embedding(MAX_CAT + 1, d)
        self.col_pos = nn.Embedding(MAX_COLS, d)
        self.is_cat = nn.Embedding(2, d)
        self.cls_emb = nn.Embedding(MAX_CLASSES, d)
        self.reg_emb = nn.Linear(1, d)
        self.task_emb = nn.Embedding(2, d)
        self.query_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.cls_head = nn.Linear(d, MAX_CLASSES)
        self.reg_head = nn.Linear(d, 1)

    def forward(self, X, cat_mask, y_ctx, task, n_classes):
        """X: (N, F) or (B, N, F) float, NaN = missing, context rows first.
        cat_mask: (F,) or (B, F) bool. y_ctx: labels of the first n_ctx rows.
        n_classes: int, or a (B,) tensor for batches.
        Returns logits (.., n_q, n_classes) or standardized regression predictions (.., n_q)."""
        squeeze = X.dim() == 2
        if squeeze:
            X, cat_mask, y_ctx = X[None], cat_mask[None], y_ctx[None]
        B, N, nf = X.shape
        n_ctx = y_ctx.shape[1]
        dev = X.device
        miss = torch.isnan(X)

        # standardize numeric columns with context statistics only
        ok = ~miss[:, :n_ctx]
        Xc = torch.nan_to_num(X[:, :n_ctx])
        cnt = ok.sum(1).clamp(min=1)                                  # (B, F)
        mean = (Xc * ok).sum(1) / cnt
        var = (((Xc - mean[:, None]) ** 2) * ok).sum(1) / cnt
        std = var.sqrt().clamp(min=1e-6)
        Xn = torch.where(miss, torch.zeros_like(X), (X - mean[:, None]) / std[:, None]).clamp(-5, 5)
        num_h = self.num(torch.stack([Xn, miss.float()], dim=-1))     # (B, N, F, d)

        codes = X.nan_to_num(0.0).long().clamp(0, MAX_CAT - 1)
        codes = torch.where(miss, torch.full_like(codes, MAX_CAT), codes)
        h = torch.where(cat_mask[:, None, :, None], self.cat(codes), num_h)
        h = h + self.col_pos(torch.arange(nf, device=dev))[None, None] + self.is_cat(cat_mask.long())[:, None]

        task_id = torch.tensor(0 if task == "classification" else 1, device=dev)
        if task == "classification":
            t_ctx = self.cls_emb(y_ctx)
        else:
            ym = y_ctx.mean(1, keepdim=True)
            ys = y_ctx.std(1, unbiased=False, keepdim=True).clamp(min=1e-6)
            t_ctx = self.reg_emb(((y_ctx - ym) / ys).unsqueeze(-1))
        t_q = self.query_tok.expand(B, N - n_ctx, -1)
        t = torch.cat([t_ctx, t_q], 1) + self.task_emb(task_id)
        h = torch.cat([h, t.unsqueeze(2)], dim=2)                     # (B, N, F+1, d)

        mask = torch.zeros(N, N, dtype=torch.bool, device=dev)
        mask[:, n_ctx:] = True                                        # keys are context rows only
        for b in self.blocks:
            h = b(h, mask)
        out = self.norm(h[:, n_ctx:, -1])                             # target token of query rows

        if task == "classification":
            res = self.cls_head(out)
            if isinstance(n_classes, int):
                res = res[..., :n_classes]
            else:
                k = torch.arange(MAX_CLASSES, device=dev)
                res = res.masked_fill(k[None, None] >= n_classes[:, None, None], float("-inf"))
        else:
            res = self.reg_head(out).squeeze(-1)
        return res[0] if squeeze else res


def episode_loss(model, X, cat_mask, y, n_ctx, task, n_classes):
    """Works for one table or a batch of same-shape tables."""
    out = model(X, cat_mask, y[..., :n_ctx], task, n_classes)
    yq = y[..., n_ctx:]
    if task == "classification":
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), yq.reshape(-1)), out
    yc = y[..., :n_ctx]
    ym = yc.mean(-1, keepdim=True)
    ys = yc.std(-1, unbiased=False, keepdim=True).clamp(min=1e-6)
    return F.mse_loss(out, (yq - ym) / ys), out
