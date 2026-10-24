"""LTM v0: in-context tabular transformer.

One table is an episode: context rows (with labels) followed by query rows.
Every cell becomes a token. Blocks alternate attention across columns (within a
row) and across rows (within a column). A target token is appended as an extra
column: it carries the label for context rows and a learned mask for query rows.
Query rows attend only to context rows, so no query label can leak.
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

    def forward(self, h, mask):  # h: (N, C, d); mask: (N, N) bool, True = not allowed
        N, C, d = h.shape
        h = self.col(h)                                   # across columns, per row
        h = self.row(h.transpose(0, 1), src_mask=mask)    # across rows, per column
        return h.transpose(0, 1)


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
        """X: (N, F) float, NaN = missing, context rows first. cat_mask: (F,) bool.
        y_ctx: labels of the first len(y_ctx) rows. Returns logits (n_q, n_classes)
        or standardized regression predictions (n_q,)."""
        N, nf = X.shape
        n_ctx = len(y_ctx)
        miss = torch.isnan(X)

        # standardize numeric columns with context statistics only
        ok = ~miss[:n_ctx]
        Xc = torch.nan_to_num(X[:n_ctx])
        cnt = ok.sum(0).clamp(min=1)
        mean = (Xc * ok).sum(0) / cnt
        std = (((Xc - mean) ** 2 * ok).sum(0) / cnt).sqrt().clamp(min=1e-6)
        Xn = torch.where(miss, torch.zeros_like(X), (X - mean) / std).clamp(-5, 5)
        num_h = self.num(torch.stack([Xn, miss.float()], dim=-1))

        codes = X.nan_to_num(0.0).long().clamp(0, MAX_CAT - 1)
        codes = torch.where(miss, torch.full_like(codes, MAX_CAT), codes)
        h = torch.where(cat_mask[None, :, None], self.cat(codes), num_h)
        h = h + self.col_pos(torch.arange(nf))[None] + self.is_cat(cat_mask.long())[None]

        task_id = torch.tensor(0 if task == "classification" else 1)
        if task == "classification":
            t_ctx = self.cls_emb(y_ctx)
        else:
            ym, ys = y_ctx.mean(), y_ctx.std(unbiased=False).clamp(min=1e-6)
            t_ctx = self.reg_emb(((y_ctx - ym) / ys)[:, None])
        t_q = self.query_tok.expand(N - n_ctx, -1)
        t = torch.cat([t_ctx, t_q], 0) + self.task_emb(task_id)
        h = torch.cat([h, t[:, None]], dim=1)             # (N, F+1, d)

        mask = torch.zeros(N, N, dtype=torch.bool)
        mask[:, n_ctx:] = True                            # keys are context rows only
        for b in self.blocks:
            h = b(h, mask)
        out = self.norm(h[n_ctx:, -1])                    # target token of query rows
        if task == "classification":
            return self.cls_head(out)[:, :n_classes]
        return self.reg_head(out).squeeze(-1)


def episode_loss(model, X, cat_mask, y, n_ctx, task, n_classes):
    out = model(X, cat_mask, y[:n_ctx], task, n_classes)
    yq = y[n_ctx:]
    if task == "classification":
        return F.cross_entropy(out, yq), out
    ym, ys = y[:n_ctx].mean(), y[:n_ctx].std(unbiased=False).clamp(min=1e-6)
    return F.mse_loss(out, (yq - ym) / ys), out
