import time
import copy
import torch
import torch.nn as nn

from transformer import (
    MultiHeadedAttention,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    EncoderDecoder,
    PositionalEncoding,
    Generator,
    PositionwiseForward,
    Embeddings
)


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.ndim > 1:
            nn.init.xavier_normal_(p)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.mask_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # Create a mask to hide padding and future words
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1).type_as(
            tgt_mask.data
        ))


class TrainState:
    """
    Track number of steps, examples and tokens processed
    """
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimiser,
    scheduler,
    mode='train',
    accum_iter=1,
    train_state=TrainState()
):
    """Train a sinlge epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
    # loss_node = loss_node / accum_iter
    if mode == 'train' or mode == "train+log":
        loss_node.backward()
        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens
        if i % accum_iter == 0:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

    total_loss += loss
    total_tokens += batch.ntokens
    tokens += batch.ntokens
    if i % 40 == 1 and (mode == 'train' or mode == 'train+log'):
        lr = optimiser.param_group[0]['lr']
        elapsed = time.time() - start
        print(
            (
                "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
            )
            % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
        )
        start = time.time()
        tokens = 0

    return total_loss / total_tokens, train_state
