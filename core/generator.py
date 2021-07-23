import torch

from app_config import DEVICE


@torch.no_grad()
def greedy_decode(encoder_input_ids, model, tokenizer, decoder_input_ids, min_length=5, max_length=50):
    while decoder_input_ids[:, -1] != tokenizer.eos_token_id and decoder_input_ids.shape[1] < max_length:
        logits = model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits.squeeze(dim=0)
        next_token = torch.argmax(logits[-1]).view(1, 1)
        if next_token == tokenizer.eos_token_id and decoder_input_ids.shape[1] < min_length:
            top_k_tokens = torch.topk(logits[-1], 2).indices
            next_token = top_k_tokens[1].view(1, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    return decoder_input_ids[0]


@torch.no_grad()
def top_k_decode(k, encoder_input_ids, model, tokenizer, decoder_input_ids, min_length=5, max_length=50):
    while decoder_input_ids[:, -1] != tokenizer.eos_token_id and decoder_input_ids.shape[1] < max_length:
        logits = model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits.squeeze(dim=0)
        top_k_tokens = torch.topk(logits[-1], k).indices
        next_token_idx = torch.randint(top_k_tokens.shape[0], size=(1,))
        next_token = top_k_tokens[next_token_idx].view(1, 1)
        while next_token == tokenizer.eos_token_id and decoder_input_ids.shape[1] < min_length:
            next_token_idx = torch.randint(top_k_tokens.shape[0], size=(1,))
            next_token = top_k_tokens[next_token_idx].view(1, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    return decoder_input_ids[0]


@torch.no_grad()
def top_p_decode(p, encoder_input_ids, model, tokenizer, decoder_input_ids, min_length=5, max_length=50):
    while decoder_input_ids[:, -1] != tokenizer.eos_token_id and decoder_input_ids.shape[1] < max_length:
        logits = model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits.squeeze(dim=0)
        m = torch.nn.Softmax(dim=1)
        probs = m(logits)
        top_k_tokens = torch.topk(probs[-1], 0)
        sum_p = torch.sum(top_k_tokens.values)
        i = 1
        while sum_p < p:
            top_k_tokens = torch.topk(probs[-1], i)
            sum_p = torch.sum(top_k_tokens.values)
            i = i + 1

        next_token_idx = torch.randint(top_k_tokens.indices.shape[0], size=(1,))
        next_token = top_k_tokens.indices[next_token_idx].view(1, 1)
        while next_token == tokenizer.eos_token_id and decoder_input_ids.shape[1] < min_length:
            if i == 1 or i == 2:
                top_k_tokens = torch.topk(probs[-1], 3)
            next_token_idx = torch.randint(top_k_tokens.indices.shape[0], size=(1,))
            next_token = top_k_tokens.indices[next_token_idx].view(1, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    return decoder_input_ids[0]
