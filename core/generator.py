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
