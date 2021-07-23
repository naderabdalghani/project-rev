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


@torch.no_grad()
def best_search_decode(model, tokenizer, encoder_input_ids, prob, n, decoder_input_ids):
    if n < 0:  # number of beams
        return None, 0.0000001
    if decoder_input_ids[:, -1] == tokenizer.eos_token_id:
        return None, prob
    logits = model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits.squeeze(dim=0)
    m = torch.nn.Softmax(dim=1)
    probs = m(logits)
    top_k_tokens = torch.topk(probs[-1], 5).indices

    max_prob = 0
    max_road = []
    max_index = -1
    for i in top_k_tokens:
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([i]).view(1, 1).to(DEVICE)], dim=1)
        best_road, prob_children = best_search_decode(model, tokenizer, encoder_input_ids, probs[0][i], n - 1,
                                                      decoder_input_ids)
        decoder_input_ids = torch.tensor(decoder_input_ids[0][:-1]).view(1, decoder_input_ids.shape[1] - 1)

        if prob_children > max_prob:
            max_prob = prob_children
            max_road = best_road
            max_index = i

    if max_road is None:
        return torch.tensor([max_index]).to(DEVICE), prob * max_prob
    else:
        return torch.cat([torch.tensor([max_index]).to(DEVICE), max_road]), prob * max_prob


@torch.no_grad()
def generate_best_n(model, tokenizer, encoder_input_ids, decoder_input_ids, min_length, n_beams=3):
    logits = model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits.squeeze(dim=0)
    m = torch.nn.Softmax(dim=1)
    probs = m(logits)
    best = torch.topk(probs[-1], n_beams)
    for i in range(n_beams):
        if best.indices[i] == tokenizer.eos_token_id and decoder_input_ids.shape[1] < min_length:
            best.values[i] = 0
    return best


@torch.no_grad()
def best_n_of_nsquare(model, tokenizer, indices, values, encoder_input_ids, decoder_input_ids, min_length, n_beams=3):
    max_indexes = []
    max_probs = None

    for k in range(len(values)):

        if indices[k].shape == torch.Size([]):
            decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([indices[k]]).view(1, 1).to(DEVICE)], dim=1)
        else:
            for x in indices[k]:
                decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([x]).view(1, 1).to(DEVICE)], dim=1)

        prob_best_n, best_n = generate_best_n(model, tokenizer, encoder_input_ids, decoder_input_ids, min_length,
                                              n_beams)

        for j, prob in zip(best_n, prob_best_n):
            if indices[k].shape == torch.Size([]):
                if not max_indexes:
                    max_indexes.append(torch.cat([indices[k].view(1), j.view(1)]))
                    max_probs = torch.tensor([values[k] * prob])
                else:
                    max_indexes.append(torch.cat([indices[k].view(1), j.view(1)]))
                    max_probs = torch.cat([max_probs, torch.tensor([values[k] * prob])])
            else:

                if not max_indexes:
                    max_indexes.append(torch.cat([indices[k], j.view(1)]))
                    max_probs = torch.tensor([values[k] * prob])
                else:
                    max_indexes.append(torch.cat([indices[k], j.view(1)]))
                    max_probs = torch.cat([max_probs, torch.tensor([values[k] * prob])])

        if indices[k].shape == torch.Size([]):
            decoder_input_ids = decoder_input_ids[0][:-1].view(1, decoder_input_ids.shape[1] - 1)
        else:
            for x in indices[k]:
                decoder_input_ids = decoder_input_ids[0][:-1].view(1, decoder_input_ids.shape[1] - 1)

    best = torch.topk(max_probs, n_beams).indices

    best_n_indexes_from_nsquare = []
    best_n_probs_from_nsquare = None
    for i in best:
        best_n_indexes_from_nsquare.append(max_indexes[i])
        if best_n_probs_from_nsquare is None:
            best_n_probs_from_nsquare = max_probs[i].view(1, 1)
        else:

            best_n_probs_from_nsquare = torch.cat([best_n_probs_from_nsquare, max_probs[i].view(1, 1)], dim=1)

    return best_n_probs_from_nsquare, best_n_indexes_from_nsquare


@torch.no_grad()
def beam_search_decode(model, tokenizer, encoder_input_ids, n_beams, decoder_input_ids, min_length=5, max_length=50):
    final = []
    final_prob = None
    remove_this = []
    top_k_tokens = generate_best_n(model, tokenizer, encoder_input_ids, decoder_input_ids, min_length, n_beams)
    probs, roads = best_n_of_nsquare(model, tokenizer, top_k_tokens.indices, top_k_tokens.values, encoder_input_ids,
                                     decoder_input_ids, min_length, n_beams)

    check = True
    length_check = True
    while check:

        for i in range(len(roads)):
            if len(roads[i]) >= max_length:
                length_check = False
                final.append(roads[i])
                if final_prob is None:
                    final_prob = torch.tensor([probs[0][i]]).view(1, 1)
                else:
                    final_prob = torch.cat([final_prob, torch.tensor([probs[0][i]]).view(1, 1)], dim=1)
                remove_this.append(i)

            if roads[i][-1] == tokenizer.eos_token_id and length_check:
                if len(roads[i]) < min_length:
                    probs[0][i] = -1
                    continue
                final.append(roads[i])
                if final_prob is None:
                    final_prob = torch.tensor([probs[0][i]]).view(1, 1)
                else:
                    final_prob = torch.cat([final_prob, torch.tensor([probs[0][i]]).view(1, 1)], dim=1)
                remove_this.append(i)
            length_check = True

        remove_this.sort(reverse=True)
        for i in remove_this:
            roads.pop(i)
            probs = torch.cat([probs[0][:i], probs[0][i + 1:]]).view(1, -1)
            n_beams = n_beams - 1
        remove_this = []
        if n_beams == 0:
            check = False
            continue

        probs, roads = best_n_of_nsquare(model, tokenizer, roads, probs[0], encoder_input_ids, decoder_input_ids,
                                         min_length, n_beams)
    best = torch.topk(final_prob, 1).indices
    return final[best]

