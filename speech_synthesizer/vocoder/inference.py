from .models.fatchord_version import WaveRNN
from .hparams import *
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=voc_rnn_dims,
        fc_dims=voc_fc_dims,
        bits=bits,
        pad=voc_pad,
        upsample_factors=voc_upsample_factors,
        feat_dims=num_mels,
        compute_dims=voc_compute_dims,
        res_out_dims=voc_res_out_dims,
        res_blocks=voc_res_blocks,
        hop_length=hop_length,
        sample_rate=sample_rate,
        mode=voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True,  batched=True, target=8000, overlap=800, 
                   progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, mu_law, progress_callback)
    return wav
