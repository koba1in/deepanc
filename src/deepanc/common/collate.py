from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    
    waveforms = pad_sequence(
        batch,
        batch_first=True,
        padding_value=0,
    )

    return waveforms