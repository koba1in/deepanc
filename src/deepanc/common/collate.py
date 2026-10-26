from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    batch = [sequence.squeeze(0) for sequence in batch]
    waveforms = pad_sequence(
        batch,
        batch_first=True,
        padding_value=0,
    )
    waveforms = waveforms.unsqueeze(1)
    return waveforms