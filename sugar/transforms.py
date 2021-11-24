import torch
import torch.nn.functional as F
import torchaudio

class PreEmphasis(torch.nn.Module):
    """
    PreEmphasis is applied to waveforms. It define as a conv1d.
    """
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        self.in_channels = 1
        self.groups = 1
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        # buffer can be .cuda() instead of nn.Parameter
        self.register_buffer(
            'weight', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer('bias', None)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.size()) == 2, 'The dimensions of x tensor must be 2!'
        # reflect padding to match lengths of in/out
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.weight).squeeze(1)


class LogMelFbanks(torch.nn.Module):
    """
    Log Mel-scale filter banks with mean and variance normalization.
        We used 'n_mels' Mel filter bank log-energies with a 25ms window
        every 10ms over the spectral bank of 20-7600 Hz. Mean and variance
        normalization is applied among input waveform.
    """

    def __init__(self, 
                 n_mels: int = 80,
                 log_input: bool = True,
                 autocast: bool = False,
                 masker: bool = False):
        super(LogMelFbanks, self).__init__()
        self.n_mels       = n_mels
        self.log_input    = log_input
        self.autocast     = autocast
        self.instancenorm = torch.nn.InstanceNorm1d(self.n_mels)
        self.torchfb      = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20., f_max=7600., 
                window_fn=torch.hamming_window, n_mels=self.n_mels)
        )
        self.masker = masker
        if self.masker:
            self.fmasking = torchaudio.transforms.FrequencyMasking(10, True)
            self.tmasking = torchaudio.transforms.TimeMasking(5, True)


    def forward(self, x):
        """
            x: Tensor (batch, wav_length)

            Return
            ------
                x : Tensor (batch, 1, feature_num, frame_num)
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                x = self.torchfb(x.float()) + 1e-6
                if self.log_input is True:
                    x = x.log()
                x = self.instancenorm(x).unsqueeze(1)
                if self.masker:
                    x = self.fmasking(x)
                    x = self.tmasking(x)
        return x


class NormMFCC(torch.nn.Module):
    def __init__(self, n_mfcc: int = 80, autocast: bool = False, masker: bool = False):
        super(NormMFCC, self).__init__()
        self.n_mfcc       = n_mfcc
        self.autocast     = autocast
        self.instancenorm = torch.nn.InstanceNorm1d(self.n_mfcc)
        melkwargs = {
            'n_fft': 512, 'win_length': 400, 'hop_length': 160, 'f_min': 20, 'f_max': 7600, 'window_fn': torch.hamming_window, 'n_mels': 128
        }
        self.torchmfcc = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc, dct_type=2, norm='ortho', log_mels=False, melkwargs=melkwargs)
        )
        self.masker = masker
        if self.masker:
            self.fmasking = torchaudio.transforms.FrequencyMasking(10, True)
            self.tmasking = torchaudio.transforms.TimeMasking(5, True)

    def forward(self, x):
        """
            x: Tensor (batch, wav_length)

            Return
            ------
                x : Tensor (batch, 1, feature_num, frame_num)
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                x = self.torchmfcc(x.float())
                x = self.instancenorm(x).unsqueeze(1)
                if self.masker:
                    x = self.fmasking(x)
                    x = self.tmasking(x)
        return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    trans = LogMelFbanks()
    # trans = NormMFCC()
    inp, sample_rate = torchaudio.load_wav('/workspace/datasets/TIMIT/TEST/DR1/FAKS0/SA1.WAV')
    print(inp.size(), sample_rate, inp[0,:3])
    spec = trans(inp)
    print(spec.shape)
    plt.figure(figsize=(8, 2))
    plt.subplot(121)
    plt.plot(inp[0])
    plt.subplot(122)
    plt.imshow(spec[0,0])
    plt.tight_layout()
    plt.show()
    # plt.savefig('/workspace/projects/sugar/tmp1.png', dpi=300)
    print(spec.min(), spec.max())
    
