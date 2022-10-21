import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch import Tensor
from typing import Optional, Callable


def window_fn(window_type: Optional[str] = None) -> Callable[..., torch.Tensor]:
    if window_type == 'hann':
        return torch.hann_window
    else:
        raise ValueError(f"Cannot recognise this type of window.")


def _conv_bn_relu(inc, ouc, ks=3, s=1, p=1, d=1, bias=True):
    """ Conv-BN-ReLU
        Args:
            inc: in_channels
            ouc: out_channels
              ks: kernel_size
               s: stride
               p: padding
               d: dilation
            bias: bias
               
        Return:
            Torch.Tensor
    """
    return nn.Sequential(
        nn.Conv2d(inc, ouc, ks, s, p, d, bias=bias),
        nn.BatchNorm2d(ouc),
        nn.ReLU(inplace=True)
        )


def _func_conv_bn_relu(
    x: Tensor,
    conv_weights: Tensor,
    conv_biases: Tensor,
    bn_weights: Tensor,
    bn_biases: Tensor
) -> Tensor:
    """ Performs 3x3 convolution, BatchNorm, ReLU activation in a functional fashion.
        Args:
            x: Input Tensor for the conv block
            conv_weights: Weights for the convolutional block
            conv_biases: Biases for the convolutional block
            bn_weights:
            bn_biases:
    """
    x = F.conv2d(x, conv_weights, conv_biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    return x


class Vgg8(nn.Module):
    """ A VGG8 network for audio classification achieved following Kong2018."""
    def __init__(
            self,
            num_class: int = -1,
            sample_rate: int = 44100,
            n_fft: int = 1024,
            win_length: int = 1024,
            hop_length: int = 512,
            f_min: int = 0,
            f_max: Optional[int] = None,
            n_mels: int = 64,
            window_type: str = 'hann',
            include_top: bool = True
    ) -> None:
        super(Vgg8, self).__init__()
        self.include_top = include_top
        # Mel-spectrogram feature extractor
        self.mel_spectrogram = MelSpectrogram(sample_rate, n_fft, win_length, hop_length,
                                              f_min, f_max, n_mels=n_mels, window_fn=window_fn(window_type))
        self.power_to_db = AmplitudeToDB(stype='power')
        self.freq_normalizer = nn.BatchNorm2d(n_mels)
        # Conv kernel followed by bn
        self.conv1_1 = _conv_bn_relu(1, 64, 3)
        self.conv1_2 = _conv_bn_relu(64, 64, 3)
        self.conv2_1 = _conv_bn_relu(64, 128, 3)
        self.conv2_2 = _conv_bn_relu(128, 128, 3)
        self.conv3_1 = _conv_bn_relu(128, 256, 3)
        self.conv3_2 = _conv_bn_relu(256, 256, 3)
        self.conv4_1 = _conv_bn_relu(256, 512, 3)
        self.conv4_2 = _conv_bn_relu(512, 512, 3)
        # Affine operations followed by sigmoid
        self.fc = nn.Linear(512, 256)
        if self.include_top:
            self.logits = nn.Linear(256, num_class)


    def forward(self, x: Tensor):
        # Extract mel spectrogram from raw waveform.
        x = self.mel_spectrogram(x).unsqueeze_(1)  # output.shape = (n_samples, n_channels=1, n_mels, n_frames)
        x = self.power_to_db(x)
        # Normalise mel spectrogram along frequency bins
        x = x.transpose(1, 2)
        x = self.freq_normalizer(x)
        x = x.transpose(1, 2)
        # Conv1 [64, 64], followed by max pool
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = F.max_pool2d(x, 2)
        # Conv2 [128, 128], followed by max pool
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool2d(x, 2)
        # Conv3 [256, 256], followed by max pool
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = F.max_pool2d(x, 2)
        # Conv4 [512, 512]
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # affine projection followed by sigmoid
        # x.size() = (n_samples, n_channels, n_mels, n_frames)
        x = torch.mean(x, dim=2)
        x_max, _ = torch.max(x, dim=2)
        x_mean = torch.mean(x, dim=2)
        x = x_mean + x_max

        x = F.relu(self.fc(x))

        if not self.include_top:
            return x
        else:
            return self.logits(x)


    def functional_forward(self, x: Tensor, weights: dict) -> Tensor:
        # Applies the same forward pass using PyTorch functional operators with a specified set of weights
        x = self.mel_spectrogram(x).unsqueeze_(1)
        x = self.power_to_db(x)

        x = x.transpose(1, 2)
        x = self.freq_normalizer(x)
        x = F.batch_norm(x, running_mean=None, running_var=None, training=True,
                         weight=weights['freq_normalizer.weight'], bias=weights['freq_normalizer.bias'])
        x = x.transpose(1, 2)

        for block_id in range(1, 5):
            x = _func_conv_bn_relu(x, weights[f'conv{block_id}_1.0.weight'], weights[f'conv{block_id}_1.0.bias'],
                                      weights.get(f'conv{block_id}_1.1.weight'), weights.get(f'conv{block_id}_1.1.bias'))
            x = _func_conv_bn_relu(x, weights[f'conv{block_id}_2.0.weight'], weights[f'conv{block_id}_2.0.bias'],
                                      weights.get(f'conv{block_id}_2.1.weight'), weights.get(f'conv{block_id}_2.1.bias'))
            x = F.max_pool2d(x, 2)

        x = torch.mean(x, dim=2)
        x_max, _ = torch.max(x, dim=2)
        x_mean = torch.mean(x, dim=2)
        x = x_mean + x_max

        x = F.relu(F.linear(x, weights['fc.weight'], weights['fc.bias']))

        if not self.include_top:
            return x
        else:
            return F.linear(x, weights['logits.weight'], weights['logits.bias'])


class MatchingNetwork(nn.Module):
    """  Creates Matching Network as described in https://github.com/oscarknagg/few-shot.
         One should also refer to Appendix session in the paper 'Matching Networks for One Shot Learning, Vinyals et al.'
         Args:
            lstm_layers: number of LSTM layers in the bi-LSTM g that embeds the support set (fce = True)
            unrolling_steps: number of unrolling steps to run the Attention LSTM
            lstm_input_size: input size for the bidirectional and Attention LSTM
            device: GPU device
            mono: bool
    """
    def __init__(
            self,
            lstm_layers: int,
            unrolling_steps: int,
            lstm_input_size: int = 256,
            device: Optional[torch.device] = None,
            mono: bool = True
    ) -> None:
        super(MatchingNetwork, self).__init__()
        self.encoder = Vgg8(include_top=False).to(device)
        self.f = AttentionLSTM(lstm_input_size, unrolling_steps, device=device, mono=mono).to(device)
        self.g = BiLSTM(lstm_input_size, lstm_layers).to(device)

    def forward(self, inputs):
        pass


class BiLSTM(nn.Module):
    def __init__(self, size, num_layers):
        """ Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set
            as described in https://github.com/oscarknagg/few-shot
            ones should also refer to the original paper https://arxiv.org/pdf/1606.04080.pdf.
        :param size: Size of input and hidden layers. These are constrained to be the same in order to implement
                     the skip connection described in Appendix A.2.
        :param num_layers: Number of LSTM layers
        """
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=num_layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (h_n, c_n) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]
        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, h_n, c_n


class AttentionLSTM(nn.Module):
    def __init__(self, size, num_layers, device, mono):
        """ Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
            in the Matching Networks paper.
        :param size: Size of input and hidden layers. These are constrained to be the same in order to implement the
                     skip connection described in Appendix A.2
        :param num_layers: Number of layers in a regular LSTM, acting as unrolling_steps in the original paper
        :param device: GPU device
        :param mono: Bool, change to sigmoid instead of softmax when calculating attention matrix
        """
        super(AttentionLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_cell = nn.LSTMCell(input_size=size, hidden_size=size)
        self.device = device
        self.mono = mono

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries)
        c = torch.zeros(batch_size, embedding_dim).to(self.device)

        for k in range(self.num_layers):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries
            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            if self.mono:
                attentions = attentions.softmax(dim=1)
            else:
                attentions = attentions.sigmoid()
            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)
            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))
        h = h_hat + queries
        return h


if __name__ == '__main__':
    # Hyper-params
    # with open('../cfgs/classicalESC50.yaml', 'r') as f:
    #     cfgs = yaml.safe_load(f)
    #
    # sample_rate = cfgs['FEATURE_EXTRACTOR']['SAMPLE_RATE']
    # n_fft = cfgs['FEATURE_EXTRACTOR']['N_FFT']
    # win_length = cfgs['FEATURE_EXTRACTOR']['WIN_LENGTH']
    # hop_length = cfgs['FEATURE_EXTRACTOR']['HOP_LENGTH']
    # f_min = cfgs['FEATURE_EXTRACTOR']['F_MIN']
    # f_max = cfgs['FEATURE_EXTRACTOR']['F_MAX']
    # n_mels = cfgs['FEATURE_EXTRACTOR']['N_MELS']
    # if cfgs.WINDOW_TYPE == 'hann':
    #     window_fn = torch.hann_window

    model = Vgg8(num_class=50,
                 sample_rate=44100,
                 n_fft=1024,
                 win_length=1024,
                 hop_length= 512,
                 f_min= 0,
                 f_max= None,
                 n_mels= 64,
                 window_type='hann',
                 include_top=True)
    # summary(model, (1, 64, 431))
    print(model.state_dict().keys())
    # model = nn.Sequential(model, nn.Linear(256, 10))
    # for name, param in model.named_parameters():
    #     if name != 'fc.weight' and name != 'fc.bias':
    #         param.requires_grad = False
    # print(list(model.named_parameters()))
