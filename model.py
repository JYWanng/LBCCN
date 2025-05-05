import torch
import torch.nn as nn

def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
        + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class Model(nn.Module):
    def __init__(self, band_num = 40):
        super().__init__()
        self.band_num = band_num
        self.stft = self.Stft(n_dft = 256, hop_size = 128)
        self.istft = self.IStft(n_dft = 256, hop_size = 128)
        self.encoder = Encoder(band_num, kernel_size = 5)
        self.rnn = RNN(kernel_size = 9, num_layers = 3)
        self.A_decoder = Decoder(kernel_size = 9)
        self.B_decoder = Decoder(kernel_size = 9)
    
    def forward(self, inputs):
        xl = self.stft(inputs[:,0])
        xr = self.stft(inputs[:,1])
        xlc = xl[:,:self.band_num,:]
        xrc = xr[:,:self.band_num,:]
        xln = xl[:,self.band_num:,:]
        xrn = xr[:,self.band_num:,:]

        x = self.encoder(xlc, xrc, xln, xrn)
        x = self.rnn(x)
        a = self.A_decoder(x)
        b = self.B_decoder(x)

        vrc = (xlc - b*xrc) / (a - b)
        vlc = a*vrc
        nlc = xlc - vlc
        nrc = xrc - vrc

        vl = torch.cat((vlc,xln),dim=1)
        vr = torch.cat((vrc,xrn),dim=1)
        voice = torch.stack((self.istft(vl), self.istft(vr)),dim=1)
        nl = torch.cat((nlc,xln),dim=1)
        nr = torch.cat((nrc,xrn),dim=1)
        noise = torch.stack((self.istft(nl), self.istft(nr)),dim=1)

        return voice, noise, vlc, vrc, nlc, nrc
    
    class Stft(nn.Module):
        def __init__(self, n_dft=1024, hop_size=512, win_length=None,
                    onesided=True, is_complex=True):
            super().__init__()
            self.n_dft = n_dft
            self.hop_size = hop_size
            self.win_length = n_dft if win_length is None else win_length
            self.onesided = onesided
            self.is_complex = is_complex

        def forward(self, x: torch.Tensor):
            "Expected input has shape (batch_size, n_channels, time_steps)"
            window = torch.hann_window(self.win_length, device=x.device)
            y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        return_complex=True, window=window, normalized=True)
            # y.shape == (batch_size*channels, time, freqs)
            if not self.is_complex:
                y = torch.view_as_real(y)
                y = y.movedim(-1, 1) # move complex dim to front
            return y

    class IStft(Stft):
        def forward(self, x: torch.Tensor):
            "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
            window = torch.hann_window(self.win_length, device=x.device)
            y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                            win_length=self.win_length, onesided=self.onesided,
                            window=window,normalized=True)
            return y



class Encoder(nn.Module):
    def __init__(self, band_num = 40, kernel_size = 5):
        super().__init__()
        self.band_num = band_num
        self.conv_1_1 = self.Conv1dBlock(in_channels = band_num, out_channels = band_num, kernel_size = kernel_size, dilation = 1)
        self.conv_1_2 = self.Conv1dBlock(in_channels= 129 - band_num, out_channels = band_num, kernel_size = kernel_size, dilation = 1)
        self.conv_2 = self.Conv1dBlock(in_channels = band_num, out_channels = band_num, kernel_size = kernel_size, dilation = 2)
        self.conv_3 = self.Conv1dBlock(in_channels = band_num, out_channels = band_num, kernel_size = kernel_size, dilation = 4)
        
    def forward(self, xlc, xrc, xln, xrn):
        xl_1 = xlc
        xl_2 = xln
        xr_1 = xrc
        xr_2 = xrn
        convx_1 = self.conv_1_1(xl_1)
        convx_2 = self.conv_1_2(xl_2)
        xl = convx_1 + convx_2
        xl = self.conv_3(self.conv_2(xl))
        convx_1 = self.conv_1_1(xr_1)
        convx_2 = self.conv_1_2(xr_2)
        xr = convx_1 + convx_2
        xr = self.conv_3(self.conv_2(xr))
        x = torch.stack((xl,xr),dim=1)
        return x

    class Conv1dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            # Use `groups` option to implement depthwise convolution
            # [M, H, K] -> [M, H, K]
            depthwise_conv = self.ComplexConv1d(in_channels, in_channels, kernel_size, 
                                        padding=dilation * (kernel_size - 1), 
                                        dilation=dilation, groups=in_channels)
            # [M, H, K] -> [M, B, K]
            chomp = self.Chomp1d(dilation * (kernel_size - 1))
            pointwise_conv = self.ComplexConv1d(in_channels, out_channels, kernel_size=1)
            prelu = self.ComplexPReLU()
            norm = self.ComplexInstanceNorm1d(out_channels, affine=True)
            self.net = nn.Sequential(depthwise_conv, chomp, pointwise_conv, norm, prelu)

        def forward(self, x):
            return self.net(x)
        
        class ComplexConv1d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                        dilation=1, groups=1, bias=True):
                super().__init__()
                self.conv_r = nn.Conv1d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)
                self.conv_i = nn.Conv1d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)

            def forward(self, input):
                return apply_complex(self.conv_r, self.conv_i, input)

        class ComplexPReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.r_prelu = nn.PReLU(dtype=torch.float32)        
                self.i_prelu = nn.PReLU(dtype=torch.float32)

            def forward(self, input):
                result = self.r_prelu(input.real) + 1j*self.i_prelu(input.imag)
                return result

        class ComplexInstanceNorm1d(nn.Module):
            def __init__(self, num_features, affine=True,):
                super().__init__()
                self.bn_r = nn.InstanceNorm1d(num_features, affine)
                self.bn_i = nn.InstanceNorm1d(num_features, affine)

            def forward(self, input):
                return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)

        class Chomp1d(nn.Module):
            def __init__(self, chomp_size):
                super().__init__()
                self.chomp_size = chomp_size

            def forward(self, x):
                return x[:, :, :-self.chomp_size].contiguous()



class RNN(nn.Module):
    def __init__(self, kernel_size=9, num_layers=3):
        super().__init__()
        self.conv = self.Conv2dBlock(2, 16, kernel_size, dilation = 1)
        # self.rnn = self.ComplexConvGRU(input_channels=16, hidden_channels=16, kernel_size=kernel_size, num_layers=num_layers)

    def forward(self, inputs):
        x = self.conv(inputs.permute(0,1,3,2))
        # x = self.rnn(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return x
    
    class Conv2dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            depthwise_conv = self.ComplexConv2d(in_channels, in_channels, kernel_size,
                                        padding=self.get_padding_2d(kernel_size, dilation=(dilation, 1)),
                                        dilation=(dilation, 1), groups=in_channels)
            chomp = self.Chomp2d(dilation * (kernel_size - 1))
            pointwise_conv = self.ComplexConv2d(in_channels, out_channels, kernel_size=(1, 1))
            prelu = self.ComplexPReLU()
            norm = self.ComplexInstanceNorm2d(out_channels, affine=True)
            self.net = nn.Sequential(depthwise_conv, chomp, pointwise_conv, norm, prelu)

        @staticmethod
        def get_padding_2d(kernel_size, dilation=(1, 1)):
            return (int((kernel_size * dilation[0] - dilation[0])), int((kernel_size * dilation[1] - dilation[1]) / 2))

        def forward(self, x):
            return self.net(x)
        
        class ComplexConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                        dilation=1, groups=1, bias=True):
                super().__init__()
                self.conv_r = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)
                self.conv_i = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)

            def forward(self, input):
                return apply_complex(self.conv_r, self.conv_i, input)

        class ComplexPReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.r_prelu = nn.PReLU(dtype=torch.float32)        
                self.i_prelu = nn.PReLU(dtype=torch.float32)

            def forward(self, input):
                result = self.r_prelu(input.real) + 1j*self.i_prelu(input.imag)
                return result

        class ComplexInstanceNorm2d(nn.Module):
            def __init__(self, num_features, affine=True,):
                super().__init__()
                self.bn_r = nn.InstanceNorm2d(num_features, affine)
                self.bn_i = nn.InstanceNorm2d(num_features, affine)

            def forward(self, input):
                return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)
            
        class Chomp2d(nn.Module):
            def __init__(self, chomp_size):
                super().__init__()
                self.chomp_size = chomp_size

            def forward(self, x):
                return x[:, :, :-self.chomp_size,:].contiguous()

    class ComplexConvGRU(nn.Module):
        def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1):
            super().__init__()
            self.gru_re = self.ConvGRU(input_channels, hidden_channels, kernel_size, num_layers)
            self.gru_im = self.ConvGRU(input_channels, hidden_channels, kernel_size, num_layers)

        def forward(self, x):
            real, imag = x.real, x.imag
            r2r_out, r2i_out = self.gru_re(real), self.gru_im(real)
            i2r_out, i2i_out = self.gru_re(imag), self.gru_im(imag)
            real_out = r2r_out - i2i_out
            imag_out = i2r_out + r2i_out
            return torch.complex(real_out, imag_out)

        class ConvGRU(nn.Module):
            def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
                super().__init__()
                self.input_channels = input_channels
                self.hidden_channels = hidden_channels
                self.kernel_size = kernel_size
                self.num_layers = num_layers
                self.cell_list = nn.ModuleList([
                    self.ConvGRUCell(
                        input_channels if i == 0 else hidden_channels,
                        hidden_channels, kernel_size
                    ) for i in range(num_layers)
                ])

            def forward(self, x):
                layer_output_list = []
                hidden = None
                for layer_idx, cell in enumerate(self.cell_list):
                    output_inner = []
                    if hidden is None:
                        hidden = torch.zeros(x.size(0), self.hidden_channels, x.size(3), device=x.device)
                    for t in range(x.size(1)):
                        hidden = cell(x[:, t, :, :], hidden)
                        output_inner.append(hidden)
                    layer_output = torch.stack(output_inner, dim=1)
                    layer_output_list.append(layer_output)
                    hidden = layer_output[:, -1, :, :]
                return layer_output_list[-1]

            class ConvGRUCell(nn.Module):
                def __init__(self, input_channels, hidden_channels, kernel_size):
                    super().__init__()
                    padding = kernel_size // 2
                    self.input_channels = input_channels
                    self.hidden_channels = hidden_channels
                    self.kernel_size = kernel_size

                    self.conv_r = nn.Conv1d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, groups=input_channels)
                    self.norm_r = nn.InstanceNorm1d(hidden_channels, affine=True)

                    self.conv_z = nn.Conv1d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, groups=input_channels)
                    self.norm_z = nn.InstanceNorm1d(hidden_channels, affine=True)

                    self.conv_h = nn.Conv1d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, groups=input_channels)
                    self.norm_h = nn.InstanceNorm1d(hidden_channels, affine=True)

                def forward(self, x, h_prev):
                    combined = torch.cat((x, h_prev), dim=1)

                    r = torch.sigmoid(self.norm_r(self.conv_r(combined)))
                    z = torch.sigmoid(self.norm_z(self.conv_z(combined)))

                    combined_r = torch.cat((x, r * h_prev), dim=1)
                    h_hat = torch.tanh(self.norm_h(self.conv_h(combined_r)))

                    h_new = (1 - z) * h_prev + z * h_hat
                    return h_new


class Decoder(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.conv_1 = self.Conv2dBlock(16, 16, kernel_size, dilation = 1)
        self.conv_2 = self.Conv2dBlock(16, 16, kernel_size, dilation = 2)
        self.conv_3 = self.Conv2dBlock(16, 1, kernel_size, dilation = 4)

    def forward(self, inputs):
        x = self.conv_3(self.conv_2(self.conv_1(inputs))).squeeze(1).permute(0,2,1)
        return x

    class Conv2dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            depthwise_conv = self.ComplexConv2d(in_channels, in_channels, kernel_size,
                                        padding=self.get_padding_2d(kernel_size, dilation=(dilation, 1)),
                                        dilation=(dilation, 1), groups=in_channels)
            chomp = self.Chomp2d(dilation * (kernel_size - 1))
            pointwise_conv = self.ComplexConv2d(in_channels, out_channels, kernel_size=(1, 1))
            prelu = self.ComplexPReLU()
            self.net = nn.Sequential(depthwise_conv, chomp, pointwise_conv, prelu)

        @staticmethod
        def get_padding_2d(kernel_size, dilation=(1, 1)):
            return (int((kernel_size * dilation[0] - dilation[0])), int((kernel_size * dilation[1] - dilation[1]) / 2))

        def forward(self, x):
            return self.net(x)
        
        class ComplexConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                        dilation=1, groups=1, bias=True):
                super().__init__()
                self.conv_r = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)
                self.conv_i = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation, groups, bias)

            def forward(self, input):
                return apply_complex(self.conv_r, self.conv_i, input)

        class ComplexPReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.r_prelu = nn.PReLU(dtype=torch.float32)        
                self.i_prelu = nn.PReLU(dtype=torch.float32)

            def forward(self, input):
                result = self.r_prelu(input.real) + 1j*self.i_prelu(input.imag)
                return result

        class ComplexInstanceNorm2d(nn.Module):
            def __init__(self, num_features, affine=True,):
                super().__init__()
                self.bn_r = nn.InstanceNorm2d(num_features, affine)
                self.bn_i = nn.InstanceNorm2d(num_features, affine)

            def forward(self, input):
                return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)
            
        class Chomp2d(nn.Module):
            def __init__(self, chomp_size):
                super().__init__()
                self.chomp_size = chomp_size

            def forward(self, x):
                return x[:, :, :-self.chomp_size,:].contiguous()
 

if __name__ == '__main__':
    model = Model()
    inputs = torch.ones((1, 2, 16000))
    outputs = model(inputs)

    from thop import profile
    from thop import clever_format
    flops, params = profile(model, inputs=(inputs, ))
    # print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs,params)
