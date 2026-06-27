import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel, norm=nn.InstanceNorm3d, use_skip=False, bias=False):
            super(CNNEncoder.Encoder, self).__init__()

            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )

            self.use_skip = use_skip

        def forward(self, feature_map):

            mid = self._input(feature_map)

            res = self._output(mid) if not self.use_skip else mid + self._output(mid)

            return res

    def __init__(self, depth, base, norm):
        super(CNNEncoder, self).__init__()

        self.__input = CNNEncoder.Encoder(1, base, norm=norm)

        self.__encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                                       CNNEncoder.Encoder(base * 2 ** i, base * 2 ** (i + 1), norm))
                                         for i in range(depth)])

    def forward(self, x):
        skips = []

        in_encoder = self.__input(x)

        skips.append(in_encoder)

        for encoder in self.__encoders:
            in_encoder = encoder(in_encoder)

            skips.append(in_encoder)

        return in_encoder, skips


class CNNDecoder(nn.Module):

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel, block_num, norm=nn.InstanceNorm3d, use_skip=False, bias=True):
            super(CNNDecoder.Decoder, self).__init__()

            self.__input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.__mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )

            self._output = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=bias),
                    norm(out_channel),
                    nn.ReLU()
                )
                for _ in range(block_num)
            ])

            self.use_skip = use_skip

        def forward(self, x, skip):
            x = self.__input(x)

            mid = self.__mid(torch.cat([x, skip], dim=1))

            return self._output(mid) if self.use_skip else mid + self._output(mid)

    def __init__(self, depth, base, block_num, norm, use_skip):
        super(CNNDecoder, self).__init__()

        self.__decoders = nn.ModuleList(
            [CNNDecoder.Decoder(base * 2 ** i, base * 2 ** (i - 1), block_num, norm, use_skip)
             for i in range(depth, 0, -1)])

    def forward(self, x, skips):
        skips.pop()

        skips.reverse()

        for decoder, skip in zip(self.__decoders, skips):
            x = decoder(x, skip)

        return x


class DIPNet(nn.Module):

    def __init__(self, depth, base, decoder_block_num, norm=nn.InstanceNorm3d, encoder_norm=nn.Identity, use_skip=False):
        super(DIPNet, self).__init__()

        self.encoder = CNNEncoder(depth, base, encoder_norm)

        self.decoder = CNNDecoder(depth, base, decoder_block_num, norm=norm, use_skip=use_skip)

        self.output = nn.Conv3d(base, 1, 1, 1, 0, bias=False)

    def forward(self, x):

        btm, skips = self.encoder(x)

        top = self.decoder(btm, skips)

        return self.output(top)


