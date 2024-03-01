import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

''' architecture for small decoder'''
class TinyVAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        self.feature_size = 2 
        self.feature_volume = kernel_num * (self.feature_size ** 2)


        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)
       
        # decoder
        decoder_config = {
            32: [1, 2, 4, 8, 8],
            64: [1, 2, 4, 8, 8, 8],
            128: [1, 2, 4, 8, 8, 8, 8],
            256: [1, 2, 4, 8, 8, 8, 8, 8],
        }

        decoder_shape = decoder_config[self.image_size]
        decoder = []
        for i in range(len(decoder_shape)-1):
            decoder.append(self._deconv(kernel_num//decoder_shape[i], kernel_num // decoder_shape[i+1]))

        decoder.append(nn.Conv2d(kernel_num // decoder_shape[-1], out_channels=channel_num, kernel_size=3, padding=1))
        decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder)


    def forward(self, image_syn):
        # sample latent code z from q given x.
        mean, logvar = image_syn[:, :, 0], image_syn[:, :, 1]
        # mean = mean + 1.*Variable(torch.randn_like(mean))
        # logvar = logvar + 1.*Variable(torch.randn_like(logvar))
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the reconstructed image.
        return x_reconstructed
    
    def forward_fixed(self, image_syn, eps=None):
        # sample latent code z from q given x.
        if len(image_syn.shape) == 3:
            mean, logvar = image_syn[:, :, 0], image_syn[:, :, 1]
        elif len(image_syn.shape) == 2:
            mean, logvar = image_syn[:, 0], image_syn[:, 1]
        else:
            raise Exception("Dimension Error")

        # no noise added
        std = logvar.mul(0.5).exp_() 
        if eps is None:
            z = mean
        else:
            z = mean + eps
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return  x_reconstructed


    # ==============
    # decoder components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)


    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'Decoder'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.LeakyReLU(),
        )

    def _deconv(self, channel_num, kernel_num, stride=2, kernel_size=4):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size, stride=stride, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.LeakyReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
    
''' architecture of medium and large decoder'''
class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 linear_decode: bool = True,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        
        self.linear_decode = linear_decode

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        if self.linear_decode:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = hidden_dims[-1] * 4

        # Build Decoder
        modules = []

        if self.linear_decode:
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        if self.linear_decode:
            result = self.decoder_input(z)
        else:
            result = z
        result = result.view(z.shape[0], self.latent_dim // 4 , 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        # return mu

    def forward(self, input: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        # input = input.reshape(-1, 3, 64, 64) # reshape
        mu, log_var = input[:, :, 0], input[:, :, 1]
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
    
    def forward_fixed(self, input: torch.Tensor, eps=None) -> list[torch.Tensor]:
        # input = input.reshape(-1, 3, 64, 64) # reshape
        mu, log_var = input[:, :, 0], input[:, :, 1]
        z = mu
        return self.decode(z)
