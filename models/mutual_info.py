# refer to https://github.com/JingZhang617/cascaded_rgbd_sod/blob/main/model/ResNet_models_combine.py#L47
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.distributions import Normal, Independent, kl
from .utils import Conv1dNormRelu, Conv2dNormRelu


def torch_L2normalize(x, d=1):
    eps = 1e-6
    norm = x ** 2
    norm = norm.sum(dim=d, keepdim=True) + eps
    norm = norm ** (0.5)
    return (x / norm)


class Mutual_info_reg_2D(nn.Module):
    def __init__(self, input_channels, hidden_channels, is_l2norm=True, is_train=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.is_l2norm = is_l2norm
        self.is_train = is_train

        self.rgb_mu = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.rgb_logvar = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_mu = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_logvar = Conv2dNormRelu(input_channels, hidden_channels, activation=None)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, point_feat):
        _, _, H, W = rgb_feat.shape

        if self.is_l2norm:
            rgb_feat = torch_L2normalize(rgb_feat, d=1)
            point_feat = torch_L2normalize(point_feat, d=1)

        mu_rgb = torch.tanh(self.rgb_mu(rgb_feat))
        logvar_rgb = torch.tanh(self.rgb_logvar(rgb_feat))
        mu_point = torch.tanh(self.point_mu(point_feat))
        logvar_point = torch.tanh(self.point_logvar(point_feat))

        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_point = self.reparametrize(mu_point, logvar_point)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_point = Independent(Normal(loc=mu_point, scale=torch.exp(logvar_point)), 1)

        z_rgb_norm = torch.sigmoid(z_rgb)
        z_point_norm = torch.sigmoid(z_point)
        ce_rgb_point = binary_cross_entropy(z_rgb_norm, z_point_norm.detach())
        ce_point_rgb = binary_cross_entropy(z_point_norm, z_rgb_norm.detach())

        bi_di_kld = torch.mean(kl.kl_divergence(dist_rgb, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_rgb))

        latent_loss = ce_rgb_point + ce_point_rgb - bi_di_kld
        latent_loss /= H * W

        return latent_loss, z_rgb, z_point


class Mutual_info_reg_2D_Event(nn.Module):
    def __init__(self, input_channels, hidden_channels, is_l2norm=True, is_train=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.is_l2norm = is_l2norm
        self.is_train = is_train

        self.rgb_mu = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.rgb_logvar = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_mu = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_logvar = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.event_mu = Conv2dNormRelu(input_channels, hidden_channels, activation=None)
        self.event_logvar = Conv2dNormRelu(input_channels, hidden_channels, activation=None)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def forward(self, rgb_feat, point_feat, event_feat):
        _, _, H, W = rgb_feat.shape

        if self.is_l2norm:
            rgb_feat = torch_L2normalize(rgb_feat, d=1)
            point_feat = torch_L2normalize(point_feat, d=1)
            event_feat = torch_L2normalize(event_feat, d=1)

        mu_rgb = torch.tanh(self.rgb_mu(rgb_feat))
        logvar_rgb = torch.tanh(self.rgb_logvar(rgb_feat))
        mu_point = torch.tanh(self.point_mu(point_feat))
        logvar_point = torch.tanh(self.point_logvar(point_feat))
        mu_event = torch.tanh(self.event_mu(event_feat))
        logvar_event = torch.tanh(self.event_logvar(event_feat))

        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_point = self.reparametrize(mu_point, logvar_point)
        z_event = self.reparametrize(mu_event, logvar_event)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_point = Independent(Normal(loc=mu_point, scale=torch.exp(logvar_point)), 1)
        dist_event = Independent(Normal(loc=mu_event, scale=torch.exp(logvar_event)), 1)

        z_rgb_norm = torch.sigmoid(z_rgb)
        z_point_norm = torch.sigmoid(z_point)
        z_event_norm = torch.sigmoid(z_event)

        ce_rgb_point = binary_cross_entropy(z_rgb_norm, z_point_norm.detach())
        ce_point_rgb = binary_cross_entropy(z_point_norm, z_rgb_norm.detach())
        ce_rgb_event = binary_cross_entropy(z_rgb_norm, z_event_norm.detach())
        ce_event_rgb = binary_cross_entropy(z_event_norm, z_rgb_norm.detach())
        ce_point_event = binary_cross_entropy(z_point_norm, z_event_norm.detach())
        ce_event_point = binary_cross_entropy(z_event_norm, z_point_norm.detach())

        bi_di_kld_rgb_point = torch.mean(kl.kl_divergence(dist_rgb, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_rgb))
        bi_di_kld_event_point = torch.mean(kl.kl_divergence(dist_event, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_event))
        bi_di_kld_rgb_event = torch.mean(kl.kl_divergence(dist_rgb, dist_event)) + \
            torch.mean(kl.kl_divergence(dist_event, dist_rgb))

        latent_loss = ce_rgb_point + ce_point_rgb + ce_rgb_event + ce_event_rgb \
            + ce_point_event + ce_event_point \
            - bi_di_kld_rgb_point - bi_di_kld_event_point - bi_di_kld_rgb_event
        latent_loss /= H * W

        return latent_loss, z_rgb, z_point, z_event


class Mutual_info_reg_3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, is_l2norm=True, is_train=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.is_l2norm = is_l2norm
        self.is_train = is_train

        self.rgb_mu = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.rgb_logvar = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_mu = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_logvar = Conv1dNormRelu(input_channels, hidden_channels, activation=None)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def forward(self, rgb_feat, point_feat):
        _, _, N = rgb_feat.shape

        if self.is_l2norm:
            rgb_feat = torch_L2normalize(rgb_feat, d=1)
            point_feat = torch_L2normalize(point_feat, d=1)

        mu_rgb = torch.tanh(self.rgb_mu(rgb_feat))
        logvar_rgb = torch.tanh(self.rgb_logvar(rgb_feat))
        mu_point = torch.tanh(self.point_mu(point_feat))
        logvar_point = torch.tanh(self.point_logvar(point_feat))

        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_point = self.reparametrize(mu_point, logvar_point)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_point = Independent(Normal(loc=mu_point, scale=torch.exp(logvar_point)), 1)

        z_rgb_norm = torch.sigmoid(z_rgb)
        z_point_norm = torch.sigmoid(z_point)
        ce_rgb_point = binary_cross_entropy(z_rgb_norm, z_point_norm.detach())
        ce_point_rgb = binary_cross_entropy(z_point_norm, z_rgb_norm.detach())

        bi_di_kld = torch.mean(kl.kl_divergence(dist_rgb, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_rgb))

        latent_loss = ce_rgb_point + ce_point_rgb - bi_di_kld
        latent_loss /= N

        return latent_loss, z_rgb, z_point


class Mutual_info_reg_3D_Event(nn.Module):
    def __init__(self, input_channels, hidden_channels, is_l2norm=True, is_train=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.is_l2norm = is_l2norm
        self.is_train = is_train

        self.rgb_mu = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.rgb_logvar = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_mu = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.point_logvar = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.event_mu = Conv1dNormRelu(input_channels, hidden_channels, activation=None)
        self.event_logvar = Conv1dNormRelu(input_channels, hidden_channels, activation=None)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def forward(self, rgb_feat, point_feat, event_feat):
        _, _, N = rgb_feat.shape

        if self.is_l2norm:
            rgb_feat = torch_L2normalize(rgb_feat, d=1)
            point_feat = torch_L2normalize(point_feat, d=1)
            event_feat = torch_L2normalize(event_feat, d=1)

        mu_rgb = torch.tanh(self.rgb_mu(rgb_feat))
        logvar_rgb = torch.tanh(self.rgb_logvar(rgb_feat))
        mu_point = torch.tanh(self.point_mu(point_feat))
        logvar_point = torch.tanh(self.point_logvar(point_feat))
        mu_event = torch.tanh(self.event_mu(event_feat))
        logvar_event = torch.tanh(self.event_logvar(event_feat))

        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_point = self.reparametrize(mu_point, logvar_point)
        z_event = self.reparametrize(mu_event, logvar_event)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_point = Independent(Normal(loc=mu_point, scale=torch.exp(logvar_point)), 1)
        dist_event = Independent(Normal(loc=mu_event, scale=torch.exp(logvar_event)), 1)

        z_rgb_norm = torch.sigmoid(z_rgb)
        z_point_norm = torch.sigmoid(z_point)
        z_event_norm = torch.sigmoid(z_event)

        ce_rgb_point = binary_cross_entropy(z_rgb_norm, z_point_norm.detach())
        ce_point_rgb = binary_cross_entropy(z_point_norm, z_rgb_norm.detach())
        ce_rgb_event = binary_cross_entropy(z_rgb_norm, z_event_norm.detach())
        ce_event_rgb = binary_cross_entropy(z_event_norm, z_rgb_norm.detach())
        ce_point_event = binary_cross_entropy(z_point_norm, z_event_norm.detach())
        ce_event_point = binary_cross_entropy(z_event_norm, z_point_norm.detach())

        bi_di_kld_rgb_point = torch.mean(kl.kl_divergence(dist_rgb, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_rgb))
        bi_di_kld_event_point = torch.mean(kl.kl_divergence(dist_event, dist_point)) + \
            torch.mean(kl.kl_divergence(dist_point, dist_event))
        bi_di_kld_rgb_event = torch.mean(kl.kl_divergence(dist_rgb, dist_event)) + \
            torch.mean(kl.kl_divergence(dist_event, dist_rgb))

        latent_loss = ce_rgb_point + ce_point_rgb + ce_rgb_event + ce_event_rgb \
            + ce_point_event + ce_event_point \
            - bi_di_kld_rgb_point - bi_di_kld_event_point - bi_di_kld_rgb_event
        latent_loss /= N

        return latent_loss, z_rgb, z_point, z_event

