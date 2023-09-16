import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, interpolate
from .pwc2d_core import FeaturePyramid2D, ContextNetwork2D, FlowEstimator2D
from .pwc3d_core import FeaturePyramid3D, FlowEstimator3D, Correlation3D
from .utils import Conv1dNormRelu, Conv2dNormRelu, project_feat_with_nn_corr, grid_sample_wrapper, project_pc2image, mesh_grid
from .utils import backwarp_2d, backwarp_3d, knn_interpolation, convex_upsample
from .csrc import correlation2d, k_nearest_neighbor
from .restormer_arch import CrossTransformerBlock2D, CrossTransformerBlock3D
from .mutual_info import Mutual_info_reg_2D, Mutual_info_reg_3D
from .mutual_info import Mutual_info_reg_2D_Event, Mutual_info_reg_3D_Event


class PyramidFeatureFuser2D(nn.Module):
    """
    Bi-CLFM (Bidirectional Camera-LiDAR Fusion Module)
    For clarity, we implementation Bi-CLFM with two individual modules: one for 2D->3D, 
    and the other for 3D->2D.
    This module is designed for pyramid feature fusion (3D->2D).
    """
    def __init__(self, in_channels_2d, in_channels_3d, num_heads, norm=None):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d + 3, in_channels_2d, norm=norm),
        )
        self.mi = Mutual_info_reg_2D(in_channels_2d, in_channels_2d//2)
        self.fuse = CrossTransformerBlock2D(dim=in_channels_2d, num_heads=num_heads)

    def forward(self, xy, feat_2d, feat_3d, nn_proj=None):
        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])

        out = self.mlps(feat_3d_to_2d)
        latent_loss, _, _ = self.mi(feat_2d, out)
        out = self.fuse(feat_2d, out)

        return out, latent_loss


class PyramidFeatureFuser3D(nn.Module):
    """Pyramid feature fusion (2D->3D)"""
    def __init__(self, in_channels_2d, in_channels_3d, num_heads, norm=None):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d, in_channels_3d, norm=norm),
        )
        self.mi = Mutual_info_reg_3D(in_channels_3d, in_channels_3d//2)
        self.fuse = CrossTransformerBlock3D(dim=in_channels_3d, num_heads=num_heads)

    def forward(self, xy, feat_2d, feat_3d):
        with torch.no_grad():
            feat_2d_to_3d = grid_sample_wrapper(feat_2d, xy)

        out = self.mlps(feat_2d_to_3d)
        latent_loss, _, _ = self.mi(feat_3d, out)
        # out = self.fuse(torch.cat([out, feat_3d], dim=1))
        out = self.fuse(feat_3d, out)

        return out, latent_loss


class CorrFeatureFuser2D(nn.Module):
    """Correlation feature fusion (3D->2D)"""
    def __init__(self, in_channels_2d, in_channels_3d, num_heads):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d * 2 + 5, in_channels_3d + in_channels_2d),
            Conv2dNormRelu(in_channels_3d + in_channels_2d, in_channels_2d),
        )
        self.head_3d = Conv2dNormRelu(in_channels_3d + 5, in_channels_2d)
        self.head_event = Conv2dNormRelu(in_channels_3d, in_channels_2d)
        self.mi = Mutual_info_reg_2D_Event(in_channels_2d, in_channels_2d//2)
        self.fuse = CrossTransformerBlock2D(dim=in_channels_2d, num_heads=num_heads)

    def forward(self, xy, feat_2d, feat_3d, efeat_2d, last_flow_2d, last_flow_3d_to_2d, nn_proj=None):
        feat_3d = torch.cat([feat_3d, last_flow_3d_to_2d], dim=1)

        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])
        feat_3d_to_2d[:, -2:] -= last_flow_2d.detach()

        latent_loss, _, _, _ = self.mi(feat_2d, self.head_3d(feat_3d_to_2d), self.head_event(efeat_2d))
        out = self.mlps(torch.cat([feat_3d_to_2d, efeat_2d], dim=1))
        out = self.fuse(feat_2d, out)

        return out, latent_loss


class CorrFeatureFuser3D(nn.Module):
    """Correlation feature fusion (2D->3D)"""
    def __init__(self, in_channels_2d, in_channels_3d, num_heads):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d + in_channels_3d + 2, in_channels_2d + in_channels_3d),
            Conv1dNormRelu(in_channels_2d + in_channels_3d, in_channels_3d),
        )
        self.head_2d = Conv1dNormRelu(in_channels_2d + 2, in_channels_3d)
        self.mi = Mutual_info_reg_3D_Event(in_channels_3d, in_channels_3d//2)
        # self.fuse = Conv1dNormRelu(in_channels_3d + in_channels_3d, in_channels_3d)
        self.fuse = CrossTransformerBlock3D(dim=in_channels_3d, num_heads=num_heads)

    def forward(self, xy, feat_corr_2d, feat_corr_3d, efeat_2d, last_flow_3d, last_flow_2d_to_3d):
        with torch.no_grad():
            feat_2d_with_flow = torch.cat([feat_corr_2d, last_flow_2d_to_3d], dim=1)
            feat_2d_to_3d_with_flow = grid_sample_wrapper(feat_2d_with_flow, xy)
            efeat_2d_to_3d = grid_sample_wrapper(efeat_2d, xy)
            feat_2d_to_3d = feat_2d_to_3d_with_flow
            feat_2d_to_3d[:, -2:] -= last_flow_3d[:, :2]

        latent_loss, _, _, _ = self.mi(feat_corr_3d, self.head_2d(feat_2d_to_3d), efeat_2d_to_3d)
        # out = self.mlps(feat_2d_to_3d)
        out = self.mlps(torch.cat([feat_2d_to_3d, efeat_2d_to_3d], dim=1))
        # out = self.fuse(torch.cat([out, feat_corr_3d], dim=1))
        out = self.fuse(feat_corr_3d, out)

        return out, latent_loss


class DecoderFeatureFuser2D(nn.Module):
    """Decoder feature fusion (3D->2D)"""
    def __init__(self, in_channels_2d, in_channels_3d, num_heads):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d + 3, in_channels_2d),
        )
        self.mi = Mutual_info_reg_2D(in_channels_2d, in_channels_2d//2)
        # self.fuse = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)
        self.fuse = CrossTransformerBlock2D(dim=in_channels_2d, num_heads=num_heads)

    def forward(self, xy, feat_2d, feat_3d, nn_proj=None):
        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])

        out = self.mlps(feat_3d_to_2d)
        latent_loss, _, _ = self.mi(feat_2d, out)
        # out = self.fuse(torch.cat([out, feat_2d], dim=1))
        out = self.fuse(feat_2d, out)

        return out, latent_loss


class DecoderFeatureFuser3D(nn.Module):
    """Decoder feature fusion (2D->3D)"""
    def __init__(self, in_channels_2d, in_channels_3d, num_heads):
        super().__init__()
        self.mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d, in_channels_3d),
        )
        self.mi = Mutual_info_reg_3D(in_channels_3d, in_channels_3d//2)
        # self.fuse = Conv1dNormRelu(in_channels_2d + in_channels_3d, in_channels_3d)
        self.fuse = CrossTransformerBlock3D(dim=in_channels_3d, num_heads=num_heads)

    def forward(self, xy, feat_2d, feat_3d):
        with torch.no_grad():
            feat_2d_to_3d = grid_sample_wrapper(feat_2d, xy)
        out = self.mlps(feat_2d_to_3d)
        latent_loss, _, _ = self.mi(feat_3d, out)
        # out = self.fuse(torch.cat([feat_2d_to_3d, feat_3d], dim=1))
        out = self.fuse(feat_3d, out)
        return out, latent_loss


class RPEFlow_core(nn.Module):
    def __init__(self, cfgs2d, cfgs3d, cfgsattention, debug=False):
        super().__init__()
        self.cfgs2d, self.cfgs3d, self.debug = cfgs2d, cfgs3d, debug
        self.cfgsattention = cfgsattention
        corr_channels_2d = (2 * cfgs2d.max_displacement + 1) ** 2
        event_bins = self.cfgs2d.event_bins * 2 if self.cfgs2d.event_polarity else self.cfgs2d.event_bins

        # PWC-Net 2D (IRR-PWC)
        self.feature_pyramid_2d = FeaturePyramid2D(
            [3, 16, 32, 64, 96, 128, 192],
            norm=cfgs2d.norm.feature_pyramid
        )
        self.feature_aligners_2d = nn.ModuleList([
            nn.Identity(),
            Conv2dNormRelu(32, 64),
            Conv2dNormRelu(64, 64),
            Conv2dNormRelu(96, 64),
            Conv2dNormRelu(128, 64),
            Conv2dNormRelu(192, 64),
        ])
        self.efeature_pyramid_2d = FeaturePyramid2D(
            [event_bins, 32, 32, 64, 96, 128, 192],
            norm=cfgs2d.norm.feature_pyramid
        )
        self.efeature_aligners_2d = nn.ModuleList([
            nn.Identity(),
            Conv2dNormRelu(32, 64),
            Conv2dNormRelu(64, 64),
            Conv2dNormRelu(96, 64),
            Conv2dNormRelu(128, 64),
            Conv2dNormRelu(192, 64),
        ])
        self.flow_estimator_2d = FlowEstimator2D(
            [64 + 64 + corr_channels_2d + 2 + 32, 192, 128, 96, 64, 32],
            norm=cfgs2d.norm.flow_estimator,
            conv_last=False,
        )
        self.context_network_2d = ContextNetwork2D(
            [self.flow_estimator_2d.flow_feat_dim + 2, 128, 128, 128, 96, 64, 32],
            dilations=[1, 2, 4, 8, 16, 1],
            norm=cfgs2d.norm.context_network
        )
        self.up_mask_head_2d = nn.Sequential(  # for convex upsampling (see RAFT)
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 4 * 9, kernel_size=1, stride=1, padding=0),
        )

        # PWC-Net 3D (Point-PWC)
        self.feature_pyramid_3d = FeaturePyramid3D(
            [16, 32, 64, 96, 128, 192],
            norm=cfgs3d.norm.feature_pyramid,
            k=cfgs3d.k,
        )
        self.feature_aligners_3d = nn.ModuleList([
            nn.Identity(),
            Conv1dNormRelu(32, 64),
            Conv1dNormRelu(64, 64),
            Conv1dNormRelu(96, 64),
            Conv1dNormRelu(128, 64),
            Conv1dNormRelu(192, 64),
        ])
        self.correlations_3d = nn.ModuleList([
            nn.Identity(),
            Correlation3D(32, 32, k=self.cfgs3d.k),
            Correlation3D(64, 64, k=self.cfgs3d.k),
            Correlation3D(96, 96, k=self.cfgs3d.k),
            Correlation3D(128, 128, k=self.cfgs3d.k),
            Correlation3D(192, 192, k=self.cfgs3d.k),
        ])
        self.correlation_aligners_3d = nn.ModuleList([
            nn.Identity(),
            Conv1dNormRelu(32, 64),
            Conv1dNormRelu(64, 64),
            Conv1dNormRelu(96, 64),
            Conv1dNormRelu(128, 64),
            Conv1dNormRelu(192, 64),
        ])
        self.flow_estimator_3d = FlowEstimator3D(
            [64 + 64 + 3 + 64, 128, 128, 64],
            cfgs3d.norm.flow_estimator,
            conv_last=False,
            k=self.cfgs3d.k,
        )

        # Bi-CLFM for pyramid features
        self.pyramid_feat_fusers_2d = nn.ModuleList([
            nn.Identity(),
            PyramidFeatureFuser2D(32, 32, num_heads=1, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(64, 64, num_heads=2, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(96, 96, num_heads=2, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(128, 128, num_heads=4, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(192, 192, num_heads=4, norm=cfgs2d.norm.feature_pyramid),
        ])
        self.pyramid_feat_fusers_3d = nn.ModuleList([
            nn.Identity(),
            PyramidFeatureFuser3D(32, 32, num_heads=1, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(64, 64, num_heads=2, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(96, 96, num_heads=2, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(128, 128, num_heads=4, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(192, 192, num_heads=4, norm=cfgs3d.norm.feature_pyramid),
        ])

        # Bi-CLFM for correlation features
        self.corr_feat_fusers_2d = nn.ModuleList([
            nn.Identity(),
            CorrFeatureFuser2D(corr_channels_2d, 32, num_heads=1),
            CorrFeatureFuser2D(corr_channels_2d, 64, num_heads=1),
            CorrFeatureFuser2D(corr_channels_2d, 96, num_heads=3),
            CorrFeatureFuser2D(corr_channels_2d, 128, num_heads=3),
            CorrFeatureFuser2D(corr_channels_2d, 192, num_heads=3),
        ])
        self.corr_feat_fusers_3d = nn.ModuleList([
            nn.Identity(),
            CorrFeatureFuser3D(corr_channels_2d, 32, num_heads=1),
            CorrFeatureFuser3D(corr_channels_2d, 64, num_heads=2),
            CorrFeatureFuser3D(corr_channels_2d, 96, num_heads=2),
            CorrFeatureFuser3D(corr_channels_2d, 128, num_heads=4),
            CorrFeatureFuser3D(corr_channels_2d, 192, num_heads=4),
        ])

        # Bi-CLFM for decoder features
        self.estimator_feat_fuser_2d = DecoderFeatureFuser2D(self.flow_estimator_2d.flow_feat_dim, 64, num_heads=2)
        self.estimator_feat_fuser_3d = DecoderFeatureFuser3D(self.flow_estimator_2d.flow_feat_dim, 64, num_heads=2)
    
        self.conv_last_2d = nn.Conv2d(self.flow_estimator_2d.flow_feat_dim, 2, kernel_size=3, stride=1, padding=1)
        self.conv_last_3d = nn.Conv1d(64, 3, kernel_size=1)

    def encode(self, image, xyzs):
        feats_2d = self.feature_pyramid_2d(image)
        feats_3d = self.feature_pyramid_3d(xyzs)
        return feats_2d, feats_3d

    def encode_event(self, event_voxel):
        return self.efeature_pyramid_2d(event_voxel)

    def decode(self, xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d, efeats_2d, camera_info):
        assert len(xyzs1) == len(xyzs2) == len(feats1_2d) == len(feats2_2d) == len(feats1_3d) == len(feats2_3d)

        flows_2d, flows_3d, flow_feats_2d, flow_feats_3d = [], [], [], []
        mi_loss = 0.0
        for level in range(len(xyzs1) - 1, 0, -1):
            xyz1, feat1_2d, feat1_3d = xyzs1[level], feats1_2d[level], feats1_3d[level]
            xyz2, feat2_2d, feat2_3d = xyzs2[level], feats2_2d[level], feats2_3d[level]
            efeat_2d = efeats_2d[level]

            batch_size, image_h, image_w = feat1_2d.shape[0], feat1_2d.shape[2], feat1_2d.shape[3]
            n_points = xyz1.shape[-1]

            # project point cloud to image
            xy1 = project_pc2image(xyz1, camera_info)
            xy2 = project_pc2image(xyz2, camera_info)

            # sensor coordinate -> image coordinate
            sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
            xy1[:, 0] *= (image_w - 1) / (sensor_w - 1)
            xy1[:, 1] *= (image_h - 1) / (sensor_h - 1)
            xy2[:, 0] *= (image_w - 1) / (sensor_w - 1)
            xy2[:, 1] *= (image_h - 1) / (sensor_h - 1)

            # pre-compute knn indices
            grid = mesh_grid(batch_size, image_h, image_w, xy1.device)  # [B, 2, H, W]
            grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]
            nn_proj1 = k_nearest_neighbor(xy1, grid, k=1)  # [B, HW, k]
            nn_proj2 = k_nearest_neighbor(xy2, grid, k=1)  # [B, HW, k]
            knn_1in1 = k_nearest_neighbor(xyz1, xyz1, k=self.cfgs3d.k)  # [bs, n_points, k]

            # fuse pyramid features
            feat1_2d_fused, mi2d_loss1 = self.pyramid_feat_fusers_2d[level](xy1, feat1_2d, feat1_3d, nn_proj1)
            feat2_2d_fused, mi2d_loss2 = self.pyramid_feat_fusers_2d[level](xy2, feat2_2d, feat2_3d, nn_proj2)
            feat1_3d_fused, mi3d_loss1 = self.pyramid_feat_fusers_3d[level](xy1, feat1_2d, feat1_3d)
            feat2_3d_fused, mi3d_loss2 = self.pyramid_feat_fusers_3d[level](xy2, feat2_2d, feat2_3d)
            feat1_2d, feat2_2d = feat1_2d_fused, feat2_2d_fused
            feat1_3d, feat2_3d = feat1_3d_fused, feat2_3d_fused

            if level == len(xyzs1) - 1:
                last_flow_2d = torch.zeros([batch_size, 2, image_h, image_w], dtype=xy1.dtype, device=xy1.device)
                last_flow_3d = torch.zeros([batch_size, 3, n_points], dtype=xy1.dtype, device=xy1.device)
                last_flow_feat_2d = torch.zeros([batch_size, 32, image_h, image_w], dtype=xy1.dtype, device=xy1.device)
                last_flow_feat_3d = torch.zeros([batch_size, 64, n_points], dtype=xy1.dtype, device=xy1.device)
                xyz2_warp, feat2_2d_warp = xyz2, feat2_2d
            else:
                # upsample 2d flow and backwarp
                last_flow_2d = interpolate(flows_2d[-1] * 2, scale_factor=2, mode='bilinear', align_corners=True)
                last_flow_feat_2d = interpolate(flow_feats_2d[-1], scale_factor=2, mode='bilinear', align_corners=True)
                feat2_2d_warp = backwarp_2d(feat2_2d, last_flow_2d, padding_mode='border')

                # upsample 3d flow and backwarp
                flow_with_feat_3d = torch.cat([flows_3d[-1], flow_feats_3d[-1]], dim=1)
                flow_with_feat_upsampled_3d = knn_interpolation(xyzs1[level + 1], flow_with_feat_3d, xyz1)
                last_flow_3d = flow_with_feat_upsampled_3d[:, :3, :]
                last_flow_feat_3d = flow_with_feat_upsampled_3d[:, 3:, :]
                xyz2_warp = backwarp_3d(xyz1, xyz2, last_flow_3d)

            # correlation (2D & 3D)
            feat_corr_3d = self.correlations_3d[level](xyz1, feat1_3d, xyz2_warp, feat2_3d, knn_1in1)
            feat_corr_2d = leaky_relu(correlation2d(feat1_2d, feat2_2d_warp, self.cfgs2d.max_displacement), 0.1)

            # fuse correlation features
            last_flow_3d_to_2d = torch.cat([
                last_flow_3d[:, 0:1] * (image_w - 1) / (sensor_w - 1),
                last_flow_3d[:, 1:2] * (image_h - 1) / (sensor_h - 1),
            ], dim=1)
            last_flow_2d_to_3d = torch.cat([
                last_flow_2d[:, 0:1] * (sensor_w - 1) / (image_w - 1),
                last_flow_2d[:, 1:2] * (sensor_h - 1) / (image_h - 1),
            ], dim=1)
            feat_corr_2d_fused, mi2d_loss3 = self.corr_feat_fusers_2d[level](
                xy1, feat_corr_2d, feat_corr_3d, efeat_2d, last_flow_2d, last_flow_3d_to_2d, nn_proj1
            )
            feat_corr_3d_fused, mi3d_loss3 = self.corr_feat_fusers_3d[level](
                xy1, feat_corr_2d, feat_corr_3d, efeat_2d, last_flow_3d, last_flow_2d_to_3d
            )
            feat_corr_2d, feat_corr_3d = feat_corr_2d_fused, feat_corr_3d_fused

            # align features using 1x1 convolution
            feat1_2d = self.feature_aligners_2d[level](feat1_2d)
            feat1_3d = self.feature_aligners_3d[level](feat1_3d)
            efeat_2d = self.efeature_aligners_2d[level](efeat_2d)
            feat_corr_3d = self.correlation_aligners_3d[level](feat_corr_3d)

            # flow decoder (or estimator)
            x_2d = torch.cat([feat_corr_2d, feat1_2d, efeat_2d, last_flow_2d, last_flow_feat_2d], dim=1)
            x_3d = torch.cat([feat_corr_3d, feat1_3d, last_flow_3d, last_flow_feat_3d], dim=1)
            flow_feat_2d = self.flow_estimator_2d(x_2d)  # [bs, 96, image_h, image_w]
            flow_feat_3d = self.flow_estimator_3d(xyz1, x_3d, knn_1in1)  # [bs, 64, n_points]

            # fuse decoder features
            flow_feat_2d_fused, mi2d_loss4 = self.estimator_feat_fuser_2d(xy1, flow_feat_2d, flow_feat_3d, nn_proj1)
            flow_feat_3d_fused, mi3d_loss4 = self.estimator_feat_fuser_3d(xy1, flow_feat_2d, flow_feat_3d)
            flow_feat_2d, flow_feat_3d = flow_feat_2d_fused, flow_feat_3d_fused

            # flow prediction
            flow_delta_2d = self.conv_last_2d(flow_feat_2d)
            flow_delta_3d = self.conv_last_3d(flow_feat_3d)

            # residual connection
            flow_2d = last_flow_2d + flow_delta_2d
            flow_3d = last_flow_3d + flow_delta_3d

            # context network (2D only)
            flow_feat_2d, flow_delta_2d = self.context_network_2d(torch.cat([flow_feat_2d, flow_2d], dim=1))
            flow_2d = flow_delta_2d + flow_2d

            # save results
            flows_2d.append(flow_2d)
            flows_3d.append(flow_3d)
            flow_feats_2d.append(flow_feat_2d)
            flow_feats_3d.append(flow_feat_3d)

            mi2d_loss = mi2d_loss1 + mi2d_loss2 + mi2d_loss3 + mi2d_loss4
            mi3d_loss = mi3d_loss1 + mi3d_loss2 + mi3d_loss3 + mi3d_loss4
            mi_loss += (10 * mi2d_loss + mi3d_loss) * (0.85 ** (level-1))

        flows_2d = [f.float() for f in flows_2d][::-1]
        flows_3d = [f.float() for f in flows_3d][::-1]

        # convex upsamling module, from RAFT
        flows_2d[0] = convex_upsample(flows_2d[0], self.up_mask_head_2d(flow_feats_2d[-1]), scale_factor=4)

        for i in range(1, len(flows_2d)):
            flows_2d[i] = interpolate(flows_2d[i] * 4, scale_factor=4, mode='bilinear', align_corners=True)

        for i in range(len(flows_3d)):
            flows_3d[i] = knn_interpolation(xyzs1[i + 1], flows_3d[i], xyzs1[i])

        return flows_2d, flows_3d, mi_loss
