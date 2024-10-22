import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import lang_tf_enc, TransformerEncoderLayer, TransformerEncoder
from .position_encoding import PositionEmbeddingSine

class SFA(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors = [1, 2, 4], fuse_type="sum"):
        super(SFA, self).__init__()
        self.stages = []
        for idx, scale in enumerate(scale_factors):
            out_dim = out_channels
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    nn.BatchNorm2d(
                    num_features=in_channels // 2, eps=1e-5, momentum=0.999, affine=True),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
                ]
                out_dim = in_channels // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)]
                out_dim = in_channels // 2
            elif scale == 1.0:
                layers = []
                out_dim = in_channels
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    ConvBatchNormReLU(out_dim, out_channels, 1, 1, 0, 1, leaky=True),
                    ConvBatchNormReLU(out_channels, out_channels, 3, 1, 1, 1, leaky=True),
                ]
            )
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
        
        self.stages = nn.ModuleList(self.stages)
            
        # 假设所有输入特征图的通道数相同
        self.lateral_convs = nn.ModuleList([
            ConvBatchNormReLU(out_channels, out_channels, 1, 1, 0, 1, leaky=True) for _ in range(3)
        ])
        
        self.output_convs = nn.ModuleList([
            ConvBatchNormReLU(out_channels, out_channels, 3, 1, 1, 1, leaky=True) for _ in range(3)
        ])

        self._fuse_type = fuse_type  # or "avg"

        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        '''
        Args:
            x: list[Tensor], T个特征图，每个特征图的尺寸和通道数相同，[x12, x9, x6]
        '''
        # 模拟bottom-up, 获取多尺度特征图
        mutil_scale_features = []
        for idx, stage in enumerate(self.stages):
            mutil_scale_features.append(stage(x[idx]))
        
        # top-down
        results = []
        prev_features = self.lateral_convs[0](mutil_scale_features[0])

        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = mutil_scale_features[idx]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features) # 1x1卷积
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))
        
        fused_features = self.downsample(results[0]) # 1/4分辨率，需要转换为1/16分辨率

        return fused_features

class ConvBatchNormReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False)
        # nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu" if leaky else "relu")   

        if instance:
            self.bn = nn.InstanceNorm2d(num_features=out_channels)
        else:
            self.bn = nn.BatchNorm2d(
                    num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
                )

        if leaky:
            self.relu = nn.LeakyReLU(0.1)
        elif relu:
            self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# class ConvBatchNormReLU(nn.Sequential):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride,
#         padding,
#         dilation,
#         leaky=False,
#         relu=True,
#         instance=False,
#     ):
#         super(ConvBatchNormReLU, self).__init__()

#         conv = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 bias=False,
#         )
#         nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="leaky_relu" if leaky else "relu")
        
#         self.add_module(
#             "conv", conv
#         )

#         if instance:
#             self.add_module(
#                 "bn",
#                 nn.InstanceNorm2d(num_features=out_channels),
#             )
#         else:
#             self.add_module(
#                 "bn",
#                 nn.BatchNorm2d(
#                     num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
#                 ),
#             )

#         if leaky:
#             self.add_module("relu", nn.LeakyReLU(0.1))
#         elif relu:
#             self.add_module("relu", nn.ReLU())

#     def forward(self, x):
#         return super(ConvBatchNormReLU, self).forward(x)


def concat_coord(x):
    ins_feat = x  # [bt, c, h, w] [512, 26, 26]
    batch_size, c, h, w = x.size()

    float_h = float(h)
    float_w = float(w)

    y_range = torch.arange(0., float_h, dtype=torch.float32)
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = torch.arange(0., float_w, dtype=torch.float32)
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = x_range[None, :]
    y_range = y_range[:, None]
    x = x_range.repeat(h, 1)
    y = y_range.repeat(1, w)

    x = x[None, None, :, :]
    y = y[None, None, :, :]
    x = x.repeat(batch_size, 1, 1, 1)
    y = y.repeat(batch_size, 1, 1, 1)
    x = x.cuda()
    y = y.cuda()

    ins_feat_out = torch.cat((ins_feat, x, x, x, y, y, y), 1)

    return ins_feat_out


class query_generator(nn.Module):
    def __init__(self, input, output, leaky=True):
        super(query_generator, self).__init__()
        self.proj1 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj3 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj = nn.Conv2d(input+6, output, 1, 1, 0, 1)

    def forward(self, x):
        x = concat_coord(x)
        x = x + self.proj1(x)
        x = x + self.proj2(x)
        x = x + self.proj3(x)
        x = self.proj(x)
        return x


class KLM(nn.Module):
    def __init__(self, f_dim, feat_dim):
        super(KLM, self).__init__()
        self.lang_tf_enc = lang_tf_enc(f_dim, f_dim, f_dim, head_num=8)

        self.pos_embedding = PositionEmbeddingSine(f_dim)
        encoder_layer = TransformerEncoderLayer(f_dim, nhead=8, dim_feedforward=f_dim,
                                                dropout=0.1, activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=nn.LayerNorm(f_dim))

        # self.catproj = nn.Linear(f_dim * 2, f_dim)

        self.fc_ker = nn.Linear(f_dim, feat_dim + feat_dim)
        self.fc_vis = nn.Linear(f_dim, feat_dim + feat_dim)
        self.ker_norm = nn.LayerNorm(feat_dim)
        self.vis_norm = nn.LayerNorm(feat_dim)

        self.channel_fc = nn.Linear(feat_dim, feat_dim)
        self.channel_norm = nn.LayerNorm(feat_dim)

        self.spatial_fc = nn.Linear(feat_dim, feat_dim)
        self.spatial_norm = nn.LayerNorm(feat_dim)

        self.out_fc = nn.Linear(feat_dim, f_dim)
        self.out_norm = nn.LayerNorm(f_dim)

        self.d_model = f_dim
        self.feat_dim = feat_dim
        self.resolution_size = 26

    def forward(self, kernel, lang_feat, visu_feat):
        # kernel    B x N x C
        # lang_feat B x T x C
        # visu_feat B x C x HW
        kernel = self.lang_tf_enc(kernel, lang_feat)    
        # B x N x C
        bs, c, hw = visu_feat.shape
        bq, nq, cq = kernel.shape
        bl, ll, cl = lang_feat.shape

        # Image Attention
        visu_feat = visu_feat.permute(0, 2, 1)      
        # B x HW x C
        pos_embed = self.pos_embedding(visu_feat)   
        # B x HW x C

        visu_feat = visu_feat.transpose(0, 1)
        pos_embed = pos_embed.transpose(0, 1)               
        visu_feat_ = self.encoder(visu_feat, pos=pos_embed)  # HW x B x C
        visu_feat_ = visu_feat_.transpose(0, 1)     # B x HW x C
        
        # repeat visual feats
        visu_feat = visu_feat_.unsqueeze(dim=1) # B x 1 x HW x C
        kernel = kernel.unsqueeze(dim=2)        # B x N x  1 x C
        lang_feat = lang_feat.unsqueeze(dim=2)  # B x Q x  1 x C
        
        kernel_in = self.fc_ker(kernel)
        kernel_out = kernel_in[:, :, :, self.feat_dim:]
        kernel_in =  kernel_in[:, :, :, :self.feat_dim]

        vis_in = self.fc_vis(visu_feat)
        vis_out = vis_in[:, :, :, self.feat_dim:]
        vis_in  = vis_in[:, :, :, :self.feat_dim]

        gate_feat = self.ker_norm(kernel_in) * self.vis_norm(vis_in)
        #[B N HW 64]

        channel_gate = self.channel_norm(self.channel_fc(gate_feat))
        channel_gate = channel_gate.mean(2, keepdim=True)   
        channel_gate = torch.sigmoid(channel_gate)
        # B x N x 1 x C

        spatial_gate = self.spatial_norm(self.spatial_fc(gate_feat))
        # spatial_gate = spatial_gate.mean(3, keepdim=True)   
        spatial_gate = torch.sigmoid(spatial_gate)          
        # B x N x HW x C

        channel_gate = (1 + channel_gate) * kernel_out      # B x N x 1 x C
        channel_gate = channel_gate.squeeze(2)              # B x N x C

        spatial_gate = (1 + spatial_gate) * vis_out         # B x N x HW x C
        spatial_gate = spatial_gate.mean(2)                 # B x N x C
        
        gate_feat = (channel_gate + spatial_gate) / 2
        # [B N 64]
        gate_feat = self.out_fc(gate_feat)
        gate_feat = self.out_norm(gate_feat)
        gate_feat = F.relu(gate_feat)
        #[B N C]

        #visu_feat_.transpose(1, 2) [B C HW]
        return gate_feat, visu_feat_.transpose(1, 2)


class KAM(nn.Module):
    def __init__(self, f_dim, num_query):
        super(KAM, self).__init__()

        self.k_size = 1

        self.proj = nn.Linear(26*26, f_dim)

        self.fc_k = nn.Linear(f_dim, f_dim)
        self.fc_m = nn.Linear(f_dim, f_dim)
        self.fc_fus = nn.Linear(f_dim * 2, f_dim)
        self.fc_out = nn.Linear(f_dim, 1)

        self.outproj = ConvBatchNormReLU(num_query, f_dim, 3, 1, 1, 1, leaky=True)
        self.maskproj = nn.Conv2d(f_dim, 1, 3, 1, 1, 1)

        self.bn = nn.BatchNorm2d(f_dim)

        self.mask_fcs = []
        for _ in range(3):
            self.mask_fcs.append(nn.Linear(f_dim, f_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(f_dim))
            self.mask_fcs.append(nn.ReLU())
        self.mask_fcs = nn.Sequential(*self.mask_fcs)


    def forward(self, kernel, visu_feat):
        # kernel [B N C]
        # visu_feat [B C HW]
        kernel = self.mask_fcs(kernel)

        B, N, C = kernel.shape
        kernel_ = kernel
        kernel = kernel.reshape(B, N, -1, C).permute(0, 1, 3, 2)    # B x N x C x 1
        kernel = kernel.reshape(B, N, C, self.k_size, self.k_size)  # B x N x C x 1 x 1
        #[B N C K K]
        visu_feat_ = visu_feat
        visu_feat = visu_feat.reshape(B, C, 26, 26)                 # B x C x H x W

        masks = []
        for i in range(B):
            masks.append(F.conv2d(visu_feat[i: i+1], kernel[i], padding=int(self.k_size // 2)))   # 1 x N x H x W
        masks = torch.cat(masks, dim=0)     # B x N x H x W

        feats = masks.reshape(B, N, -1)     # B x N x HW
        feats = self.proj(feats)            # B x N x C

        weights_kern = F.relu(self.fc_k(kernel_))
        weights_mask = F.relu(self.fc_m(feats))

        weights = torch.cat([weights_kern, weights_mask], dim=-1)   # B x N x 2C
        weights = F.relu(self.fc_fus(weights))                      # B x N x C
        weights = self.fc_out(weights)                              # B x N x 1
        weights = F.softmax(weights, dim=1)                         # B x N x 1

        weights = weights.unsqueeze(-1)     # B x N x 1 x 1

        mask = weights * masks              # B x N x H x W
        mask = self.outproj(mask)           # B x C x H x W
        mask = self.maskproj(mask)          
        mask = F.sigmoid(mask)              # B x 1 x H x W

        visu_feat = visu_feat * mask        # B x C x H x W

        visu_feat = self.bn(visu_feat)
        visu_feat = visu_feat.reshape(B, C, -1) + visu_feat_
        visu_feat = F.relu(visu_feat)
        return visu_feat

        