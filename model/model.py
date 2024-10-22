import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules import ConvBatchNormReLU, SFA
from .modules import *
from .position_encoding import *

import clip
import math
import sys

sys.path.append('../')
from utils.utils import *


class Simple_fusion(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=768, proj_dim=1024, jemb_drop_out=0.1, leaky=True):
        super(Simple_fusion, self).__init__()
        self.proj_dim = proj_dim
        self.mapping_visu = ConvBatchNormReLU(visual_dim, proj_dim, 1, 1, 0, 1, leaky=leaky)
        self.lang_attn = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Dropout(jemb_drop_out),
            nn.Softmax(dim=1))
        
        self.lang_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.LeakyReLU(0.1))

        self.fusion = nn.Sequential(
            nn.BatchNorm2d(proj_dim),
            nn.LeakyReLU(0.1))
    
    def forward(self, visual_feat, lang_feat):
        # visual proj
        visual_feat_proj = self.mapping_visu(visual_feat) # [bt, 1024, 13, 13]
        
        """
        # lang attn
        lang_feat_attn = self.lang_attn(lang_feat) #[bt, 15, 768] 
        lang_feat_new = lang_feat * lang_feat_attn
        lang_feat_new = lang_feat_new.sum(dim=1) #[bt, 768]
        """

        lang_feat = lang_feat.squeeze(1)
        # lang proj
        #lang_feat_new = self.lang_proj(lang_feat_new) #[bt, 1024]
        lang_feat_new = self.lang_proj(lang_feat) #[bt, 1024]

        # fusion
        h, w = visual_feat.shape[-2], visual_feat.shape[-1]
        lang_feat_new_tile = lang_feat_new.view(-1, self.proj_dim, 1, 1).repeat(1, 1, h, w) # [bt, 1024, 13, 13]
        fusion_feat = lang_feat_new_tile * visual_feat_proj
        fusion_feat = self.fusion(fusion_feat)
        return fusion_feat

class up_proj_cat_proj(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(up_proj_cat_proj, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_2, input_2, 1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input_1+input_2, do, 1, 1, 0, 1, leaky=leaky)
    
    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        y = self.proj1(y)
        out = torch.cat([x,y], dim=1)
        out = self.proj2(out)
        return out

class pool_proj_cat_proj(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(pool_proj_cat_proj, self).__init__()
        self.downsample = nn.AvgPool2d(2, 2)
        self.proj1 = ConvBatchNormReLU(input_2, do // 2,    1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(do // 2, do,         3, 1, 1, 1, leaky=leaky)
        self.proj3 = ConvBatchNormReLU(input_1+do, do,      1, 1, 0, 1, leaky=leaky)

    def forward(self, x, y):
        y = self.downsample(y)
        y = self.proj1(y)
        y = self.proj2(y)
        output = self.proj3(torch.cat([x,y], dim=1))
        return output

class proj_cat_proj(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(proj_cat_proj, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_2, input_2,        1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input_1 + input_2, do,   1, 1, 0, 1, leaky=leaky)
    
    def forward(self, x, y):
        y = self.proj1(y)
        out = torch.cat([x, y], dim=1)
        out = self.proj2(out)
        return out

class proj_cat(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(proj_cat, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_1, do // 2,    1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(do // 2, do,         3, 1, 1, 1, leaky=leaky)

    def forward(self, x, y):
        x = self.proj1(x)
        x = self.proj2(x)
        output = torch.cat([x,y], dim=1)
        return output

class mask_decoder(nn.Module):
    def __init__(self, input_1, seg_out_stride=2, leaky=True):
        super(mask_decoder, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_1, input_1//2, 3, 1, 1, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input_1//2, input_1//2, 3, 1, 1, 1, leaky=leaky)

        self.proj3 = ConvBatchNormReLU(input_1//2, input_1//2, 3, 1, 1, 1, leaky=leaky)
        self.proj4 = ConvBatchNormReLU(input_1//2, input_1//2, 3, 1, 1, 1, leaky=leaky)
        self.proj5 = ConvBatchNormReLU(input_1//2, input_1//2, 3, 1, 1, 1, leaky=leaky)
        #self.proj = nn.Conv2d(input_1, 1, 3, 1, 1, 1)
        self.proj = nn.Conv2d(input_1//2, 32, 3, 1, 1, 1)

    def forward(self, x, seg_out_stride):
        x = self.proj1(x)
        x = self.proj2(x)


        if seg_out_stride <= 8:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.proj3(x)

        if seg_out_stride <= 4:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.proj4(x)

        if seg_out_stride <= 2:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.proj5(x)

        x = self.proj(x)
        
        return x


# class FeatureSelector(nn.Module):
#     def __init__(self, img_feature_dim, text_feature_dim, output_dim):
#         super(FeatureSelector, self).__init__()
#         # 使用nn.Sequential来简化MLP的构建
#         self.mlp = nn.Sequential(
#             nn.Linear(img_feature_dim * 3 + text_feature_dim * 3, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )

#     def forward(self, img_features, text_feature):
#         # 将图像特征和文本特征拼接
#         combined_features = torch.cat(img_features + text_feature, dim=1) # 
#         # 通过MLP得到输出得分
#         scores = self.mlp(combined_features)
#         return scores


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionblk(nn.Module):
    def __init__(self, clip_module):
        super().__init__()

        self.clip_module = clip_module

        self.selected_tokens = int(676 * 0.8)

        #self.norm = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, lang_tokens=None, index=0):


        if lang_tokens is None:
            x = x + self.clip_module.attention(self.clip_module.ln_1(x))
        else:

            #if index >= 4 and index <= 7:
            #    self.selected_tokens = int (676 * 0.8)
            #elif index>=8 and index <=11:
            #    self.selected_tokens = int (676 * 0.5) 
            #print(index)
            #print(self.selected_tokens)

            N, B, C = x.shape   # N x B x C
            cls_x = x[:1, :, :] # 1 x B x C
            x = x[1:, :, :]     # M x B x C

            ###img_cls text_cls
            #x = torch.mul(x, cls_x)
            #x = self.norm(x.reshape((N-1)*B, C))
            #x = x.reshape(N-1, B, C)

            ### text eos token
            #score = torch.bmm(x.transpose(0,1), lang_tokens).squeeze(-1)
            
            ### text features mean
            score = torch.bmm(x.transpose(0, 1), lang_tokens.permute(1, 2, 0)).mean(dim=-1)   # B x N
            score = score.transpose(0, 1)   # N x B

            sorted_scores, sorted_indices = torch.sort(score, descending=True, dim=0)

            # high_mask = sorted_scores > sorted_scores[self.selected_tokens:self.selected_tokens+1, :]
            high_mask = torch.ones_like(sorted_scores)
            for i in range(B):
                high_mask[sorted_indices[self.selected_tokens:, i], i] = 0
            high_mask = high_mask > 0.5

            delta_x = x[high_mask].reshape(-1, B, C)        # M x B x C
            low_x = x[~high_mask].reshape(-1, B, C)         # N-M x B x C
            low_score = score[~high_mask].reshape(-1, B, 1) # N-M x B x 1

            low_x = low_x * torch.softmax(low_score, dim=0) # N-M x B x C
            low_x = low_x.sum(dim=0, keepdim=True)          # 1 x B x C

            delta_x = torch.cat([cls_x, delta_x, low_x], dim=0) # M+1 x B x C
            delta_x = self.clip_module.attention(self.clip_module.ln_1(delta_x))

            # for i in range(B):
            #     x[high_mask[:, i], i, :] += delta_x[1:-1, i, :]
            #     x[~high_mask[:, i], i, :] += delta_x[-1:, i, :]
            #     cls_x[:, i] += delta_x[:1, i, :]
            temple = torch.zeros_like(x).type(delta_x.type())
            temple[high_mask] = delta_x[1:-1, :, :].reshape(-1, C)
            temple[~high_mask] = delta_x[-1:, :, :].reshape(-1, 1, C).repeat(1, 676 - self.selected_tokens, 1).reshape(-1, C)
            x = x + temple
            cls_x = cls_x + delta_x[:1, :, :]

            x = torch.cat([cls_x, x], dim=0)

        x = x + self.clip_module.mlp(self.clip_module.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, clip_model='RN50', tunelang=False, fusion_dim=2048, num_query=16, do=512, leaky=True, length=17):
        super(Model, self).__init__()

        self.tunelang = tunelang
        self.length = length

        ## Init Encoders
        clip_models = clip.load(clip_model, jit=False, device=torch.device("cpu"))[0].cuda()

        self.visumodel = clip_models.visual
        self.visu_dim = 768

        self.cut_list = []
        self.visu_resblocks = nn.ModuleList([ResidualAttentionblk(self.visumodel.transformer.resblocks[i]) for i in range(12)])
        self.visu_proj = nn.ModuleList([nn.Linear(do, self.visu_dim) for _ in range(len(self.cut_list))])

        self.positional_embedding = nn.Parameter(torch.FloatTensor(1, 26 ** 2 + 1, 768))
        v = self.resize_pos_embed(self.visumodel.positional_embedding.data.unsqueeze(0), self.positional_embedding, 26, 26)
        self.positional_embedding.data.copy_(v)

        self.textmodel = clip_models.transformer
        self.textmodel_token_embedding = clip_models.token_embedding
        self.textmodel_pos_embed = nn.Parameter(clip_models.positional_embedding[:self.length, :].unsqueeze(0))
        self.textmodel_ln_final = clip_models.ln_final
        self.textdim = self.textmodel_pos_embed.shape[-1]
        for module in self.textmodel.resblocks:
            module.attn_mask = self.build_attention_mask()

        # vis select
        self.vis_select = nn.Linear(self.visu_dim, do, bias=False)

        ## Fusion

        # fusion with x12
        self.fusion = Simple_fusion(visual_dim=self.visu_dim, text_dim=self.textdim, proj_dim=fusion_dim)

        # fusion with x6
        self.up_proj_cat_proj_1 = proj_cat_proj(input_1=fusion_dim, input_2=self.visu_dim, do=fusion_dim)
        self.pool_proj_cat_proj_2 = proj_cat_proj(input_1=fusion_dim, input_2=self.visu_dim, do=do)
        
        # fusion with x9
        self.proj_cat = proj_cat(input_1=fusion_dim, input_2=do, do=do)
        self.up_proj_cat_2 = proj_cat_proj(input_1=fusion_dim, input_2=do * 2, do=do)     
        self.proj_0 = ConvBatchNormReLU(do, do, 1, 1, 0, 1, leaky=leaky)

        self.fpn = SFA(in_channels=self.visu_dim, out_channels=do)

        ## Align dim
        f_dim = 512
        self.fc_2 = nn.Linear(f_dim, f_dim, bias=False)
        self.norm1 = nn.LayerNorm(f_dim)
        self.norm2 = nn.LayerNorm(f_dim)
        
        # visual branch
        self.pos_embedding = PositionEmbeddingSine(f_dim)
        encoder_layer = TransformerEncoderLayer(f_dim, nhead=8, dim_feedforward=f_dim,
                                                dropout=0.1, activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=nn.LayerNorm(f_dim))

        ## Decoder
        self.mask_decoder = mask_decoder(f_dim, seg_out_stride=2) 

        # text branch
        
        ## coef
        self.lang_tf_enc = lang_tf_enc(do, do, do, head_num=8)
        self.proj1 = ConvBatchNormReLU(do, do, 3, 1, 1, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(do, do, 3, 1, 1, 1, leaky=leaky)
        self.proj3 = nn.Conv2d(do, 32, 3, 1, 1, 1)
        self.projout = nn.Linear(26*26*32, 32, bias=False)
        
        
        self.feature_selector_l = nn.Linear(do, 1, bias=True) 
        self.feature_selector_m = nn.Linear(do, 1, bias=True)

    def resize_pos_embed(self, posemb, posemb_new, hight, width):
        ntok_new = posemb_new.shape[1]

        posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1

        gs_old = int(math.sqrt(len(posemb_grid)))
        print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        return posemb


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.length, self.length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image, word_id, word_mask):
        ## Visual Module

        batch_size = image.size(0)

        # Extract features from vision
        x = self.visumodel.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visumodel.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.visumodel.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        raw_fword = self.textmodel_token_embedding(word_id).squeeze(1)
        raw_fword = raw_fword + self.textmodel_pos_embed
        raw_fword = raw_fword.permute(1, 0, 2) # NLD -> LND
      
        visu_list_l = []
        visu_list_m = []
        
        scores_l = []
        scores_m = []

        for i, [blk_visu, blk_lang] in enumerate(zip(self.visu_resblocks, self.textmodel.resblocks)):
            x = blk_visu(x) # [677, bs, 768]
            raw_fword = blk_lang(raw_fword)

            img_cls = self.vis_select(x[0, :, :]) # [B, C]
            tex_cls = raw_fword[word_id.argmax(dim=-1).reshape(-1), torch.arange(raw_fword.shape[1]), :] # [B, C]
            score = img_cls * tex_cls # [B, C]
            score = score.unsqueeze(1) # [B, 1, C]
            
            if i >=3 and i <= 5:
                visu_list_l.append(x)
                scores_l.append(score)

            if i>=6 and i <=8:
                visu_list_m.append(x)
                scores_m.append(score)


        scores_l = torch.cat(scores_l, dim=1)  # [B, 3, C]
        scores_m = torch.cat(scores_m, dim=1)  # [B, 3, C]

        scores_l = self.feature_selector_l(scores_l).squeeze(-1) # [B, 3]
        scores_l = F.softmax(scores_l, dim=-1)
        scores_m = self.feature_selector_m(scores_m).squeeze(-1) # [B, 3]
        scores_m = F.softmax(scores_m, dim=-1)

        visu_list_l = torch.cat(visu_list_l, dim=0).reshape(len(visu_list_l), -1, batch_size, self.visu_dim).permute(0,2,1,3)
        visu_list_m = torch.cat(visu_list_m, dim=0).reshape(len(visu_list_m), -1, batch_size, self.visu_dim).permute(0,2,1,3)


        x6 = visu_list_l[scores_l.argmax(dim=-1).reshape(-1), torch.arange(visu_list_l.shape[1]), :, :].permute(1,0,2)
        x9 = visu_list_m[scores_m.argmax(dim=-1).reshape(-1), torch.arange(visu_list_m.shape[1]), :, :].permute(1,0,2)
         
        
        x6 = x6.permute(1, 0, 2)[:, 1:, :].reshape(-1, 26, 26, self.visu_dim).permute(0, 3, 1, 2)
        x9 = x9.permute(1, 0, 2)[:, 1:, :].reshape(-1, 26, 26, self.visu_dim).permute(0, 3, 1, 2)
        x12 = x.permute(1, 0, 2)[:, 1:, :]
        x12 = x12.reshape(-1, 26, 26, self.visu_dim).permute(0, 3, 1, 2) # [bs, 768, 26, 26]


        raw_fword = raw_fword.permute(1, 0, 2)
        raw_fword = self.textmodel_ln_final(raw_fword)
        
        if not self.tunelang:
            raw_fword = raw_fword.detach()

        eos_token = raw_fword[torch.arange(raw_fword.shape[0]), word_id.argmax(dim=-1).reshape(-1), :]

        F_g = self.fusion(x12, eos_token)
        F_tf = self.fpn([F_g, x9, x6])

        # Main body
        b,  c,  h,  w = F_tf.shape

        flatten_length = h*w
        visu_feat = F_tf.reshape(b, c, flatten_length)
        visu_feat = F.relu(visu_feat)
        lang_feat = F.relu(self.fc_2(raw_fword))

        visu_feat = visu_feat.permute(0, 2, 1)   
        pos_embed = self.pos_embedding(visu_feat)  
        visu_feat = visu_feat.transpose(0, 1)
        pos_embed = pos_embed.transpose(0, 1) 
        visu_feat = self.encoder(visu_feat, pos=pos_embed)
        #[HW B C]
        
        visu_feat_ = visu_feat.permute(1,0,2)

        # mask decoder
        visu_feat = visu_feat.reshape(h, w, b, c)
        visu_feat = visu_feat.permute(2,3,0,1)
        proto_masks = self.mask_decoder(visu_feat, 2)

        #[B C H W]
        proto_masks = F.relu(proto_masks)

        # coef
        coef = self.lang_tf_enc(visu_feat_, lang_feat)
        coef = coef.view(b, h, w, c)
        coef = coef.permute(0, 3, 1, 2)

        coef = self.proj1(coef)
        coef = self.proj2(coef)
        coef = self.proj3(coef)
        coef = coef.permute(0, 2, 3, 1)
        coef = coef.contiguous().view(b, h*w*32)
        # [b, 1, 32]
        coef = self.projout(coef).unsqueeze(-1)
        coef = F.tanh(coef)
        
        # mask assemble
        proto_masks = proto_masks.permute(0, 2, 3, 1)
        proto_masks = proto_masks.view(b, -1, 32)
        #[B HW N] [32 208*208 32]

        mask_out = torch.bmm(proto_masks, coef, out=None)
        mask_out = mask_out.view(b, 208, 208, 1)
        mask_out = mask_out.permute(0, 3, 1, 2)
        return mask_out
        