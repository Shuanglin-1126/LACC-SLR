import torch.nn as nn
import math
from functools import partial
import torch
import torch.nn.functional as F
from models.MViT.LA_attention import MultiScaleBlock
from models.MViT.common import TwoStreamFusion
from models.MViT.reversible_mvit import ReversibleMViT
from models.MViT.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
)
from torch.nn.init import trunc_normal_
from models.loss.info_loss import CenterLoss
from models.MViT import stem_helper  # noqa
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None
from collections import OrderedDict


class AggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_q = LayerNorm(d_model)
        self.ln_kv = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_mlp = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, kv):
        return self.attn(q, kv, kv, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, q: torch.Tensor, kv):
        q = q.transpose(0, 1)
        kv = kv.transpose(0, 1)
        q = q + self.attention(self.ln_q(q), self.ln_kv(kv))
        q = q + self.mlp(self.ln_mlp(q))
        return q.transpose(0, 1)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg, spatial_size=224, in_chans=3):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        temporal_size = cfg.DATA.NUM_FRAMES
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:
            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(embed_dim, dim_mul.prod(), divisor=num_heads)

            self.fuse = TwoStreamFusion(cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim)

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:
            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if cfg.MVIT.DIM_MUL_IN_ATT:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=cfg.MVIT.SEPARATE_QKV,
                )

                if cfg.MODEL.ACT_CHECKPOINT:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _get_pos_embed(self, pos_embed, bcthw):
        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :].reshape(1, p_t, p_h, p_w, -1).permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x):
        # (self, x, label=None, traget=None, lam=0, bboxes=None, return_attn=False) target为原始标签
        B, T, C, H, W = x.shape

        x, bcthw = self.patch_embed(x.transpose(1, 2))
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        # assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        return x, thw



class MViT_two_streams(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.keypoint_shape = cfg.DATA.pose_CROP_SIZE
        self.num_keypoint = cfg.DATA.INPUT_CHANNEL_NUM[1] - 1
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING

        self.rgb_backbone = MViT(cfg, spatial_size=cfg.DATA.TRAIN_CROP_SIZE, in_chans=cfg.DATA.INPUT_CHANNEL_NUM[0])
        self.pose_backbone = MViT(cfg, spatial_size=cfg.DATA.pose_CROP_SIZE, in_chans=cfg.DATA.INPUT_CHANNEL_NUM[1])

        self.fused_feature_layers = cfg.MVIT.FUSED_FEATURE_LAYERS
        self.fused_rgb = nn.ModuleList()
        self.fused_pose = nn.ModuleList()
        d_model = [192, 384, 768]
        n_head = [2, 4, 8]
        for i in range(len(self.fused_feature_layers)):
            fused_rgb = AggregationBlock(d_model=d_model[i], n_head=n_head[i])
            fused_pose = AggregationBlock(d_model=d_model[i], n_head=n_head[i])
            self.fused_rgb.append(fused_rgb)
            self.fused_pose.append(fused_pose)

        self.fused_rgb = checkpoint_wrapper(self.fused_rgb)
        self.fused_pose = checkpoint_wrapper(self.fused_pose)


        self.norm_rgb =LayerNorm(768)
        self.norm_pose = LayerNorm(768)
        self.fuse_feature = AggregationBlock(d_model=768, n_head=8)
        self.lamda = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

        self.head = CenterLoss(num_classes=cfg.MODEL.NUM_CLASSES, feat_dim=768)

        self.apply(self._init_weights)

        checkpoint = torch.load(r'path to classic MViTv2 checkpoint')
        load_matched_state_dict(self.rgb_backbone, checkpoint)
        load_matched_state_dict(self.pose_backbone, checkpoint)
        del checkpoint

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    def gen_gaussian_hmap(self, coords, crood_scores):

        skeletons = torch.tensor([[0, 1], [0, 2], [1, 3], [3, 5], [5, 7], [7, 9], [2, 4], [4, 6], [6, 8], [8, 10],
                                  [9, 11], [11, 12], [12, 13], [13, 14], [14, 15],
                                  [11, 16], [16, 17], [17, 18], [18, 19],
                                  [11, 20], [20, 21], [21, 22], [22, 23],
                                  [11, 24], [24, 25], [25, 26], [26, 27],
                                  [11, 28], [28, 29], [29, 30], [30, 31],
                                  [10, 32], [32, 33], [33, 34], [34, 35], [35, 36],
                                  [32, 37], [37, 38], [38, 39], [39, 40],
                                  [32, 41], [41, 42], [42, 43], [43, 44],
                                  [32, 45], [45, 46], [46, 47], [47, 48],
                                  [32, 49], [49, 50], [50, 51], [51, 52]])
        skeletons_num = len(skeletons)

        B, T = coords.shape[0], coords.shape[1]
        x, y = torch.meshgrid(torch.arange(self.keypoint_shape), torch.arange(self.keypoint_shape))
        grid = gird_sk = torch.stack([x.cuda(), y.cuda()], dim=2)  # [H,H,2]
        grid = grid.repeat((B, T, self.num_keypoint, 1, 1, 1))  # [B,T,n,H,H,2]
        gird_sk = gird_sk.repeat((B, T, skeletons_num, 1, 1, 1))
        sigma, sigma_sk = 3., 0.1

        x_idx = skeletons[..., 0]
        y_idx = skeletons[..., 1]
        coords /= 4
        start_coord = coords[:, :, x_idx, :]
        end_coord = coords[:, :, y_idx, :]
        start_score = crood_scores[..., x_idx]
        end_score = crood_scores[..., y_idx]
        max_score = torch.max(torch.stack([start_score, end_score], dim=-1), dim=-1)[0]

        heatmap = torch.exp(-1 * ((grid - coords[:, :, :, None, None, :]) ** 2).sum(dim=-1) / (2 * sigma ** 2)) / (sigma * (2 * torch.pi) ** 0.5)
        heatmap = heatmap * crood_scores[:, :, :, None, None] * 30
        dis = ((gird_sk - start_coord[:, :, :, None, None, :]) ** 2).sum(dim=-1) ** 0.5 + \
              ((gird_sk - end_coord[:, :, :, None, None, :]) ** 2).sum(dim=-1) ** 0.5 - \
              (((start_coord - end_coord) ** 2).sum(dim=-1) ** 0.5)[:, :, :, None, None]
        hmap_sk = torch.exp(-1 * dis / (2 * sigma_sk ** 2)) / (
                sigma_sk * (2 * torch.pi) ** 0.5)
        hmap_sk = hmap_sk * max_score[:, :, :, None, None] * 10

        hmap_sk = torch.max(hmap_sk, dim=2)[0]
        hmap = torch.concat([heatmap, hmap_sk.unsqueeze(2)], dim=2)
        return hmap


    def forward(self, x_rgb, keypoints, keypoint_scores, target=None):
        x_pose = self.gen_gaussian_hmap(keypoints.to(x_rgb.device), keypoint_scores.to(x_rgb.device))
        x_rgb, thw_rgb = self.rgb_backbone(x_rgb)
        x_pose, thw_pose = self.pose_backbone(x_pose.to(x_rgb.dtype))

        fused_idx = 0
        for idx in range(len(self.pose_backbone.blocks)):
            x_rgb, thw_rgb = self.rgb_backbone.blocks[idx](x_rgb, thw_rgb)
            x_pose, thw_pose = self.pose_backbone.blocks[idx](x_pose, thw_pose)

            if idx in self.fused_feature_layers:
                x_rgb = self.fused_rgb[fused_idx](x_rgb, x_pose)
                x_pose = self.fused_pose[fused_idx](x_pose, x_rgb)
                fused_idx += 1

        if self.use_mean_pooling:
            if self.cls_embed_on:
                x_rgb = x_rgb[:, 1:]
                x_pose = x_pose[:, 1:]
            x_rgb = x_rgb.mean(1)
            x_rgb = self.norm_rgb(x_rgb)
            x_pose = x_pose.mean(1)
            x_pose = self.norm_pose(x_pose)
        elif self.cls_embed_on:
            x_rgb = self.norm_rgb(x_rgb)
            x_rgb = x_rgb[:, 0]
            x_pose = self.norm_pose(x_pose)
            x_pose = x_pose[:, 0]
        else:  # this is default, [norm->mean]
            x_rgb = self.norm_rgb(x_rgb)
            x_rgb = x_rgb.mean(1)
            x_pose = self.norm_pose(x_pose)
            x_pose = x_pose.mean(1)

        feature = self.lamda[0] * x_rgb + self.lamda[1] * x_pose

        return self.head(feature, target)

def load_matched_state_dict(model, checkpoint, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict:
            if curr_state_dict[key].shape == state_dict[key].shape:
                curr_state_dict[key] = state_dict[key]
                num_matched += 1
            else:
                print(key)
        else:
            print(key)
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def mvit_s_two(num_frames=16, **kwargs):
    from models.MViT.defaults import get_cfg
    config_file = r'models/MViT/MVITv2_S_16x4.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.NUM_CLASSES = kwargs['num_class']
    cfg.MVIT.FUSED_FEATURE_LAYERS = [kwargs['fused_posi']]
    cfg.DATA.NUM_FRAMES = num_frames
    model = MViT_two_streams(cfg)

    return model

def mvit_b_two(num_frames=32, **kwargs):
    from models.MViT.defaults import get_cfg
    config_file = r'models/MViT/MVITv2_B_32x3.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.NUM_CLASSES = kwargs['num_class']
    cfg.DATA.NUM_FRAMES = num_frames
    model = MViT_two_streams(cfg)

    return model

if __name__ == '__main__':
    from models.MViT.defaults import get_cfg

    model = mvit_s_two(num_class=100).cuda().eval()
    x_1 = torch.randn(2, 16, 3, 224, 224).cuda()
    x_21 = torch.randn(2, 16, 53, 2)
    x_22 = torch.randn(2, 16, 53)
    y = model(x_1, x_21, x_22)
    print(len(y))