"""LeRobot ACT with intent tokens (docs/requirements/INTENT_ACT_GUIDE.md §2.2-2.3, §2.6).

The guide's changes to the original ACT (detr_vae.py / transformer.py) map onto LeRobot's `ACT` as follows:
  * transformer encoder input  [latent, robot_state, (env_state), *intent_tokens, *image_tokens]
    (`encoder_1d_feature_pos_embed` grows from n_1d to n_1d + N learned position embeddings),
  * CVAE encoder input         [cls, robot_state, *intent_tokens, *action_sequence] when `in_cvae`
    (fixed sinusoidal table 1 + 1 + N + chunk; the intent positions are never padding),
  * optional FiLM of every camera's backbone feature map by the intent tokens (`film`).
`intent_cfg=None` builds exactly LeRobot's ACT (same modules, same parameter initialisation, same RNG use) and
`forward` is then LeRobot's own `ACT.forward`.

The intent enters through the batch: `batch['intent']` = {obj, act, tau (B, 2), xi (B, M, J, D) normalised} and
optionally `batch['intent_keep']` = {component: BoolTensor (B,)} (None: every component kept).
"""
import einops
import torch
from torch import Tensor, nn
from lerobot.policies.act.modeling_act import ACT, create_sinusoidal_pos_embedding
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from intent_policy.policies.intent_encoder import ALL_COMPONENTS, IntentEncoder, IntentFiLM


class IntentACT(ACT):
    def __init__(self, config, intent_cfg: dict | None = None):
        super().__init__(config)
        self.use_intent = intent_cfg is not None
        self.n_extra = 0
        if not self.use_intent:
            return
        d = config.dim_model
        self.intent_encoder = IntentEncoder(d, intent_cfg['n_obj'], intent_cfg['n_act'], intent_cfg['n_waypoints'],
                                            intent_cfg['n_joints'], intent_cfg.get('kp_dim', 3),
                                            components=tuple(intent_cfg.get('components', ALL_COMPONENTS)),
                                            n_fourier=intent_cfg.get('n_fourier', 6), tte_max=intent_cfg.get('tte_max', 3.0))
        self.n_extra = self.intent_encoder.n_tokens
        self.intent_in_cvae = bool(intent_cfg.get('in_cvae', True)) and config.use_vae
        n_1d = 1 + bool(config.robot_state_feature) + bool(config.env_state_feature)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d + self.n_extra, d)          # LeRobot: Embedding(n_1d, d)
        if self.intent_in_cvae:
            n_vae = 1 + bool(config.robot_state_feature) + self.n_extra + config.chunk_size
            self.register_buffer('vae_encoder_pos_enc', create_sinusoidal_pos_embedding(n_vae, d).unsqueeze(0))
        self.intent_film = IntentFiLM(d, self.encoder_img_feat_input_proj.in_channels) \
            if intent_cfg.get('film') and config.image_features else None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if not self.use_intent:
            return super().forward(batch)
        if 'intent' not in batch:
            raise KeyError("intent tokens are enabled but batch['intent'] is missing")
        if self.config.use_vae and self.training:
            assert ACTION in batch, 'actions must be provided when using the variational objective in training mode.'

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        intent_tok = self.intent_encoder(batch['intent'], batch.get('intent_keep'))           # (B, N, D)

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, '1 d -> b 1 d', b=batch_size)
            parts = [cls_embed]
            if self.config.robot_state_feature:
                parts.append(self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
            if self.intent_in_cvae:
                parts.append(intent_tok)
            parts.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_encoder_input = torch.cat(parts, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            n_prefix = 1 + bool(self.config.robot_state_feature) + (self.n_extra if self.intent_in_cvae else 0)
            prefix_is_pad = torch.full((batch_size, n_prefix), False, device=batch[OBS_STATE].device)
            key_padding_mask = torch.cat([prefix_is_pad, batch['action_is_pad']], axis=1)
            cls_token_out = self.vae_encoder(vae_encoder_input.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2),
                                             key_padding_mask=key_padding_mask)[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(batch[OBS_STATE].device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        encoder_in_tokens.extend(list(intent_tok.permute(1, 0, 2)))                          # N x (B, D)

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)['feature_map']
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                if self.intent_film is not None:
                    cam_features = self.intent_film(cam_features, intent_tok)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, 'b c h w -> (h w) b c')
                cam_pos_embed = einops.rearrange(cam_pos_embed, 'b c h w -> (h w) b c')
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros((self.config.chunk_size, batch_size, self.config.dim_model),
                                 dtype=encoder_in_pos_embed.dtype, device=encoder_in_pos_embed.device)
        decoder_out = self.decoder(decoder_in, encoder_out, encoder_pos_embed=encoder_in_pos_embed,
                                   decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1))
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)
