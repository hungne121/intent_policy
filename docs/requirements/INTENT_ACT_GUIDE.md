# INTENT_ACT_GUIDE — Thêm intent vào ACT (phạm vi: định nghĩa intent → chèn vào ACT → demo)

> Tài liệu dành cho **coding agent** (Claude Code). Đặt ở gốc repo `tonyzhaozh/act` và thêm vào `CLAUDE.md`:
> `Đọc và làm theo INTENT_ACT_GUIDE.md. Làm lần lượt M0 → M3, dừng sau mỗi milestone để người dùng xác nhận.`

**Phạm vi:** chỉ gồm (1) schema intent và nhãn, (2) sửa kiến trúc ACT để nhận intent, (3) demo chứng minh intent đi vào được policy.
**Ngoài phạm vi:** các biến thể thí nghiệm (shuffled, aux, mask), đánh giá thống kê, predictor. Không triển khai các phần này.

---

## 0. Quy tắc cho agent

1. **Đọc code thật trước khi sửa.** Các đoạn code gốc ACT trích trong tài liệu này viết theo trí nhớ. Hãy `view` file, tìm đúng vị trí, không đoán số dòng.
2. **Không phá baseline.** Khi `--use_intent` tắt, model phải giống hệt ACT gốc: cùng `state_dict` keys, cùng output với cùng seed.
3. **Không đổi hyperparameter gốc.** Mọi tham số mới là flag có default.
4. Mỗi milestone kết thúc bằng test xanh và một tóm tắt ngắn các thay đổi.
5. Chỗ cần người dùng quyết định (`TODO(user)`): tạo stub và báo lại, không tự bịa.

### Bốn bộ dữ liệu

| Bộ | Dữ liệu | Dùng intent |
|---|---|---|
| A | Episode "phản hồi khi intent rõ" | Không |
| B | Episode "phản hồi sớm" | Không |
| C | **Cùng episode với A**, cộng thêm nhãn `/intent/*` | Có |
| D | **Cùng episode với B**, cộng thêm nhãn `/intent/*` | Có |

Chỉ cần thu 2 bộ episode. Nhãn intent được **thêm vào cùng file HDF5**. A/B train với `--use_intent` tắt (bỏ qua key `/intent/*`), C/D train với flag bật.

### File mới / sửa
```
act/
├── configs/intent_schema.yaml       # NEW
├── tools/build_intent_labels.py     # NEW
├── tools/demo_intent.py             # NEW
├── detr/models/intent_encoder.py    # NEW
├── detr/models/detr_vae.py          # MODIFY
├── detr/models/transformer.py       # MODIFY
├── policy.py                        # MODIFY
├── utils.py                         # MODIFY
├── imitate_episodes.py              # MODIFY (flags)
└── tests/                           # NEW
```

---

## M0 — Đọc repo và chạy baseline

- Chạy train/infer ACT gốc trên 1–2 episode của bộ A trong vài epoch.
- Ghi lại vị trí thực của: `DETRVAE.__init__`/`forward`, `build()` trong `detr_vae.py`; `Transformer.forward` trong `transformer.py`; `ACTPolicy.__call__` trong `policy.py`; `EpisodicDataset`, `get_norm_stats`, `load_data` trong `utils.py`.
- **Xong khi:** baseline chạy và có danh sách vị trí cần sửa.

---

## M1 — Định nghĩa intent và gán nhãn

### 1.1. Bốn thành phần intent

| Ký hiệu | Câu hỏi | Mỗi timestep |
|---|---|---|
| `obj` | What-target: người hướng tới vật/vùng nào | int, có lớp `none` |
| `act` | What-action: **người** định làm gì | int, có lớp `none` |
| `tau` | When: còn bao lâu tới sự kiện | `[phase ∈ [0,1], tte (giây)]` |
| `xi` | How: quỹ đạo tay tương lai | `(M, J, 3)`, tương đối so với tay hiện tại |

`act` mô tả hành động của người, **không bao giờ** là hành động robot (tránh rò rỉ đáp án).

### 1.2. `configs/intent_schema.yaml`
```yaml
# TODO(user): điền vocab thật
obj_vocab: [none, cup, box, region_left, region_right, robot, workspace]
act_vocab: [none, take, give_to_robot, receive_from_robot, place, point_command, pass_by, reach_into, approach]
n_waypoints: 8                 # M
horizon_s: 1.2                 # H cho xi
keypoints: [r_wrist, r_index_tip, l_wrist, l_index_tip]   # J
keypoint_dim: 3                # TODO(user): 3 nếu có depth, 2 nếu chỉ có ảnh (toạ độ ảnh chuẩn hóa)
tte_max_s: 3.0
fps: 50                        # TODO(user)
```
`none` là nhãn dữ liệu, nghĩa là người chưa có ý định. Nó khác với `UNKNOWN`, là token dropout do model tự thêm ở index `N`.

### 1.3. Key thêm vào mỗi `episode_*.hdf5`
```
/intent/obj        (T,)          int64
/intent/act        (T,)          int64
/intent/phase      (T,)          float32
/intent/tte        (T,)          float32
/intent/xi         (T, M, J, D)  float32     # D = keypoint_dim, chưa chuẩn hóa
/human/keypoints   (T, J, D)     float32
attrs["intent_segments"] = JSON [{"t_onset","t_clear","t_event","obj","act"}, ...]
```

### 1.4. `tools/build_intent_labels.py`
- **Input:** thư mục HDF5 và `segments.json` = `{episode_id: [{"t_onset","t_clear","t_event","obj","act"}]}` (đơn vị: index timestep). File này do người gán nhãn xuất từ ELAN/CVAT/Label Studio.
- **Keypoints:** nếu HDF5 chưa có `/human/keypoints`, chạy MediaPipe Hands trên camera chỉ định (`--kp_camera`). Nếu `keypoint_dim=3` thì nâng lên 3D bằng depth và extrinsic (`TODO(user)`).

```python
def per_timestep_labels(T, segments, obj_vocab, act_vocab):
    """'none' ngoài mọi segment; nhãn segment trong [t_onset, t_event]."""

def phase_tte(T, segments, fps, tte_max_s):
    """Trong segment: phase = clip((t-t_onset)/(t_event-t_onset), 0, 1), tte = (t_event-t)/fps.
    Ngoài segment: phase = 0, tte = tte_max_s. Sau t_event (đến segment kế): phase = 1, tte = 0."""

def future_waypoints(kp, fps, horizon_s, M):
    """Làm mượt kp bằng Savitzky–Golay (window 7, order 2).
    xi[t, m] = kp[min(t + k_m, T-1)] - kp[t], k_m = round((m+1)/M * horizon_s * fps)."""
```
Script ghi vào HDF5 (`r+`) và in thống kê: phân bố nhãn, độ dài segment, `t_clear - t_onset`.

### 1.5. Dataset (`utils.py`)
- `get_norm_stats`: thêm `xi_mean`, `xi_std` tính trên train, với `std` clip tối thiểu `1e-2`.
- `EpisodicDataset(..., use_intent=False)`: khi bật, trả thêm phần tử thứ 5:
```python
intent = {
    "obj": torch.tensor(f["/intent/obj"][start_ts]),
    "act": torch.tensor(f["/intent/act"][start_ts]),
    "tau": torch.tensor([f["/intent/phase"][start_ts], f["/intent/tte"][start_ts]]),
    "xi":  (torch.tensor(f["/intent/xi"][start_ts]) - xi_mean) / xi_std,
}
```
Cập nhật mọi chỗ unpack batch trong `imitate_episodes.py` để xử lý được cả 4 và 5 phần tử.

### Test M1 (`tests/test_labels.py`)
Dùng episode giả có 1 segment, kiểm tra: ngoài segment là `none`; `phase` đơn điệu và bằng 1 tại `t_event`; `xi` ở cuối episode không bị index out-of-range; dataloader trả đúng shape.

---

## M2 — Chèn intent vào ACT

### 2.1. `detr/models/intent_encoder.py` (file mới)
```python
import math
import torch
import torch.nn as nn

ALL_COMPONENTS = ("obj", "act", "tau", "xi")
TYPE_ID = {"obj": 0, "act": 1, "tau": 2, "xi": 3}


class IntentEncoder(nn.Module):
    """intent dict -> token (B, N, d).
    keep[c] (BoolTensor B): False -> token UNKNOWN/null học được của thành phần c.
    obj/act nhận LongTensor (nhãn cứng) hoặc FloatTensor (B, n_cls) xác suất mềm."""

    def __init__(self, d, n_obj, n_act, n_waypoints, n_joints, kp_dim=3,
                 components=ALL_COMPONENTS, n_fourier=6, tte_max=3.0):
        super().__init__()
        self.components = tuple(c for c in ALL_COMPONENTS if c in components)
        self.n_obj, self.n_act, self.M = n_obj, n_act, n_waypoints
        self.n_fourier, self.tte_max = n_fourier, tte_max
        self.obj_emb = nn.Embedding(n_obj + 1, d)          # index n_obj = UNKNOWN
        self.act_emb = nn.Embedding(n_act + 1, d)          # index n_act = UNKNOWN
        self.tau_mlp = nn.Sequential(nn.Linear(2 + 2 * n_fourier, d), nn.GELU(), nn.Linear(d, d))
        self.null_tau = nn.Parameter(torch.randn(1, d) * 0.02)
        self.xi_proj = nn.Linear(n_joints * kp_dim, d)
        self.xi_time = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.null_xi = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.type_emb = nn.Embedding(4, d)

    @property
    def n_tokens(self):
        return sum(self.M if c == "xi" else 1 for c in self.components)

    @staticmethod
    def _label_tok(x, emb, n_cls, keep):
        tok = emb(x.long()) if not x.is_floating_point() else x @ emb.weight[:n_cls]
        return torch.where(keep[:, None], tok, emb.weight[n_cls].expand_as(tok))

    def _tau_feat(self, tau):
        phase, tte = tau[:, :1], tau[:, 1:].clamp(0, self.tte_max)
        k = torch.arange(self.n_fourier, device=tau.device, dtype=tau.dtype)
        ang = (2.0 ** k) * math.pi * phase
        return torch.cat([phase, torch.log1p(tte), ang.sin(), ang.cos()], dim=-1)

    def sample_keep(self, B, device, p_drop, p_drop_all):
        keep = {c: torch.rand(B, device=device) >= p_drop for c in self.components}
        drop_all = torch.rand(B, device=device) < p_drop_all
        return {c: k & ~drop_all for c, k in keep.items()}

    def forward(self, intent, keep=None):
        B = intent[self.components[0] if self.components[0] in intent else "obj"].shape[0]
        dev = self.type_emb.weight.device
        if keep is None:
            keep = {c: torch.ones(B, dtype=torch.bool, device=dev) for c in self.components}
        toks = []
        for c in self.components:
            if c == "obj":
                t = self._label_tok(intent["obj"], self.obj_emb, self.n_obj, keep["obj"])[:, None]
            elif c == "act":
                t = self._label_tok(intent["act"], self.act_emb, self.n_act, keep["act"])[:, None]
            elif c == "tau":
                t = self.tau_mlp(self._tau_feat(intent["tau"]))
                t = torch.where(keep["tau"][:, None], t, self.null_tau.expand(B, -1))[:, None]
            else:
                t = self.xi_proj(intent["xi"].flatten(2)) + self.xi_time
                t = torch.where(keep["xi"][:, None, None], t, self.null_xi.expand(B, -1, -1))
            toks.append(t + self.type_emb.weight[TYPE_ID[c]])
        return torch.cat(toks, dim=1)
```

### 2.2. `detr/models/transformer.py`
Trong `Transformer.forward`, nhánh `len(src.shape) == 4`, ACT gốc ghép `addition_input = torch.stack([latent_input, proprio_input], axis=0)` vào trước `src` (cần xác minh). Sửa:
```python
def forward(self, src, mask, query_embed, pos_embed, latent_input=None,
            proprio_input=None, additional_pos_embed=None, extra_input=None):
    ...
    addition_input = torch.stack([latent_input, proprio_input], axis=0)      # (2,B,d)
    if extra_input is not None:                                               # (N,B,d)
        addition_input = torch.cat([addition_input, extra_input], axis=0)
    src = torch.cat([addition_input, src], axis=0)
```
`additional_pos_embed` truyền vào có kích thước `(2+N, d)`. Nếu dùng key-padding `mask`, thêm `N` cột `False`.

### 2.3. `detr/models/detr_vae.py`
**`__init__`** nhận thêm `intent_cfg=None`:
```python
self.use_intent = intent_cfg is not None
self.n_extra = 0
if self.use_intent:
    self.intent_encoder = IntentEncoder(hidden_dim, intent_cfg["n_obj"], intent_cfg["n_act"],
        intent_cfg["n_waypoints"], intent_cfg["n_joints"], intent_cfg["kp_dim"],
        components=intent_cfg["components"])
    self.n_extra = self.intent_encoder.n_tokens
    self.intent_in_cvae = intent_cfg.get("in_cvae", True)
self.additional_pos_embed = nn.Embedding(2 + self.n_extra, hidden_dim)   # gốc: Embedding(2, ...)
n_cvae = self.n_extra if (self.use_intent and self.intent_in_cvae) else 0
self.register_buffer("pos_table",
    get_sinusoid_encoding_table(1 + 1 + n_cvae + num_queries, hidden_dim))
```
Khi `intent_cfg=None`, mọi kích thước phải giữ nguyên như gốc.

**`forward(self, qpos, image, env_state, actions=None, is_pad=None, intent=None, intent_keep=None)`:**
```python
intent_tok = self.intent_encoder(intent, intent_keep) if self.use_intent else None   # (B,N,d)

# CVAE encoder (training): chèn intent giữa qpos và action
parts = [cls_embed, qpos_embed]
if self.use_intent and self.intent_in_cvae:
    parts.append(intent_tok)
parts.append(action_embed)
encoder_input = torch.cat(parts, axis=1)
n_prefix = 2 + (self.n_extra if (self.use_intent and self.intent_in_cvae) else 0)
cls_joint_is_pad = torch.full((bs, n_prefix), False, device=qpos.device)   # gốc: (bs, 2)
is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

# Decoder
extra = intent_tok.permute(1, 0, 2) if self.use_intent else None
hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                      self.additional_pos_embed.weight, extra_input=extra)[0]
```
Giữ nguyên quirk `[0]` của ACT gốc. Trong `build(args)`: đọc `args.intent_cfg` và truyền vào.

Lý do đưa intent vào cả CVAE encoder: nếu không, latent `z` có thể mã hóa intent lấy từ chuỗi hành động demo, trong khi lúc test `z=0` nên thông tin đó mất. Có cờ `in_cvae` để tắt khi cần so sánh.

### 2.4. `policy.py` (`ACTPolicy`)
```python
def __call__(self, qpos, image, actions=None, is_pad=None, intent=None, intent_keep=None):
    image = normalize(image)
    if actions is not None:
        if self.model.use_intent and intent_keep is None:
            intent_keep = self.model.intent_encoder.sample_keep(
                qpos.shape[0], qpos.device, self.p_drop, self.p_drop_all)
        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, None, actions, is_pad,
                                                     intent, intent_keep)
        # loss giữ nguyên như ACT gốc (L1 + kl_weight * KL)
    else:
        a_hat, _, _ = self.model(qpos, image, None, intent=intent, intent_keep=intent_keep)
        return a_hat
```
Component dropout (`p_drop`, `p_drop_all`) chỉ áp dụng khi train. Khi infer, mặc định giữ toàn bộ intent.

### 2.5. Head hành động rời rạc (tùy chọn)
Flag `--action_mode {continuous, discrete}`, default `continuous`. Với `discrete`:
- `action_head = nn.Linear(hidden_dim, 9)`;
- CVAE encoder dùng `nn.Embedding(9, hidden_dim)` cho action;
- loss là cross-entropy có mask `is_pad`;
- dataset đọc `/action_discrete` (`TODO(user)` nếu chưa có key này).

### 2.6. FiLM fallback (chỉ dùng khi demo M3 cho thấy intent bị bỏ qua)
```python
class IntentFiLM(nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.to_gb = nn.Linear(d, 2 * C)
        nn.init.zeros_(self.to_gb.weight); nn.init.zeros_(self.to_gb.bias)   # khởi tạo identity
    def forward(self, feat, intent_tok):          # feat (B,C,H,W)
        g, b = self.to_gb(intent_tok.mean(1)).chunk(2, dim=-1)
        return feat * (1 + g[..., None, None]) + b[..., None, None]
```
Áp FiLM lên feature backbone của từng camera, trước `input_proj`. Bật bằng flag `--intent_film`.

### 2.7. Flags trong `imitate_episodes.py`
```
--use_intent          (default False)
--intent_components   "obj,act,tau,xi"
--intent_in_cvae      (default True)
--p_drop 0.2   --p_drop_all 0.1
--action_mode continuous|discrete
--intent_film         (default False)
```
Khi bật `--use_intent`: đọc `intent_schema.yaml` để tạo `intent_cfg`, rồi đưa vào `policy_config`.

### Test M2 (`tests/test_model.py`)
1. `intent_cfg=None`: `state_dict` keys và shapes trùng model gốc, cùng seed thì output trùng.
2. Mọi tổ hợp `components`: `n_tokens` đúng, train/infer chạy, loss hữu hạn.
3. `keep[c]=False` thì token bằng UNKNOWN/null cộng `type_emb`.
4. Model khởi tạo ngẫu nhiên: đổi `act` làm output thay đổi.
5. One-hot dạng float cho cùng token như nhãn int.
6. `loss.backward()` cho gradient khác 0 ở `intent_encoder.*`.

---

## M3 — Demo

### 3.1. Train demo
Train **D** (đủ 4 thành phần) trên bộ D. Nếu muốn nhanh thì giảm epoch. Đồng thời train **B** với cùng cấu hình, `--use_intent` tắt, để đối chiếu.

### 3.2. `tools/demo_intent.py`
```
python tools/demo_intent.py --ckpt runs/D/policy_last.ckpt --episode <path> \
    --swap act --to give_to_robot --out demo/
```
Với một episode test, chạy inference (`z=0`) tại mọi timestep trong 4 chế độ:
1. intent gốc;
2. swap một thành phần (`--swap {obj,act,tau,xi}`, `--to <giá trị>`; với `tau`/`xi` thì lấy từ episode khác qua `--from_episode`);
3. drop thành phần đó (`keep[c]=False`);
4. drop toàn bộ intent.

Output gồm:
- `demo/actions.png`: hành động bước đầu của chunk theo thời gian, 4 đường. Vạch dọc tại `t_onset`, `t_clear`, `t_event`.
- `demo/summary.txt`: tỉ lệ timestep mà hành động thay đổi giữa (1) và (2), tách riêng `t < t_clear` và `t ≥ t_clear`. Với `discrete`: argmax khác nhau. Với `continuous`: L1 > ngưỡng `--eps`.
- (tùy chọn) `demo/video.mp4`: ảnh camera kèm nhãn intent và hành động dự đoán.

### 3.3. Kết quả mong đợi
- Khi swap `act` hoặc `obj`, hành động **phải đổi**, nhất là ở đoạn `t < t_clear`. Nếu gần như không đổi, intent đang bị bỏ qua: bật `--intent_film` và train lại, rồi báo người dùng.
- Khi drop toàn bộ intent, hành vi nên gần với model B.

**Xong khi:** có `actions.png` và `summary.txt` cho ít nhất 3 episode (mỗi scenario 1 episode), kèm nhận xét ngắn của agent.

---

## Lưu ý
- Oracle `tau`/`xi` chứa thông tin tương lai, nên chỉ dùng cho demo offline hoặc khi chuyển động người được phát lại cố định.
- Mọi model so sánh (B và D) phải dùng cùng `chunk_size`, số epoch và quy tắc chọn checkpoint.
- Không gộp `none` (nhãn dữ liệu) với `UNKNOWN` (token dropout).
