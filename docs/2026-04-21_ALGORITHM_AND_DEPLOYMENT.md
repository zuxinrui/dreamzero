# DreamZero 算法与真机部署技术手册

> 本文覆盖从模型架构、训练算法、推理机制、微调方式,到**从仿真评测基线跨越到真机闭环部署**的所有关键工程技巧。面向要自行扩展、迁移或部署 DreamZero 的研究/工程人员。

---

## 目录

- [Part 1 · 模型架构](#part-1--模型架构)
- [Part 2 · 训练算法](#part-2--训练算法)
- [Part 3 · 推理算法](#part-3--推理算法)
- [Part 4 · 微调方式](#part-4--微调方式)
- [Part 5 · 从仿真到真机的工程鸿沟](#part-5--从仿真到真机的工程鸿沟)
- [Part 6 · 真机部署的五大技巧](#part-6--真机部署的五大技巧)
- [Part 7 · DreamZero 相对 SOTA 的现状与改造建议](#part-7--dreamzero-相对-sota-的现状与改造建议)

---

## Part 1 · 模型架构

### 1.1 I/O 契约

```
输入:
  videos:       [B, T=33, H, W, 3]          多帧 RGB (1 首帧 + 32 后续帧, 3 路相机拼接)
  text:         token ids (umt5-xxl)         自然语言指令
  state:        [B, T_state, state_dim]      机器人本体状态 (joint + gripper)
  action:       [B, horizon=24, action_dim]  GT 动作 (仅训练)
  embodiment_id: int                         本体 ID (DROID=26, YAM=17 等)

输出:
  video_pred:   [B, T, C, H', W']            去噪后视频 latent
  action_pred:  [B, 24, action_dim]          一个 chunk 的动作 (8 维: 7 关节 + 1 夹爪)
```

### 1.2 子模块清单(`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:174-262`)

| 模块 | Wan 2.1 (14B) | Wan 2.2 (5B) | 作用 |
|---|---|---|---|
| **VAE** `WanVideoVAE` / `WanVideoVAE38` | 16 ch, 8× 下采样 | 48 ch, 16× 下采样 | 视频 ↔ latent |
| **Text encoder** umt5-xxl | ~11B, 冻结 | 同 | 指令 → prompt embedding |
| **Image encoder** open_clip ViT-H/14 | ~1B, 冻结 | 同 | 首帧 → CLIP feature |
| **DiT** `CausalWanModel` | dim=5120, 40 L, 40 H, in_dim=36 | dim=3072, 30 L, 24 H, in_dim=48 | **核心去噪 + 动作预测** |
| **Scheduler** `FlowMatchScheduler` / `FlowUniPCMultistepScheduler` | — | — | 训练加噪 / 推理采样 |

### 1.3 关键创新 —— Action/State Register

DiT 看到的 token 序列:

```
[video_tokens  |  action_register]
 ─── 3D RoPE ───    ─── 1D RoPE ───
```

- `video_tokens`:每帧经 patch embedding (stride 1×2×2) 变成 tokens
- `action_register`:每个 image block 对应 `num_action_per_block=24` 个 action token + `num_state_per_block=1` 个 state token
- **Blockwise causal mask**:第 i 块视频只能 attend 到第 1..i 块(含首帧);action/state 只能看见过去的视频 + 自己块内

DiT 同步输出两路 noise:
- `video_noise_pred` → 视频 latent 去噪(dynamics loss)
- `action_noise_pred` → `action_decoder` 从 register slice 解码出动作去噪(action loss)

**关键参数**:`num_frame_per_block=2`, `num_action_per_block=24`, `num_state_per_block=1` → **一块视频 (2 帧) ↔ 一个 action chunk (24 步动作)**。

### 1.4 "Backbone" 其实是 Identity

源码 `backbone/identity.py` 是个返回零长度 tensor 的 dummy。**真正的 WAN 扩散模型塞在 action_head 内部**。整个系统本质就是一个 action-conditioned diffusion model,不要被"backbone"命名骗了。

---

## Part 2 · 训练算法

### 2.1 损失函数(`wan_flow_matching_action_tf.py:760-803`)

```
loss = dynamics_loss + action_loss

dynamics_loss = E_t[ w(t) · ‖ε̂_video(x_t, t) − target_video(x_0, ε, t)‖² ]
action_loss   = E_t[ w(t) · ‖ε̂_action(a_t, t) − target_action(a_0, ε_a, t)‖²
                       · action_mask · has_real_action ]
```

- **Flow matching**(非 DDPM):target = `x_0 − ε`(velocity),`FlowMatchScheduler.training_target` 计算
- **加噪**:`x_t = scheduler.add_noise(x_0, ε, t)` — rectified flow 线性插值
- **w(t)**:`scheduler.training_weight(t)` —— 高噪声 timestep 赋更大权重
- **action_mask**:对齐 padding;**has_real_action**:无 GT action 的 sample 只训 video

### 2.2 三种噪声采样模式

| 模式 | Video timestep | Action timestep | 用途 |
|---|---|---|---|
| `STANDARD` | Uniform | 与 video 耦合 | 标准训练 |
| `HIGH_NOISE_EMPHASIS` | Beta(α, 1) 偏高噪 | 与 video 耦合 | 强化高噪声去噪能力 |
| `DECOUPLED` | Beta(3, 1) 偏高噪 | **独立** Uniform | **训练-推理对齐**:推理时视频可"半噪",动作全去噪 |

Decoupled 模式匹配推理端的 decoupled inference(视频 sigma 走到 0.8 停,动作 sigma 走到 0)。

### 2.3 训练超参(来自 `scripts/train/droid_training_full_finetune_wan22.sh`)

| 参数 | 默认 |
|---|---|
| Optimizer | DeepSpeed ZeRO-2 (optional CPU offload) |
| Precision | bf16 + tf32 |
| Gradient checkpointing | on |
| Learning rate | 1e-5 |
| Warmup ratio | 0.05 |
| Weight decay | 1e-5 |
| Max steps | 200 000 (WAN2.2 full FT) |
| Per-device batch | 1 |
| Global batch | NUM_GPUS × 1 (可 grad-accum 扩) |
| num_frames | 33 |
| Resolution | 320×160 (5B) / 320×176 (14B) |
| action_horizon | 24 |
| views | 3 (ext_1, ext_2, wrist) |
| Dataset | DROID LeRobot ~131 GB |

### 2.4 训练路线图

```
从头训 (from-scratch):
  Wan2.1-I2V-14B + umt5 + CLIP ──► DreamZero-DROID ckpt
  (action_encoder/decoder/state_encoder 随机初始化)

发布的两个起点:
  DreamZero-DROID    →  DROID 基线 SOTA
  DreamZero-AgiBot   →  通用 pretrain, 用于新 embodiment 微调
```

---

## Part 3 · 推理算法

### 3.1 自回归块扩散主循环

```
for observation in robot_loop:
    if language_changed or first_call:
        reset_kv_cache();  current_start_frame = 0

    for step in schedule:               # num_inference_steps = 16
        if should_run_dit(step):         # dit_step_mask 控制
            pred_cond, pred_uncond = DiT(...)          # CFG 两路 forward
            pred = pred_uncond + cfg_scale·(pred_cond - pred_uncond)
        else:
            pred = cache_predict_order1(...)           # Adams 一阶外推

        noisy_video  = scheduler_video.step(pred.video, ...)    # 可 sigma=0.8 停
        noisy_action = scheduler_action.step(pred.action, ...)   # sigma=0.0

    current_start_frame += num_frame_per_block
    return action_chunk                                 # 24 步
```

### 3.2 三级加速栈

| 手段 | 来源 | 硬件 | 作用 |
|---|---|---|---|
| **KV cache over blocks** | `_create_kv_caches` | 所有 | 新块只算新 token,旧 K/V 复用 |
| **DiT step skipping** | `dit_step_mask` + `should_run_model` | 所有 | 16 步去噪只真跑 8 次 DiT |
| **2-GPU CFG split** | `parallelize` + P2POp | 2× GPU | 正/负 prompt 分卡并行 |
| **TensorRT FP4** | `trt_engine` (opt-in) | GB200 only | Blackwell 专属 FP4 kernel |

**端到端延迟**(README.md:142):GB200 + TRT FP4 ≈ **0.6 s**,H100 ≈ **3 s**,双 3090 估计 **5–10 s**。

### 3.3 关于"两张 GPU"的真相 —— 是 CFG 并行,不是模型并行

`wan_flow_matching_action_tf.py:1386-1393`:

```python
def parallelize(self, device_mesh: DeviceMesh) -> None:
    ip_mesh = device_mesh["ip"]
    self.ip_rank = ip_mesh.get_local_rank()
    self.ip_size = ip_mesh.size()
    assert self.ip_size == 1 or self.ip_size == 2
```

- `ip_size == 2`:rank 0 跑正 prompt,rank 1 跑负 prompt,P2P 交换 noise prediction 后 CFG 合成
- `ip_size == 1`:单卡串行跑两次 forward,效果相同,只是慢 2×
- **每张 GPU 持有完整模型副本(复制不分片)**,所以单卡 VRAM 必须装得下整个模型

### 3.4 Decoupled Inference(可选优化)

```
video sigma:  1.0 → 0.9 → 0.8 (stop early, 还是噪声)
action sigma: 1.0 → ... → 0.0 (完全去噪)
```

视频只是 auxiliary conditioning,提前停省 ~30% 推理时间。

---

## Part 4 · 微调方式

### 4.1 LoRA 模式(默认,`train_architecture="lora"`)

```python
lora_rank = 4
lora_alpha = 4
lora_target_modules = "q,k,v,o,ffn.0,ffn.2"  # 注意力 QKV/O + FFN 两线性层
init_lora_weights = "kaiming"
```

**可训参数**:
- LoRA adapters(< 1% 主干参数)
- `action_encoder`, `action_decoder`, `state_encoder`(全量训)

**冻结**:DiT 主干(除 LoRA)、text_encoder、image_encoder、VAE。

### 4.2 全量微调(`train_architecture="full"`, `save_lora_only=false`)

整个 DiT + action/state 模块都训,text/image/VAE 仍冻结。配合 DeepSpeed ZeRO-2 + CPU offload。VRAM 与优化器状态 ×10+,只在大数据 / 大分布漂移时用。

### 4.3 Embodiment 迁移范式

```
用户新机器人  ──> ~30 min teleoperation 数据  ──> LoRA fine-tune ~1–2 天
                  from DreamZero-AgiBot                → pick-and-place 能跑
```

三个官方脚本:`droid_training_lora.sh` / `agibot_training.sh` / `yam_training.sh`,关键差异:

- `data=dreamzero/{droid|yam|agibot}_relative`
- `max_action_dim`, `max_state_dim` 由数据集 embodiment.json 决定
- `action_loss_embodiment_ids: [26, 17, 32]` —— 只在这些 ID 上算 action loss(其它 embodiment 只训 video,做 co-train)

---

## Part 5 · 从仿真到真机的工程鸿沟

### 5.1 控制回路基本面 —— Receding Horizon Control

```
预测 H 步, 执行 k < H 步, 丢弃 H-k 步, 观察新状态, 再次预测 ...
```

DreamZero 默认配置:

| 名词 | 值 |
|---|---|
| prediction horizon `H` | `action_horizon = 24` |
| execution horizon `k` | `open_loop_horizon = 8` (`run_sim_eval.py:44`) |
| 控制频率 | DROID 标称 15 Hz → 每步 66 ms |
| 整 chunk 执行时长 | 24 × 66 ms ≈ 1.6 s |
| 开环执行段时长 | 8 × 66 ms ≈ **533 ms** |

**为什么 k < H**:预测越远方差越大;长 horizon 训给模型 supervisory signal,推理时丢尾巴。

### 5.2 同步执行 vs 异步流水线

**同步阻塞**(DreamZero 参考 client 的做法,`run_sim_eval.py:73-95`):

```
t=0   observe(0) → INFER 阻塞 (5s) → chunk_0 ready
t=5   EXEC 8 步 (0.53s)
t=5.53 observe → INFER 阻塞 (5s) ...

有效工作时间 = 0.53 / 5.53 ≈ 10%    (仿真 OK, 真机不可接受)
```

**异步流水线**(真机标配):

```
想象并排两条线, 无缝衔接:
  EXEC:   [===chunk_0===][===chunk_1===][===chunk_2===]
  INFER:     [infer_1]     [infer_2]     [infer_3]
```

**异步工作的必要条件**:`T_infer ≤ T_exec = open_loop_horizon × dt`。

**否则**(如 3090 上 5s 推理 + 0.5s 执行):异步根本救不了,甚至可能更糟 —— obs 在机器人未动时观察,推理出的 chunk 和同步没区别。

### 5.3 Observation 时机 —— 关键细节

**Cold start**(第一次):`obs(0) ≡ obs(5)`,因为机器人在 cold start 期间完全没动。这 5s 盲区**无法消除**。

**Steady state**(T_exec ≥ T_infer):

```
t=0     obs(0), INFER_1 启动
t=5     chunk_1 ready, 开始 EXEC_1 (持续 6s, 假设 T_exec=6s)
t=6     obs(6) ← 机器人已经动了 1 秒, 状态有意义
        INFER_2 启动 (5s)
t=11    INFER_2 ready, 无缝接 EXEC_2
```

**observe 最优时机**:`τ = EXEC_end − T_infer + ε`,即"上一块结束前 T_infer 秒"。太早 = obs 无变化;太晚 = chunk 衔接断开。

### 5.4 **根本性 Staleness 问题**(无法通过异步消除)

即便最新鲜的 chunk,**也是基于 `T_infer` 秒前的观测做出的**:

```
机器人真实位置:    q(11)
chunk_2 预测起点:  基于 obs(6) → 模型"以为"机器人还在 q(6) 附近
系统性误差:       Δq = q(11) − q(6) ≈ T_infer 秒运动累积
```

这个 Δq 无法通过异步或 ensembling 完全消除,必须靠 **Part 6** 的五招组合应对。

---

## Part 6 · 真机部署的五大技巧

### 招 1 · Temporal Ensembling(ACT, Zhao et al. 2023)

**做什么**:对每个物理时刻 t,加权平均所有"恰好覆盖 t 时刻"的历史 chunks:

```
a(t) = Σ_k  w(age_k) · a_k[t − t_anchor_k]
        ↑ exp(−λ · age) 指数衰减, λ 典型 0.5–2 /s
```

**能解决**:chunk 边界的阶跃 → 指数平滑过渡。
**不能解决**:所有被 ensemble 的预测都过时,ensemble 后平均起来依然偏。
**意识**:把时域抖动转移到空间系统偏差。

**代码形态**(真机常见):

```python
class TemporalEnsembleBuffer:
    def __init__(self, lam=1.0):
        self.chunks = []  # [(t_ready, t_anchor, dt, actions)]
        self.lam = lam

    def append(self, t_ready, t_anchor, dt, actions):
        self.chunks.append((t_ready, t_anchor, dt, actions))

    def query(self, t_now):
        num, den = 0.0, 0.0
        for t_ready, t_anchor, dt, actions in self.chunks:
            idx = int((t_now - t_anchor) / dt)
            if 0 <= idx < len(actions):
                w = math.exp(-self.lam * (t_now - t_ready))
                num += w * actions[idx]
                den += w
        return num / max(den, 1e-8)
```

### 招 2 · Delta / Relative Action Space

| 类型 | staleness 敏感度 | DreamZero 现状 |
|---|---|---|
| 绝对关节位置 (DROID 格式) | 高 | ✅ 当前使用 |
| 相对增量 (Δq) | 低 | ❌ 未使用 |

**绝对 → 相对 的事后 rebase**(5 行代码):

```
q_cmd(t) = q_actual(t_now) + (q_predicted(t) − q_predicted(t_anchor))
          └──── 当前真实位置 ────┘  └──── 模型预测的相对起点位移 ────┘
```

把绝对动作事后转成相对动作,对 stale obs 问题有奇效。**代价**:相对动作对 drift 敏感,长 horizon 控制需要周期性 rebase。

### 招 3 · 条件于 Inflight Actions(action-conditioned prediction, π0 做法)

推理时不要只喂 `obs(6)` 给模型,还喂 **"在 obs 时刻到 chunk 起效时刻之间即将执行的动作序列"**:

```python
obs_now           = observe(6)           # 当前观察
actions_inflight  = chunk_1[1..6]        # 接下来 5 秒要执行的动作
text              = task_instruction

# 模型内部相当于:
s_future = learned_dynamics(obs_now, actions_inflight)  # 预测 t=11 状态
chunk_2  = policy(s_future, text)                        # 从未来状态开始规划
```

**训练数据**:采样 `(past_obs, inflight_actions, future_gt_actions)`,让模型学会补偿延迟。**最干净的解法**,从根源消除 staleness。

**DreamZero 现状**:state register 只吃当前 state,**不吃 inflight actions**。改造需要改训练数据管道 + DiT 输入。

### 招 4 · 柔顺控制器(Impedance / Admittance)

任何 VLA 输出都不应该接 hard position servo,必接一层 impedance:

```
τ = K_p · (q_target − q_actual) + K_d · (q̇_target − q̇_actual) + g(q)
         ↑ spring                     ↑ damper                   ↑ 重力补偿
```

- `q_target` 跳变 → 扭矩只按 `K_p` 成比例响应
- 物理上等效弹簧阻尼,自带一阶 low-pass
- Franka Panda、AgiBot、UR 等都有 impedance mode

**这招不减少误差,但把误差时域轮廓从阶跃变成一阶 low-pass**。对机械寿命和执行器冲击是决定性的。

### 招 5 · 压缩 T_infer(π0 哲学,最暴力也最有效)

**前四招都是在 staleness 存在的前提下缓解**,这招是**不让 staleness 严重到需要缓解**。

π0 paper:
- Transformer inference ≤ 50 ms
- chunk 小(50–200 ms)
- 15–30 Hz 控制 @ 真机
- **T_infer ≪ T_exec,staleness 只有 50ms**,其它招数都不再关键

手段:
- FAST tokenization:action 离散成 token → Transformer 直接解码
- 小参数量:~3 B
- Speculative decoding:小模型起草 + 大模型验证
- 硬件:H100 / GB200 + TRT FP4

**这才是业界事实上的方向**。DreamZero 的 14B + diffusion 在 GB200 + TRT FP4 上达到 0.6s,已经贴近这一路线。

### 招式组合 · 真机 pipeline 典型结构

```
┌─────────────────────────────────────────────────────────────┐
│  Policy (GPU): 200ms infer @ 4Hz                            │
│  输入: obs + inflight actions (招 3)                         │
│  输出: delta actions (招 2)                                  │
└──────────────┬──────────────────────────────────────────────┘
               │ chunk (24 steps)
               ↓
┌─────────────────────────────────────────────────────────────┐
│  Temporal Ensembling Buffer (CPU): λ=1.0 /s (招 1)           │
│  保留最近 3 个 chunk, 滑窗 weighted avg                      │
└──────────────┬──────────────────────────────────────────────┘
               │ ensembled target
               ↓
┌─────────────────────────────────────────────────────────────┐
│  Quintic Interpolator (RT thread, 1 kHz):                    │
│  把 4Hz 的 target 平滑插值到 1kHz                             │
└──────────────┬──────────────────────────────────────────────┘
               │ q_smooth(t), q̇_smooth(t)
               ↓
┌─────────────────────────────────────────────────────────────┐
│  Joint Impedance Controller (1 kHz) (招 4)                   │
│  τ = K_p·(q_smooth − q_actual) + K_d·(q̇_smooth − q̇_actual)   │
└──────────────┬──────────────────────────────────────────────┘
               │ torques
               ↓
             Robot
```

招 5(低推理延迟)体现在整个 pipeline 的 "200ms inference" 前提里。

---

## Part 7 · DreamZero 相对 SOTA 的现状与改造建议

### 7.1 DreamZero 当前支持 vs 缺失

| 技巧 | DreamZero 现状 | 真机需要 |
|---|---|---|
| KV cache / DiT step skip | ✅ 已实现 | ✅ |
| 2-GPU CFG split | ✅ 已实现 | 可选 |
| TensorRT FP4 | ✅ (GB200 only) | ✅ |
| Temporal Ensembling | ❌ 丢弃旧 chunk | ⚠️ 真机必须 |
| Delta action space | ❌ 绝对 joint pos | ⚠️ 真机强烈建议 |
| Inflight action conditioning | ❌ state register 不吃 inflight | ⚠️ 建议 |
| Impedance / compliant control | ❌ 客户端不管下游 | ⚠️ 真机必须 |
| Quintic interpolator | ❌ 无 | ⚠️ 真机必须 |

**结论**:DreamZero **目前是评测基线实现,不是真机部署框架**。README 也没提真机部署,因为 NVIDIA 这个 release 的目标是:
1. 发布 checkpoint 让别人在 sim-evals / RoboArena 上评测刷榜
2. 发布训练代码让别人在新 embodiment 上 fine-tune
3. **不包含**真机闭环脚手架

### 7.2 真机改造优先级

若要将 DreamZero 接到真机:

1. **先在 sim-evals 跑通同步 client**(现成)
2. **加 Temporal Ensembling**(~50 行代码,改 `run_sim_eval.py.infer()`)
3. **客户端改异步**(policy_client 用 asyncio,非阻塞发送)
4. **低层接柔顺控制器**(确认真机跑 impedance mode)
5. **降推理延迟**:INT8 TensorRT,`NUM_DIT_STEPS` 5/6,目标压到 1-2 s
6. **最后考虑**:`action_horizon` 扩 48 或 64,给异步流水线更多 buffer
7. **架构级改造**(如果愿意重训):action 输出改 delta,或加 inflight action conditioning

### 7.3 本地硬件能做什么

| 硬件 | 能做的 |
|---|---|
| **双 RTX 3090 (48 GB)** | 跑 sim-evals client(Isaac Sim 侧);跑自训的 Wan2.2-5B(LoRA 微调)。**无法跑官方 14B checkpoint**(单卡 VRAM 装不下完整模型,复制非分片) |
| **H100** | 14B 推理 ~3 s;sim-evals 评测;真机同步阻塞可接受 |
| **GB200 + TRT FP4** | 14B 推理 ~0.6 s;接近真机异步流水线门槛 |
| **真机部署** | 必须 **招 1+4** 最少,否则动作轨迹不连续会伤硬件 |

### 7.4 一句话总结

> DreamZero 是一个**为评测和研究设计**的世界-动作模型,它的训练、推理、微调代码都是第一流的,但**真机部署所需的 temporal ensembling / delta action / inflight conditioning / impedance control / 极低推理延迟** 这五件事里,它**原生只做了第五件的一部分**。真机落地需要在它之上自己搭实时控制栈。

---

## 附录 A · 关键文件定位

| 功能 | 文件路径 |
|---|---|
| 主 action head | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` |
| DiT (CausalWanModel) | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| Flow matching scheduler | `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` |
| Policy wrapper | `groot/vla/model/n1_5/sim_policy.py` |
| 推理 WebSocket server | `socket_test_optimized_AR.py` |
| sim-evals 客户端 | `eval_utils/run_sim_eval.py` |
| Policy client (websocket) | `eval_utils/policy_client.py` |
| 训练入口 | `groot/vla/experiment/experiment.py` |
| 训练脚本 (DROID full FT WAN22) | `scripts/train/droid_training_full_finetune_wan22.sh` |
| LoRA 脚本 | `scripts/train/droid_training_lora.sh` |
| 迁移脚本 (AgiBot/YAM) | `scripts/train/{agibot,yam}_training.sh` |
| WAN22 架构文档 | `docs/WAN22_BACKBONE.md` |
| DROID 数据转换 | `docs/DROID_CONVERSION.md` |
| 新 embodiment 指引 | `docs/DATASET_TO_GEAR_AND_TRAIN.md` |

## 附录 B · 参数速查

| 想改什么 | 改哪里 |
|---|---|
| 骨架 (14B ↔ 5B) | `model/dreamzero/action_head=wan_flow_matching_action_tf{_wan22}` |
| LoRA 秩 / 目标层 | `action_head_cfg.config.lora_rank / lora_target_modules` |
| 推理步数 | `num_inference_steps` (默认 16) |
| CFG 强度 | `cfg_scale` (默认 5.0) |
| DiT step skip | `NUM_DIT_STEPS` 环境变量 (5/6/7/8) |
| Video 提前停 | `decouple_inference_noise` + `video_inference_final_noise` |
| 两卡 / 单卡 | `torchrun --nproc_per_node=1 or 2` |
| TensorRT FP4 | `LOAD_TRT_ENGINE=/path/to/WanModel_nvfp4.trt`(GB200 only) |
| 开环步数 | `run_sim_eval.py:44` `open_loop_horizon = 8` |

## 附录 C · 参考文献与对标工作

- **ACT**: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023 — Temporal Ensembling 原论文
- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023 — receding horizon + diffusion 范式
- **π0**: Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", 2024 — FAST tokenization + action-conditioned 低延迟 VLA
- **OpenVLA**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024 — VLA 开源基线
- **DROID**: Khazatsky et al., "DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset", RSS 2024 — 训练数据来源
- **WAN 2.1 / 2.2**: Alibaba 视频生成模型,DreamZero 的骨架来源
