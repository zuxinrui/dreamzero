# CLAUDE.md — DreamZero Quick Reference

NVIDIA GEAR Lab 的 **World Action Model**:用 Wan 视频扩散模型 + action/state register 改造成机器人策略,同步预测视频和 24 步动作 chunk,flow matching 训练,自回归块扩散推理。完整技术细节见 `docs/2026-04-21_ALGORITHM_AND_DEPLOYMENT.md`。

---

## 架构一句话

**Identity backbone + Action Head (含完整 Wan DiT)**。"backbone" 目录是 dummy,真正的模型全在 `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`。

子模块:VAE (16 或 48 ch) + umt5-xxl 文本编码器 + open_clip ViT-H/14 图像编码器 + CausalWanModel DiT + FlowMatchScheduler。DiT 序列 = `[video_tokens | action_register]`,3D RoPE for video,1D RoPE for action/state,blockwise causal mask。

一块视频 (2 帧) ↔ 一个 action chunk (24 步),控制 8 维 (7 关节 + 1 夹爪)。

## 两个骨架规格

| | Wan 2.1 (14B) | Wan 2.2 (5B) |
|---|---|---|
| DiT dim / layers / heads | 5120 / 40 / 40 | 3072 / 30 / 24 |
| in_dim / out_dim | 36 / 16 | 48 / 48 |
| VAE | 16 ch, 8× | 48 ch, 16× |
| 分辨率 | 320×176 | 320×160 |
| 官方 checkpoint | ✅ DreamZero-DROID / -AgiBot | ❌ 仅训练配置 |

## 训练

- **Objective**: Flow matching,target = x_0 − ε,rectified flow 线性加噪
- **Loss**: `dynamics_loss (MSE video)` + `action_loss (MSE action, masked)`,均按 `w(t)` 加权
- **噪声采样**: STANDARD (uniform) / HIGH_NOISE_EMPHASIS (Beta(α,1)) / DECOUPLED (video Beta + action uniform)
- **Config**: bf16 + tf32 + grad checkpoint + DeepSpeed ZeRO-2,lr 1e-5,wd 1e-5,warmup 5%,per-device bs=1
- **Data**: DROID LeRobot ~131GB,num_frames=33,action_horizon=24,3 camera views
- **脚本**: `scripts/train/droid_training_{lora,full_finetune_wan21,full_finetune_wan22}.sh`;迁移用 `{agibot,yam}_training.sh`

## 推理

- **循环**: 自回归块扩散 + KV cache (每块新 token 才算)
- **步数**: `num_inference_steps=16`,`NUM_DIT_STEPS` env 可 5/6/7/8 (skip 用 `cache_predict_order1` Adams 外推)
- **CFG**: `cfg_scale=5.0`,`ip_size=2` 时两卡各跑一边正/负 prompt 然后 P2P 交换 —— **复制非分片,每卡持有完整模型**
- **Decoupled inference**: 视频 sigma 走到 0.8 停,动作走到 0,省 ~30% 时间
- **延迟**: GB200+TRT FP4 ≈ 0.6s / H100 ≈ 3s / 3090 估 5-10s
- **入口**: `socket_test_optimized_AR.py` (WebSocket server) + `eval_utils/run_sim_eval.py` (Isaac Lab client)

## 微调

- **LoRA (默认)**: rank=4, alpha=4, target `q,k,v,o,ffn.0,ffn.2`,额外训 `action_encoder/decoder/state_encoder`,text/image/VAE 冻结
- **Full FT**: `train_architecture=full`,配 ZeRO-2 + CPU offload
- **迁移**: 从 `DreamZero-AgiBot` LoRA 微调新 embodiment,~30min teleop 数据可得到基础能力

## 仿真与硬件能力

本地仿真 = **Isaac Lab + Isaac Sim (Omniverse)**,不是 MuJoCo/PyBullet。任务 = DROID Franka Panda 桌面 pick-and-place 3 scenes。client-server 架构走 WebSocket,两边可以用不同 conda env(推荐:isaacsim env 跑 Isaac Sim client,dreamzero env 跑 policy server)。

| 硬件 | 能做什么 |
|---|---|
| 双 RTX 3090 (48 GB) | sim client;5B LoRA 微调。**装不下官方 14B checkpoint**(复制非分片,单卡 24GB 不够) |
| H100 | 14B 推理 ~3s;sim-evals 评测 |
| GB200 + TRT FP4 | 14B 推理 ~0.6s;接近真机异步流水线门槛 |

**推荐路径**(想在 3090 上玩):申请官方托管 API,本地只跑 Isaac Sim 仿真器。代码见 `eval_utils/run_sim_eval.py`。

## 真机部署的 staleness 问题(重要)

参考 client `run_sim_eval.py:44` 用 `open_loop_horizon=8` 同步阻塞调用,**仿真可用,真机不丝滑**。核心约束:异步流水线要求 `T_infer ≤ T_exec = k × dt`,否则无用。且即便最新鲜的 chunk 也是 `T_infer` 秒前的观测做出的,staleness 无法靠异步消除。

**业界五招组合应对**(DreamZero 原生只实现了招 5 的一部分):

1. **Temporal Ensembling** (ACT): 对同一物理时刻,加权平均来自多个历史 chunks 的预测,exp(-λ·age) 衰减。~50 行代码加在客户端。**解决边界抖,不解决系统偏**。
2. **Delta Action Space**: 输出 Δq 而非绝对 q_target,对 stale obs 不敏感。DreamZero 默认绝对,可事后 rebase:`q_cmd(t) = q_actual(now) + (q_pred(t) − q_pred(anchor))`。
3. **Inflight Action Conditioning** (π0): 推理时除 obs 外,把"接下来要执行的 inflight actions"也喂给模型,模型内部隐式预测 t+T_infer 的状态。需重训。
4. **Impedance / Compliant Control**: 低层用 `τ = K_p·Δq + K_d·Δq̇ + g(q)` 替代 hard position servo,命令跳变被物理 low-pass 吸收。Franka/AgiBot/UR 都支持。
5. **压缩 T_infer** (π0 哲学): 最暴力最有效。FAST tokenization / 小模型 / INT8-FP4 量化 / TRT。目标把 staleness 压到 50ms 以下。

**DreamZero 改造优先级**(若要真机): ② Ensembling → ③ 异步客户端 → ④ 低层柔顺 → ⑤ 推理压缩 → ⑥ horizon 扩大 → ⑦ 架构级改 delta/inflight。

## 参数速查

| 想改什么 | 位置 |
|---|---|
| 骨架 (14B↔5B) | `model/dreamzero/action_head=wan_flow_matching_action_tf{_wan22}` |
| LoRA | `action_head_cfg.config.lora_{rank,alpha,target_modules}` |
| 推理步数 | `num_inference_steps` (默认 16) |
| CFG 强度 | `cfg_scale` (默认 5.0) |
| DiT skip | env `NUM_DIT_STEPS` ∈ {5,6,7,8} |
| Video 提前停 | `decouple_inference_noise` + `video_inference_final_noise` |
| 单卡/双卡 | `torchrun --nproc_per_node={1,2}` |
| TRT FP4 | env `LOAD_TRT_ENGINE=/path/to/WanModel_nvfp4.trt` (GB200 only) |
| 开环步数 | `run_sim_eval.py:44` `open_loop_horizon` (默认 8) |

## 关键文件速查

| 功能 | 路径 |
|---|---|
| 主 action head (含 DiT 组装、训练、推理) | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` |
| DiT (CausalWanModel) | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| Flow matching scheduler | `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` |
| Policy wrapper (VRAM mgmt, device mesh) | `groot/vla/model/n1_5/sim_policy.py` |
| WebSocket server | `socket_test_optimized_AR.py` |
| Isaac Lab sim client | `eval_utils/run_sim_eval.py` |
| WebSocket client | `eval_utils/policy_client.py` |
| 训练入口 | `groot/vla/experiment/experiment.py` |
| 详细技术手册 | `docs/2026-04-21_ALGORITHM_AND_DEPLOYMENT.md` |
| 5B 架构说明 | `docs/2026-04-19_WAN22_BACKBONE.md` |
| 新 embodiment 指南 | `docs/2026-02-28_DATASET_TO_GEAR_AND_TRAIN.md` |

## 环境与安装

- Python 3.11, PyTorch 2.8 + CUDA 12.9, bf16 默认
- flash-attn 装 prebuilt wheel 别编译: `cu12torch2.8 + cp311 + cxx11abiTRUE`(用 `torch.compiled_with_cxx11_abi()` 确认)
- Isaac Sim 环境推荐独立 conda env,通过 WebSocket 和 dreamzero env 通信 —— **不要合并两个 env**,Isaac Sim 对 torch/cuda 版本强 pin

## 常见坑

- `backbone/` 是 Identity dummy,真代码在 `action_head/`
- 2-GPU 是 CFG 并行,不是模型分片 —— 单卡 VRAM 要能装下整个模型
- 官方 checkpoint 都是 14B,3090 装不下;5B 只有训练配置没发 ckpt
- `run_sim_eval.py` 是同步阻塞 client,不能直接用于真机
- action 是绝对 joint position (DROID 格式),不是 delta

## `eval_utils/run_sim_eval.py` 复用:接 openpi 的 policy(已验证)

这个脚本**不止能跑 DreamZero 自己的 policy,也能直接接 openpi 的 DROID 策略**。dreamzero 14B 装不进双 3090,但 openpi 的 π0.5-DROID(~3B)轻松跑,所以 sim-eval client 复用 + openpi policy server 是本机可跑的**最实用组合**。

### 已验证工作的最小配置

- **policy server**(openpi `.venv`,GPU1): `uv run scripts/serve_policy.py --port 6000 policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`
- **sim client**(env_isaaclab_legacy,**GPU0 接显示器**): `python eval_utils/run_sim_eval.py --episodes 1 --scene 1 --host localhost --port 6000 --no-headless`
- 端口**必须 6000**(run_sim_eval.py 默认);双 3090 开 GUI 必须把 sim 放在接显示器的那张卡
- 完整踩坑手册见 **openpi 仓的 `docs/DROID_ISAAC_SIM_DEPLOYMENT.md`**

### 本地对 `run_sim_eval.py` 的未提交 patch

`eval_utils/run_sim_eval.py:198-202` 有个本地 patch:注释掉 `cv2.imshow("Right Camera", ...)`,因为 env_isaaclab_legacy 里是 `opencv-python-headless`(无 GTK/Qt backend),Isaac Sim 自己的 viewport 已经显示场景,cv2 预览冗余。撤回:`git checkout -- eval_utils/run_sim_eval.py`。

### Docstring 里的路径已过时

`run_sim_eval.py:11-14` 写的 `pi0_fast_droid_jointpos` + `s3://openpi-assets-simeval/...` 是**上游旧命名**。现在正确名字带 `_polaris` 后缀,路径在 `gs://openpi-assets/checkpoints/polaris/`。以 openpi 仓 `src/openpi/training/misc/polaris_config.py` 为准。
- 仿真控制频率(代码里 mediapy fps=15 暗示 15Hz,但实际由 sim env 决定)
