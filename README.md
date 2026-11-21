# GenEvolve 2B · 从基模开始的三阶段 RETHINK 实验（4×A100 80GB）

默认入口 `all` 从公开的 **Qwen3-VL-2B-Instruct** 起跑，不需要事先拿到别人的 SFT-550。先训练 GenEvolve 自己的图文 SFT，再训练 RETHINK 的 Route-SFT，最后训练 40 步 RETHINK RL。这里的“从头”是从预训练基模开始后训练，不是随机初始化。

| 顺序 | 起点与训练 | 可续训断点 | GenEvolve-Bench 评测 |
|---|---|---|---|
| 1 · GenEvolve 原生 SFT | 2B 基模；8,800 条原生图文轨迹，四卡全局 batch 32，2 个 epoch = 550 步 | 275、550（ZeRO-3 状态） | **只在 550 步测固定 8 题** |
| 2 · Route-SFT | 上一阶段 SFT-550；每批全局 6 Good＋2 Bad，共 281 批，连续两轮 = 562 步 | 281、562；562 另导出 HF 权重 | **只在 562 步测同一组 8 题** |
| 3 · RETHINK RL | Route-SFT-562；GenEvolve GRPO＋原生 SDL，Actor 触发的局部 hint 仅给 Teacher。40 步为一个连续优化周期 | 10、20、30、40（含优化器、调度器、各卡 RNG 和经验库）；每处另留可评测权重 | **10、20、30、40 各测同一组 8 题**；第 40 步再单独跑 **全量 594 题** |

固定 8 题是[锁定的官方测试划分](configs/split_sha256.json)中 ID 0–7，顺序、Actor 温度 0.6、top-p 0.9、每题种子 42＋原下标均相同。它们供配对比较，不能当作全量成绩。两次 SFT 和四次 RL 子集评测，加上末轮 594 题，是 **6 份固定子集结果＋1 份全量结果**；不做 0 步或 SFT 中点的生图 benchmark。原生 SFT 自己的 200 条验证损失按原配方保留，与生图 benchmark 不同。评测不会更新经验库或参与反传。

### 一次启动

需要同一台 Linux 机器上 **4 张可用 A100 80GB**（每张至少 68 GiB 空闲），Python 3.11、CUDA Toolkit 12.8、可访问公开模型与数据的网络，以及足够的服务器磁盘。全程同时保留原生 ZeRO 断点、Route 断点、4 份 RL 断点和导出，另有原生图片数据、生成图、日志与缓存；请勿按“只存最后一份权重”预留空间。训练所需的公开仓库、模型、数据、训练器均锁定版本，见 [sources.json](sources.json)。

```bash
git clone https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x.git
cd genevolve-2b-rethink-type-kl-a100-4x
bash scripts/setup.sh
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3      # 换成分配给本实验的四张卡
export WORK="$PWD/work"                  # 持久化服务器磁盘；后续续跑用同一绝对路径
# 先在 shell 注入下述服务的环境变量。不要把密钥写入仓库或日志。
python run.py --work "$WORK" all
```

- **搜索、评分与局部 hint：**按[服务模板](configs/services.env.example)设置 Serper、Qwen Judge（固定 `qwen3.7-flash-2026-07-15`，关闭 thinking）及支持 `glm-5.3-flash` 的 hint 接口。不要用不同 Judge 得到的分数冒充同一组结果。
- **绘图是另一个服务，不是 Judge API。** `QWEN_SERVICE_URL` 须指向兼容的 Qwen-Image-Edit-2511 渲染端点，接受参考图并返回图片。RL 训练中也要绘图计算奖励，请将渲染放在独立服务或另外分配的 GPU；不要挤占这四张训练卡。自部署可在绘图机的同版本检出运行 `bash scripts/setup.sh --renderer`、`bash scripts/serve_renderer.sh --cpu-offload`；仅绑定/开放给授权调用方。完整流程会请求真实服务并产生费用；训练中断后重用同一 WORK。
- **中断续跑：**重新运行同一条 `all`。阶段交接核对数据、父模型及导出身份；RL 每到一个里程点先落盘、释放训练进程再评测，通过后才继续下一段。各段始终使用 40 步总学习率计划，不是四个独立的 10 步实验。不要删除中间产物或把另一次运行混进这个 WORK。旧的 322 步 RL 工作目录不能当作本实验续跑。
- **已有 SFT-550 可显式跳过原生训练：**`python run.py --work "$WORK" all --parent /path/to/checkpoint-550`。仍会评该模型的固定 8 题。接受有历史 SHA 锁的原始 2B SFT-550，或由本仓库原生配方训练并核验的 550；不接受裸基模、8B 权重或仅把目录命名为 550 的模型。显式外部起点不替你补做它的续训状态。不同设备或软件重新训练不会保证逐字节复现历史权重。

若绘图服务在另一台获授权主机上并保持默认的本地监听，可从训练机建立 SSH 转发：
`ssh -N -L 18001:127.0.0.1:8001 USER@RENDER_HOST`，并在训练进程设置
`QWEN_SERVICE_URL=http://127.0.0.1:18001`。不需要把绘图服务开放到公网。
hint 的 `GENEVOLVE_HINT_API_URL` 必须是完整 HTTPS `/chat/completions` 地址，不是只到 `/v1` 的基址。

### 单独准备或运行

```bash
python run.py --work "$WORK" prepare-native       # 原生 SFT 基模与官方图文数据
python run.py --work "$WORK" prepare-route        # 公开 281 批 Route 包
python run.py --work "$WORK" native-sft --eval    # 只到原生 550，并评固定 8；纯训练可不加 --eval
export PARENT="$WORK/native_sft/checkpoint-550"
python run.py --work "$WORK" check-route --parent "$PARENT"
python run.py --work "$WORK" route-sft --parent "$PARENT" --eval
python run.py --work "$WORK" rethink --sft-checkpoint "$WORK/route_sft/hf-step-00562"
```

原生配方、图文格式及断点见 [NATIVE_SFT.md](NATIVE_SFT.md)。Route-SFT 使用[公开 281 批最小包](https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x/releases/tag/route-sft-reviewed-v4-public)，2,248 条：1,686 Good、562 Bad。同一全局 batch 四卡各处理两条。公开包因脱敏有 1,679 条工具历史变化；每轮 233,323 个监督 token、当前 reasoning、标注和批次顺序不变。持有原始 v4 的人可加 `--data /path/to/original-v4`，不要与公开包共用续训目录。[数据对账](DATASET.md) · [Route-SFT 损失](TYPE_KL.md)。8B 是 Route 样本的采集策略，并非任何训练阶段的起点。

### 产物和运行前检查

- 原生 SFT：`$WORK/native_sft/checkpoint-275`、`checkpoint-550`。Route-SFT：`$WORK/route_sft/checkpoint-00281`、`checkpoint-00562`、`hf-step-00562`。
- RL：`$WORK/rethink/checkpoint-00010`、`00020`、`00030`、`00040`（均以 `checkpoint-` 开头）；评测模型在 `hf-step-00010`、`hf-step-00020`、`hf-step-00030` 及 `final`。权重可用于推理不代表可续训，归档时要同时保留原生训练状态、配置、数据身份、日志和评测目录。
- 评测：`$WORK/eval/` 按模型和范围分目录保存病例记录、图片和官方评分汇总；固定 8 题另存 `fixed8_metrics.json`。基础设施失败不是模型得零分；先排查服务再续跑。最终报告请使用 **40 步模型的全量目录**，不要误取其 8 题均分。
- 不占 GPU 的检查：`bash scripts/setup.sh --sources-only`，安装 `requirements-smoke.txt` 及 CPU PyTorch 后执行 `python -m unittest discover -s tests -v`。要在目标机验证 Route-SFT 真卡显存与通信，可单独运行 `torchrun --standalone --nproc-per-node=4 scripts/check_route_four_gpu.py --parent "$PARENT" --data "$WORK/route_data/route-sft-reviewed-v4-public"`；这只做一次丢弃的内存更新，不改训练断点。
