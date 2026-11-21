# GenEvolve 2B · Route-SFT：旧 281 批库，两轮，Good 动作类型 KL

**默认起点是 GenEvolve 的 Qwen3-VL-2B-Instruct SFT checkpoint-550，不是裸基模，也不是已经训过一轮的 281 步模型。** 之后用同一份 2,248 条决策的库连续训练两个 epoch。**第 281 步只保存可续训状态；第 562 步保存可续训状态、导出 HF 模型，再评测固定 8 题。** 不重置优化器或学习率，不在中点启动生图服务或 Judge。

| 项目 | 本版设置 |
|---|---|
| 数据 | 1,686 Good＋562 Bad，其中 373 条自然 Bad、189 条合成 Bad；每批 6 Good＋2 Bad |
| 训练量 | 281 个有序批次走两遍，共 562 次参数更新；单卡，micro-batch 1，全局 batch 8 |
| 优化 | BF16，语言参数训练、视觉塔冻结；AdamW，LR `2e-6`，betas `(0.9, 0.95)`，weight decay `0.01`，clip `1.0`，seed 0；56 步 warmup，随后 cosine，整个训练仅一个调度器 |
| 监督 | reasoning 不算 CE；每条后缀按自身 token 数取均值。保留独立 marker BCE（0.1），只在 Good 动作边界额外加冻结 SFT-550 的 Tool／Answer／Other 正向 KL（0.1） |
| 上下文 | 原生文本工具协议，最大 12,288 tokens，不静默截断；不加 `<action>` 外壳，不启用 reasoning KL |
| 最终评测 | 锁定版本 GenEvolve-Bench 的第 0–7 题；不是整套 594 题，也不是随机抽取 8 题。缺图按 0 计入全 8 题均分；另报成功评分图数和自由生成 RETHINK 的回合／题目计数 |

### 1. 准备代码、起点和数据

训练环境：Linux、Python 3.11、CUDA 12.8，依赖和上游提交均在仓库内固定。本配方在单卡大显存环境训练；完整状态比单份 HF 权重大，建议为阶段一输出准备至少 **58 GiB** 可用空间（不含环境、起点、数据、渲染器及图片缓存）。仓库名称里的 `a100-4x` 是沿用的项目名，四卡是**可选二阶段**的配置。

```bash
git clone https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x.git
cd genevolve-2b-rethink-type-kl-a100-4x
bash scripts/setup.sh
source .venv/bin/activate
export WORK="$PWD/work-281x2"
export PARENT=/path/to/GenEvolve-2B-SFT/checkpoint-550
python run.py --work "$WORK" prepare-route
python run.py --work "$WORK" check-route --parent "$PARENT"
```

- **权重需由运行者提供**：使用与对照相同的完整 HF 导出，包含模型、tokenizer、processor 和 step 记录。可参考[原 2B SFT 工程（固定提交）](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)。架构相同或目录名叫 `checkpoint-550` 都不够：会核对原实验的模型和 tokenizer 哈希。这里不发布权重，也不以随机或裸基模兜底。
- **训练样本独立提供**：[下载 281 批最小包](https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x/releases/tag/route-sft-reviewed-v4-public)。默认公开包保留每轮全部 233,323 个监督 token 和原批次顺序；因脱敏，1,679/2,248 条样本的工具历史前缀有变，不保证复现原权重或原分数。[数据说明](DATASET.md)给出完整对账。如果已有经过校验的**原始 v4**，可在 `check-route`、`route-sft` 或 `all` 上加 `--data /path/to/original-v4` 使用原实验输入。两种模式不可共用同一工作目录续训。旧公开 v3 的 297 批和仅用自然样本的 186 批都不适用。

### 2. 只做一阶段

```bash
python run.py --work "$WORK" route-sft --parent "$PARENT"
```

仅训练**不需要**搜索、Judge 或 hint 密钥。保存路径分别为 `$WORK/route_sft/checkpoint-00281`、`checkpoint-00562`（续训状态）和 `hf-step-00562`（最终 HF 导出）。如在首轮结束后中断，重跑同一条命令、使用同一起点/数据/工作目录，可从已保存断点恢复；不要指定旧单轮实验的输出作为 `PARENT`。

### 3. 末轮评测

**只有一张卡时，先完成上面训练，再启动绘图器**，这样绘图器不会在 SFT 期间占用显存。另一个终端运行：

```bash
bash scripts/setup.sh --renderer
bash scripts/serve_renderer.sh --cpu-offload
```

在训练环境中设置搜索和 Judge 凭据（见 [配置模板](configs/services.env.example)），并确认绘图服务可达：

```bash
export QWEN_SERVICE_URL=http://127.0.0.1:8001
read -rsp 'Serper key: ' SERPER_API_KEY; echo; export SERPER_API_KEY
read -rsp 'Judge key: ' JUDGE_API_KEY; echo; export JUDGE_API_KEY
# 若服务商不同，设置自己的 JUDGE_BASE_URL；须能提供锁定的 Judge 模型。
python run.py --work "$WORK" prepare
python run.py --work "$WORK" eval --split bench --policy "$WORK/route_sft/hf-step-00562"
```

如已有**独立绘图服务**，可直接 `python run.py --work "$WORK" route-sft --parent "$PARENT" --eval`，只在最终导出后测一次。评测使用 Actor temperature 0.6、top-p 0.9、逐题 seed=42+序号；渲染器 Qwen-Image-Edit-2511，Judge `qwen3.7-flash-2026-07-15`，关闭 thinking。服务不可用时不要换 Judge 后仍宣称属于同一对照。真实在线检索、图片下载与服务端评分会波动；固定题目不代表逐次生成相同图片。

看 `summary.json` 的 `all.overall_missing_zero`、`count_eval_success`，以及 `route_telemetry.json` 的 `marker_turns / turns` 和 `cases_with_marker / cases`。不要把成功出图的均分或给定前缀的 marker 概率当成前述指标。需要整个榜单时显式加 `--full-benchmark`。

### 4. 可选：接原有四卡二阶段

在相同工作目录准备官方数据，按[服务模板](configs/services.env.example)配置搜索、评分、绘图和局部 hint 服务，然后：

```bash
python run.py --work "$WORK" prepare
python run.py --work "$WORK" rethink --sft-checkpoint "$WORK/route_sft/hf-step-00562"
```

`all --parent "$PARENT"` 是两阶段入口，不是“只做 SFT”；`all --eval` 评的是阶段二的最终策略。二阶段仍是原有四卡 **GRPO＋GenEvolve SDL**：Actor 自己触发重审，局部 hint 仅用于 Teacher；没有把其他环境的蒸馏机制替换进来。[算法说明](TYPE_KL.md)给出一阶段的准确公式。
