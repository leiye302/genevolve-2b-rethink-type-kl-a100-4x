# GenEvolve 2B：原版 Route-SFT + Good 类型 KL（4×A100 80GB）

这是一个独立消融仓库：以[最初 Route-SFT 版本 b18f659](https://github.com/leiye302/genevolve-2b-rethink-a100-4x/tree/b18f659e563d36fc161357da5ddfd65f4266af8f)为基础，仅在一阶段增加 **Good 样本上的 Tool/Answer 类型 KL**，轮次按后续统一设置改为 **1 epoch**。不带 marker 校准、类型 BCE、动作标签 CE 重加权，也不换成 `<action>` 协议。其他仓库不受影响。

在 **GenEvolve 的 Qwen3-VL-2B-Instruct SFT checkpoint-550** 后接一轮 Route-SFT，再在原版 GenEvolve 的 **GRPO + SDL** 上加入 Actor-routed 局部 hint。和 [ALFWorld 主方法](https://github.com/leiye302/rethink-alfworld-qwen25-3b) 一样，只让 Actor 决定触发、只给原有 Teacher 增加局部信息；保留目标环境自己的蒸馏方式。与 2B 基线共享同一份初始 SFT 权重，不重新做冷启动 SFT。

新增项保留冻结 SFT-550 在同一真实 `H/R` 下的“继续工具 / 收尾回答”相对分布，**不是提高全库 Answer 固定先验**，也不把 Bad 的纠正目标拉回原动作。默认系数 `good_type_kl_lambda=0.1` 是可配置的实验起点，尚无改善收尾、reasoning 或最终评分的实测结论。完整概率定义、监督位置和实现边界见 [TYPE_KL.md](TYPE_KL.md)。

对照来源：[2B GRPO](https://github.com/leiye302/genevolve-2b-grpo-a100-4x/tree/2f197591100d8b0bf61e65be9748a0752f91ae30) · [2B GenEvolve 自进化及 SFT](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e) · [作者代码](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c)。做配对比较时，用 baseline 实际读取的同一份 HF 导出，逐文件比较权重 SHA-256，不能只比较模型名、目录名或训练步数。

## 与对照组共用的设置

| 项目 | 固定口径 |
| --- | --- |
| 冷启动 | 同一份 GenEvolve 2B SFT-550；仅方法组再做 Route-SFT |
| 数据与 rollout | 固定版本 GenEvolve-Data-RL，训练 2,575 题；seed 42，每步 8 题 × 每题 6 条轨迹，四卡各负责 2 个完整题组 |
| 采样 | temperature 0.7，top-p 0.95，最多 11 回合／每回合 4,096 新 token，整轨迹 30,000，初始输入 6,144，上下文 65,536 |
| GRPO | 原组内标准化、0.20/0.28 非对称裁剪、轨迹 KL 惩罚系数 0.001、程序奖励权重 0.5、LR 1e-6、cosine；保留基线 FSDP 布局 |
| GenEvolve SDL | 保留原经验提取、检索、动作区间 top-10% 选择、SDL 损失及其系数 2.0；不改成 SDAR 门控 |
| 保存与续训 | 二阶段每 10 步和最后一步存，保留最近两份可续训断点；另导出最终 HF 模型 |
| 渲染与评分 | 固定 Qwen-Image-Edit-2511，40 步、CFG 4、seed 0、最多两张参考图；基线的 `qwen3.7-flash-2026-07-15` 奖励和 KScore 评审 |
| 终评 | GenEvolve-Bench 同序 594 题、KScore、原版 prompt fallback；temperature 0.6、top-p 0.9、逐题 seed=42+序号；可选同版 RL-val 600 或 WISE 1000 |

二阶段默认与两个对照入口一样：`ceil(2575 / 8) = 322` updates。传 `--steps N` 可指定配对预算；**三组必须使用相同的计划步数**，因为 cosine 也以该步数为周期。在计划跑 322 步但停于 30 步，与从头设 `--steps 30` 不等价。实际运行长度以每组运行清单为准。

## 一阶段：6 Good + 2 Bad，恰好一轮

封版包共 2,376 条：1,782 Good / 594 Bad。Bad 含 394 条自然错误和 200 条同状态构造对照（与源 Good 同 batch）。Good 保留真实采样动作；Bad 使用原生 `<tool_call>…</tool_call>` 或 `<answer>…</answer>` 纠正后缀，前面加一个注册后的 `<RETHINK>`。训练与 rollout 保持同一文本工具历史，不给 Actor 插入额外图像观察或换解析器。

**数据来源与训练起点不同。** 复审 v3 数据的离线轨迹由 GenEvolve 8B 采集；本实验从 2B SFT-550 训练。默认使用[公开版训练包](DATASET.md)：只脱敏历史工具响应中的机器路径、访问参数和联系方式，重新计算受影响的历史 token 与边界。样本、批序、当前 reasoning 和受监督 token 不变。启动时用实际 2B SFT-550 tokenizer 核对全部历史 token、边界和解码；若不一致就报错，不临时重分词或跳过样本。

- mask 至 `</think>\n`（含分隔符），只监督动作后缀及 EOS。保留原来的逐样本平均后缀 CE + 0.1 × **全词表** marker BCE，新增 0.1 × Good 类型 KL。Bad 的 marker 仍参与原后缀 CE；不做 marker 校准。无 reasoning CE/KL、正文 KL、LoRA 或额外类型 BCE。
- 按原序扫描 297 个固定批次一遍：全局 8，四卡 × micro 1 × 累积 2，**297 updates**。LR 2e-6，前 10% warmup、cosine、AdamW (0.9, 0.95)、weight decay 0.01、clip 1.0、seed 0。语言参数全参训练，视觉塔不动。
- 开训前四卡分片计算冻结起点在 1,782 条 Good 上的两类概率，缓存于 `WORK/route_sft/good_type_reference`；按起点、数据、配置和 SHA 校验，可断点续算。预计算后卸载参考模型；训练中额外执行候选 `<` 前缀的可微前向，不常驻第二份模型。这个额外计算开销尚未在四张 A100 上实测。
- 公开包最长 11,081 tokens（原档 11,223）；使用 12,288 上限，超长直接报错，不现场重分词或截断。每轮存原生可续训状态和 HF 导出。该封版包只供训练，没有另造 SFT val split；日志中的 train loss 和给定前缀 marker 概率不是自由生成指标。

## 二阶段：GenEvolve GRPO + SDL，Actor 路由局部 hint

完整轨迹做原来的 GRPO。保留 GenEvolve 的高低奖励轨迹配对、视觉经验提取、经验库和检索；与原版一样，新经验在本轮更新后进入库，不用于同一轮检索。Teacher 用**更新前同一个 Actor 的无梯度副本**对原始生成 token 评分：

| Actor 的该回合 | Teacher 的额外条件 |
| --- | --- |
| 无特殊 token `<RETHINK>` | 原样使用检索到的视觉经验；没有检索经验时不蒸馏，照常做 GRPO |
| Actor 生成特殊 token `<RETHINK>` | 原来检索的经验 + 当前回合 hint；空库时可以用 hint 单独给 Teacher 提供条件；hint 不合格时回退到原版条件 |

Teacher 分数进入原版 GenEvolve 的 action-span token 选择与 SDL（仍使用原比例、系数和截断），不是另一个损失。基线的 frozen reference 仍只用于原奖励 KL。局部 API 每条带 marker 的轨迹发一组请求（失败有限重试），不打 Good/Bad、不替 Actor 执行动作、也不替代或写入原经验库。Student、工具、终评均不读 hint。marker 的位置只做统计，不作为路由门禁。

## 运行

Linux x86-64、Python 3.11、CUDA 12.8 和 C/C++ 构建工具。沿用对照的四张 A100 80GB、FSDP 和 CPU offload；使用同版独立渲染服务，或另开渲染机执行 `bash scripts/setup.sh --renderer`、`bash scripts/serve_renderer.sh`。每个训练阶段首次启动至少需 100 GiB 可用空间；两阶段同盘时还要计入已保存的一阶段产物、数据、生成图以及渲染器缓存。实际峰值尚未在四卡验证。

运行方只需提供与对照组相同的起点权重；训练数据可公开下载：

- `PARENT`：对照组实际使用的 **GenEvolve 2B checkpoint-550 完整 HF 导出**，可指向 [2B 复现仓库](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)的 `WORK/sft` 或同文件的 `checkpoint-550`。须含权重、tokenizer、processor、`trainer_state.json`；不能传原始基模、8B、二阶段权重或只有 ZeRO 状态的目录。
- **Route-SFT 数据**：[本仓库 GitHub Release 训练包](https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x/releases/tag/route-sft-reviewed-v3-public)，不需要账号权限或另找本地文件。与原版公开包逐字节一致。`prepare-route` 下载、校验并解压到 `WORK/route_data/route-sft-reviewed-v3-public`；`all` 和 `route-sft` 缺数据时也会自动下载。如果服务器连不上 GitHub Releases，可在联网机器上下载并解压，再用 `--data /path/to/extracted-pack` 指定。

用 `check-sft --sft-checkpoint "$PARENT"` 分别检查方法和基线的 HF 导出并核对权重 SHA-256。[v3 的 8B 采集来源指纹](configs/route_data_source_8b_sha256.json)仅用于校验数据。已经使用原始 v3 开训时，继续传原来的 `--data`；公开版使用新的 `WORK`，不能跨数据指纹续训。旧 v2 不兼容。

```bash
git clone https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x.git
cd genevolve-2b-rethink-type-kl-a100-4x
bash scripts/setup.sh
source .venv/bin/activate
export WORK=/path/to/independent-run
export PARENT=/path/to/genevolve-2b-sft/checkpoint-550
python run.py --work "$WORK" prepare-route
# 使用自己的四张获配 GPU；由调度器指定时保持它的映射。
python run.py --work "$WORK" check-route --parent "$PARENT"
```

二阶段和评测需要 Serper + 基线 Judge；二阶段原有的视觉经验提取也继续使用基线 Judge。方法训练额外需要局部 hint API（`glm-5.3-flash`、低思考、HTTPS OpenAI-compatible chat completions；OpenCode Go 每条轨迹使用独立会话 ID）。先确认服务商允许接收训练轨迹。**局部 hint 请求**只带本条 rollout 的文本证据和指定步号，不带真值图或本地图片路径；**原版经验提取**按既有实现向 Judge 提供 best/worst 渲染及参考目标图。凭据通过环境注入，不写进 Git 或命令行。详见 [环境变量模板](configs/services.env.example)。

```bash
export QWEN_SERVICE_URL=http://render-host:8001
export GENEVOLVE_HINT_API_URL=https://opencode.ai/inference/go/openai/v1/chat/completions
read -rsp 'Serper key: ' SERPER_API_KEY; printf '\n'
read -rsp 'Judge key: ' JUDGE_API_KEY; printf '\n'
read -rsp 'Hint key: ' GENEVOLVE_HINT_API_KEY; printf '\n'
export SERPER_API_KEY JUDGE_API_KEY GENEVOLVE_HINT_API_KEY
# 准备官方数据 → 一阶段 Route-SFT + Good 类型 KL → 核验 297 步导出 → 二阶段
python run.py --work "$WORK" all --parent "$PARENT"
# 可选独立终评；或在 all 命令末尾加 --eval，训练完成后自动做全量终评。
python run.py --work "$WORK" eval --split bench
```

只训练一阶段时，可单独执行 `python run.py --work "$WORK" route-sft --parent "$PARENT"`，无需 API；手动接二阶段时先执行 `python run.py --work "$WORK" prepare`，再执行 `python run.py --work "$WORK" rethink --sft-checkpoint "$WORK/route_sft/hf-step-00297"`。二阶段入口自动准备固定版本的 Qwen3-Embedding-0.6B，也可用 `prepare-embedding` 提前下载。重新执行 `all` 会复用已验收的数据和一阶段最终导出；若一阶段尚未结束，则从已保存的 epoch 断点续训。任何一阶段报错或导出校验失败均不会开二阶段。

同 2B 纯 GRPO 对照可用本仓库的 `grpo --sft-checkpoint "$PARENT"`，不调用 hint API；不带 RETHINK 的 GenEvolve 对照请用独立的 2B 复现仓库。原冷启动和 Route-SFT 的 HF 模型可复用同一 `eval --policy ... --split bench`；小规模试测可用 `--max-cases N`，不应混入正式终评。重启使用原命令、同一 WORK、数据、起点、计划步数与服务配置；二阶段断点保存原版经验库。**本消融必须使用新的 WORK**，不能复用无 KL、类型 BCE 或双改动版本的训练状态／HF 导出；入口校验完整配置。不要把 8B 的断点用于 2B。

终评在不改分数的前提下另写 `route_telemetry.json`，按决策回合统计原生特殊 marker，并单列位置正确率，不按总 token 数作分母。RL-val 同样记录。

## 离线检查

```bash
bash scripts/setup.sh --sources-only
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -v
```

离线检查以 mock 工具、Judge、hint 覆盖 rollout、解码、Teacher 输入、梯度和断点写入。真实 GPU 训练、API 质量和图像评分需在目标机验证。
