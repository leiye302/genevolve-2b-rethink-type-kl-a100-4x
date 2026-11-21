# 原生 GenEvolve 冷启动：从公开 2B 基模到 SFT-550

这是 Route-SFT **之前**的多模态指令微调。它不加入 `<RETHINK>`，不使用 Good/Bad、路由 BCE、类型 KL、检索、绘图或 Judge 服务。执行的是固定的 [GenEvolve-2B-Reproduction 配方](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)，**不是声称从原论文下载了 2B 训练器或复现作者的 8B 权重**。[原作者仓库](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c)公开的策略基座是 8B，并提供多轮 SFT 轨迹和参考图。

在本仓库根目录执行：

```bash
bash scripts/setup.sh
source .venv/bin/activate
export WORK=/path/to/your-work
python run.py --work "$WORK" prepare-native
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --work "$WORK" native-sft
```

`prepare-native` 只需 CPU；它取固定版本的 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203) 至 `$WORK/base`、[GenEvolve-Data-SFT](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench/tree/9f291879d5dfe1195f47883e23069d3f6e829afe/GenEvolve-Data-SFT) 至 `$WORK/data/GenEvolve-Data-SFT`。已有上述完整目录时可直接复制到位，再运行同一命令离线核对。不需下载图像渲染器。权重约 4.26 GB；作者标注此 SFT 数据约 7.4 GB。含 50,291 张不同的参考 JPEG，小文件传输可能比较慢。程序检查模型/配置/分词器/处理器和两个 JSONL 的 SHA-256、图文占位数、8,800 条训练及 200 条验证记录，并从图像实际字节重建官方 Git-LFS 图像目录指纹；不会把缺图样本改成纯文本。

**四张卡的数值设置本来就是上游配方，不是把单卡 batch 放大四倍。** 每卡 2 条、累积 4 次，全局 batch 是 $4\times2\times4=32$。每轮 $8800/32=275$ 步，两轮共 550 步。多轮 ShareGPT 图文输入、真实图像占位、`qwen3_vl_nothink` 模板；所有历史及当前 assistant 回合都监督（包括原生 reasoning 和工具/最终作画程序），不监督 system/user/视觉 token。模板名的 `nothink` 不会抹掉数据里已有的显式推理。首次开跑先用固定的 LLaMA-Factory 真实预处理器取一条含图样本检查掩码。

全量训练语言参数，冻结视觉塔及多模态投影；不 packing，32,768 token 截断。每图 1,024–589,824 pixels。AdamW，学习率 $10^{-5}$，weight decay $10^{-6}$，cosine、2% warmup；seed/data seed 42，BF16、FlashAttention 2、梯度检查点，4 卡 DeepSpeed ZeRO-3，无 optimizer/parameter CPU offload。训练和验证均用原始 epoch 调度；原生阶段的 `eval_loss` 与最终 GenEvolve-Bench 图像分数不是一回事。[原始训练配置](configs/native_sft.json)保持上游原样；实际启动只叠加[存盘频率覆盖项](configs/native_sft_save.json)：`save_strategy=steps`、`save_steps=550`。`eval_strategy=epoch`、训练目标、优化器和两轮训练量不变。

四张独占 A100-80GB（每卡至少 68 GiB 空闲），Python 3.11、PyTorch 2.8.0/CUDA 12.8、LLaMA-Factory 固定提交，按安装脚本准备。冷启动新训保守预检要求工作盘还剩至少 75 GiB，不包含后续两个阶段和额外渲染模型的空间预算。GPU 内核、驱动和数据加载会影响逐位结果，因此新训的 550 不冒充历史 550 的 SHA。

第 275 步照常记录验证 loss，不保存权重、优化器、RNG 或 HF 导出。第 550 步才保存完整的 **`$WORK/native_sft/checkpoint-550`**，其中有 HF 权重、tokenizer/processor、Trainer 进度、各 rank 的 RNG 和 ZeRO 优化器状态；用它作为 Route-SFT 的 `--parent`。要评估冷启动模型的固定八题，使用 `native-sft --eval`（另需评测服务）。

若在 550 步前中断，没有中途续训点，再开跑将从已校验的裸基模开始。已完成并校验的 550 会直接复用；若终点目录存在但缺训练成功记录，不覆盖、不补签，需保留现场并使用新 WORK 或恢复原始记录。老版本已封存的 550 仍可经逐文件核验用于交接。旧 275 不会删除，也不能作为新配方的中途起点。
