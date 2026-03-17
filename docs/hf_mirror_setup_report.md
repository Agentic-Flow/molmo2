# Molmo2 环境接入 HF 镜像站工作报告

日期：2026-03-10（UTC）

## 1. 目标

为当前 `uv + venv` 的 Molmo2 复现环境接入 `https://hf-mirror.com`，缓解 `huggingface.co` 直连超时导致的模型下载失败问题；并给出可重复执行的自检方案。

## 2. 本次完成的工作

### 2.1 镜像连通性验证

已执行并通过：

- `curl -I https://hf-mirror.com`
- `curl -I https://hf-mirror.com/allenai/Molmo2-4B/resolve/main/config.json`

结果：

- 站点可访问（HTTP 200）
- 模型资源解析可用（HTTP 307，返回镜像侧缓存/解析路径）

### 2.2 持久化配置（全局 shell）

已写入 `~/.bashrc`：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

用途：使 `huggingface_hub` / `transformers` 默认走镜像 endpoint（新 shell 生效）。

### 2.3 通过镜像进行实际下载验证

在激活 `.venv` 且加载 `.bashrc` 后，已成功下载以下 Molmo2 元数据文件：

- `config.json`
- `processor_config.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`

缓存路径示例：

`~/.cache/huggingface/hub/models--allenai--Molmo2-4B/snapshots/<commit>/...`

### 2.4 处理运行时兼容性（与下载相关的关键前置）

为保证后续视频/多模态链路可用，本次同步将依赖栈稳定在兼容组合：

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchcodec==0.2.0`
- `FFmpeg 7.1 shared`（安装于 `~/ffmpeg`，并由 `~/.bashrc` 注入 PATH/LD_LIBRARY_PATH）

该组合已验证可正常 `import torchcodec` 和 `from torchcodec.decoders import VideoDecoder`。

## 3. 结果结论

1. `HF_ENDPOINT` 持久化配置已完成。
2. 镜像站可访问，且 `allenai/Molmo2-4B` 的关键元数据可成功下载到本地缓存。
3. 之前的下载/初始化阻塞点（直连超时）已得到可操作的替代路径。

## 4. 下一步自检方式（建议按顺序执行）

> 先执行：`source ~/.bashrc && source .venv/bin/activate`

### A. 快速自检（30 秒）

```bash
python - <<'PY'
import os, torch
print('HF_ENDPOINT=', os.environ.get('HF_ENDPOINT'))
print('torch=', torch.__version__)
print('cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())
PY
```

通过标准：

- `HF_ENDPOINT` 显示为 `https://hf-mirror.com`
- `cuda=True`

### B. 下载链路自检（1~2 分钟）

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ['config.json', 'processor_config.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
    p = hf_hub_download('allenai/Molmo2-4B', f)
    print('ok:', f, '->', p)
PY
```

通过标准：4 个文件均下载/命中缓存成功。

### C. Processor 自检（2~5 分钟）

```bash
python - <<'PY'
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('allenai/Molmo2-4B', trust_remote_code=True, padding_side='left')
print(type(p).__name__)
PY
```

通过标准：输出 `Molmo2Processor`（或等效 Molmo2 processor 类型）。

### D. 端到端推理自检（可选，耗时最长）

当网络和磁盘配额允许时再执行完整权重下载 + 单图推理。若失败，先区分：

- 网络问题：重复 B/C 步是否稳定
- 存储问题：检查缓存目录空间
- 鉴权问题：私有仓库需 `huggingface-cli login`（本模型通常不需要）

## 5. 常见注意事项

1. 修改了 `~/.bashrc` 后，必须 `source ~/.bashrc` 或新开终端。
2. `transformers` 关于 `trust_remote_code` 的提示是正常安全提示。
3. 若后续某些命令需要临时绕过镜像，可单次覆盖：

```bash
HF_ENDPOINT=https://huggingface.co python your_script.py
```
