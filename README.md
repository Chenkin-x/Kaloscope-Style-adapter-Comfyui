# Kaloscope Style Adapter for ComfyUI

为 **Krea-2-Raw、Krea-2-Turbo 和 MOSS-Image-Base** 提供画师名称、多画师融合和参考图风格生成。

本包专用于图像生成。画师名称反推与特征分析请使用 [comfyui-kaloscope](https://github.com/Chenkin-x/comfyui-kaloscope)；两者可同时安装，并共享匹配的 Kaloscope artist-v1 编码器。

| 资源 | 链接 |
| --- | --- |
| 模型权重与模型卡 | [Hugging Face](https://huggingface.co/heathcliff01/Kaloscope-style-adapter) |
| 训练代码 | [Kaloscope Adapter Trainer](https://github.com/Chenkin-x/Kaloscope-adapter-trainer) |
| 示例工作流 | [examples/release](examples/release) |

## 效果展示

### 画师名称

![画师名称生成](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/artist-name.jpg)

### 多画师融合

![多画师融合](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/artist-mix.jpg)

### 参考图

![参考图生成](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/reference.jpg)

名称与融合示例来自 Turbo，参考图示例包含 Raw 和 Turbo。

## 安装

在 ComfyUI 的 `custom_nodes` 目录中克隆仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Chenkin-x/Kaloscope-Style-adapter-Comfyui.git
python -m pip install -r Kaloscope-Style-adapter-Comfyui/requirements.txt
```

请使用运行 ComfyUI 的 Python 环境安装依赖。Windows 便携版使用其 `python_embeded/python.exe`。安装后重启 ComfyUI 并刷新浏览器。

请更新 ComfyUI 至支持 Krea2 / MOSS-Image 的版本。MOSS 使用内置的 Load Diffusion Model、CLIPLoader 和 VAELoader，无需额外安装 MOSS 加载插件。

## 模型下载

| 模型 | 匹配底模 | 下载 | 模型卡 |
| --- | --- | --- | --- |
| Krea Kaloscope Raw | Krea-2-Raw | [权重](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Krea-Kaloscope-Raw.safetensors?download=true) | [说明](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/raw/README.md) |
| Krea Kaloscope Turbo | Krea-2-Turbo BF16 | [权重](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Krea-Kaloscope-Turbo.safetensors?download=true) | [说明](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/turbo/README.md) |
| Kaloscope MOSS | MOSS-Image-Base | [权重](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Kaloscope-MOSS.safetensors?download=true) | [说明](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/moss/README.md) |

名称与名称融合需要 [artist-v1 风格库](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/style-bank/artist-v1.safetensors?download=true)。参考图需要 [artist-v1 编码器](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/tree/main/encoder/artist-v1)目录中的配置、权重和许可。

```text
ComfyUI/models/
├── kaloscope_adapters/
│   ├── Krea-Kaloscope-Raw.safetensors
│   ├── Krea-Kaloscope-Turbo.safetensors
│   └── Kaloscope-MOSS.safetensors
├── kaloscope_style_banks/
│   └── artist-v1.safetensors
└── kaloscope/
    └── artist-v1/
        ├── config.json
        ├── model.safetensors
        └── LICENSE.md
```

按需下载对应的 adapter。底模、文本编码器和 VAE 需单独安装：

- **Krea2**：从 [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) 下载对应底模、文本编码器和 `qwen_image_vae.safetensors`，分别放入 `diffusion_models/`、`text_encoders/` 和 `vae/`。
- **MOSS**：从 [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) 下载 `moss_image_base_bf16.safetensors`、`qwen3.5_9b_bf16.safetensors` 和 `flux2-vae.safetensors`，分别放入 `diffusion_models/`、`text_encoders/` 和 `vae/`。CLIPLoader 的类型选择 `moss_image`。当前 adapter 匹配 **Base**，不要替换为 SFT。

已有反推插件的匹配 artist-v1 或 v1 classifier 时，直接选择原模型目录，无需再次下载编码器。也可将原 `Kaloscope Model Loader` 的输出连接到本包 `Kaloscope Reference Encode`。纯名称生图不需要加载编码器。

模型仓库还提供安装脚本：

```bash
hf download heathcliff01/Kaloscope-style-adapter --local-dir Kaloscope-style-adapter
python Kaloscope-style-adapter/install_models.py --comfyui "/path/to/ComfyUI" --variant turbo --with-encoder
```

`--variant` 支持 `raw`、`turbo`、`moss` 和 `all`；已有编码器或仅使用名称入口时可省略 `--with-encoder`。

## 使用

从 [examples/release](examples/release) 导入工作流。不带 `_api` 后缀的 JSON 可直接拖入 ComfyUI；带 `_api` 的文件用于 HTTP 队列。

1. 选择对应底模、文本编码器、VAE 和 adapter。
2. 选择风格来源：
   - **画师名称**：选择风格库，填写画师名。
   - **多画师融合**：填写 `fkey, (rolua:0.8)` 等加权组合。
   - **参考图**：选择匹配的编码器，在 Load Image 中选择图片。
3. 在提示词中描述画面内容，设置尺寸和种子，运行工作流。

名称节点支持本地搜索、别名和近似拼写匹配，候选以当前风格库实际收录的画师为准。参考图提供风格条件，不是图生图的内容输入。

### 起始参数

| 参数 | Raw | Turbo | MOSS |
| --- | ---: | ---: | ---: |
| 采样器 | Euler | Euler | Euler |
| 步数 | 25 | 8 | 50 |
| 文本 CFG | 2 | 1 | 3 |
| 风格 CFG | 3.5 | 1 | 6 |
| Adapter strength | 1 | 1 | 1 |
| 时间表 | Raw 分辨率调度 | Turbo，固定 μ=1.15 | MOSS，shift=5 |

可从 512×512 开始，宽高使用64的倍数。Raw/Turbo 可固定种子比较强度1和0.8，强度1能取得更好的风格参考效果但角色肢体和画面构图的不稳定性会增加，turbo在8步采样下可能出现降噪不完全现象，可以将采样步数提升至12步。风格强度或风格CFG设为0可生成底模对照。

Raw/Turbo 使用单帧 `EmptyHunyuanLatentVideo`；MOSS 使用 `EmptyFlux2LatentImage`。工作流中的时间表与风格注入节点需要配套使用。

### 节点

节点位于 `Kaloscope Adapter` 分类。

| 节点 | 功能 |
| --- | --- |
| Kaloscope Style Encoder Loader | 加载参考图风格编码器 |
| Kaloscope Adapter Loader | 加载风格适配器 |
| Kaloscope Artist Style | 查询单个画师的风格原型 |
| Kaloscope Artist Mix | 加权融合多个画师 |
| Kaloscope Reference Encode | 编码参考图风格 |
| Kaloscope Blend Styles | 融合多组风格特征 |
| Kaloscope Apply Style | 将风格条件应用到生成模型 |
| Kaloscope Krea2 Sampling Schedule | 生成 Raw/Turbo 采样时间表 |
| Kaloscope MOSS Sampling Schedule | 生成 MOSS 采样时间表 |

## 使用注意

- Adapter需匹配底模，Krea2与MOSS的权重、latent、VAE和时间表不可混用。VAE没有随adapter微调，直接使用对应的官方VAE。
- 参考图编码器必须匹配artist-v1特征空间；相同输出维度不代表兼容。
- 风格强度过高可能损害人体和构图。参考图、混合风格和不同种子的效果存在差异。
- MOSS使用官方加载器的默认模型采样设置；不要额外添加 `ModelSamplingSD3`。Apply Style已包含负分支skip22，不要重复叠加SLG或其他CFG补丁。FP8/GGUF、ControlNet、编译及第三方补丁组合尚未全面验证。

旧UI工作流加载时会自动迁移。旧API工作流、合并扩展版安装及模型目录兼容详见[迁移指南](docs/MIGRATION_ZH.md)。MOSS使用细节见 [MOSS说明](README_MOSS_ZH.md)。

## 开发

```bash
python -m unittest discover -s tests -v
node tests/test_workflow_frontend.mjs
```

导出训练检查点：

```bash
python scripts/export_adapter.py checkpoint.pt adapter.safetensors
```

测试范围与结果见[验证记录](docs/VALIDATION_ZH.md)。问题反馈请提交到 [Issues](https://github.com/Chenkin-x/Kaloscope-Style-adapter-Comfyui/issues)。

## 许可与致谢

本项目基于 [comfyui-kaloscope](https://github.com/Chenkin-x/comfyui-kaloscope) 的相关接口与实现开发，独立维护生图功能。节点代码沿用 GPL-3.0，见 [LICENSE](LICENSE)；所含 DINOv3 源码保留其[独立许可](kaloscope_style/_vendor/dinov3/LICENSE.md)。来源与归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Krea Kaloscope Raw / Turbo 权重使用 Krea 2 Community License；Kaloscope MOSS 权重使用 Apache-2.0；配套编码器保留 DINOv3 License。具体条款见[模型许可](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/LICENSE.md)与 [NOTICE](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/NOTICE)。
