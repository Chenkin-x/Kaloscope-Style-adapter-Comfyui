# MOSS Base：画师名称与参考图生图

使用 `examples/release/Kaloscope_moss_artist_name.json` 或 `Kaloscope_moss_reference.json`。对应 `_api.json` 为HTTP队列输入。

更新 ComfyUI 后，使用三个内置加载节点：

| 节点 | 文件 / 设置 | 模型目录 |
| --- | --- | --- |
| Load Diffusion Model (`UNETLoader`) | `moss_image_base_bf16.safetensors`，weight_dtype=`default` | `models/diffusion_models/` |
| CLIPLoader | `qwen3.5_9b_bf16.safetensors`，type=`moss_image` | `models/text_encoders/` |
| VAELoader | `flux2-vae.safetensors` | `models/vae/` |

文件来自 [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main)。当前adapter对应 **MOSS-Image-Base**，不要选择SFT。无需安装独立的MOSS加载插件。

适配器选择 `Kaloscope-MOSS.safetensors`，风格库选择 `artist-v1.safetensors`。参考图可用新 Style Encoder Loader 或原反推包加载器，两者共用 `models/kaloscope/` 中训练匹配的artist-v1编码器。

| 参数 | 比较起点 |
| --- | --- |
| 分辨率 | 512×512，宽高64对齐 |
| 采样 | Euler，50步 |
| 时间表 | MOSS Sampling Schedule，shift=5 |
| 文本CFG / 风格CFG | 3 / 6 |
| adapter强度 | 1 |
| latent | `EmptyFlux2LatentImage`，128通道 |

保留官方模型加载后的默认采样设置。不要添加 `ModelSamplingSD3`，它会改变时间步倍率；如需单独调整模型shift，可用 `ModelSamplingAuraFlow`。示例使用专用SIGMAS节点设置实际采样时间表。

Apply Style内置负分支skip22和独立风格CFG，不要再叠加SLG或另一套CFG补丁。风格强度或风格CFG设为0可生成同seed底模对照。Krea2与MOSS权重、latent、VAE和时间表不能混用。

模型卡、限制和下载见[公开模型仓库](https://huggingface.co/heathcliff01/Kaloscope-style-adapter)。当前MOSS更容易回到通用插画风格，不承诺与Krea2相同的画师区分能力。FP8/GGUF、编译、ControlNet及第三方补丁组合尚未全面验证。

维护者导出权重：

```bash
python scripts/export_adapter.py checkpoint.pt Kaloscope-MOSS.safetensors
```

权重读取严格检查架构、42层映射与有限值。既有权重无需转换；代码命名更新不改变模型数值。

旧HF目录加载节点仅用于兼容已有工作流，不出现在新节点菜单中。使用原生加载请导入本仓库当前示例，并选择对应的单文件模型。
