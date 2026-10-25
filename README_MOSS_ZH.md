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
| 采样 | Euler，25步 |
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

加载器支持 dim256 的 B4（4组）与 B8（8组）MOSS adapter，按训练映射 `floor(layer_index * banks / 42)` 注入全部42层，并严格检查架构、层映射与有限值。导出保留全部权重张量。

使用 B8 权重前，请更新本节点包并重启 ComfyUI，再在 Adapter Loader 中选择对应文件。节点会显示实际共享组数；旧 B4 MOSS 和 Krea2 权重仍可加载。

如果在风格分支采样时遇到 `RuntimeError: aimdo memory compile error`，且堆栈包含本插件 `MOSSStyleGuidance` 和原生 MOSS 的 `prefetch_queue_pop`，请更新本插件到 **3.2.3 或更新版本，并完全退出后重新启动 ComfyUI**。3.2.3 修复了风格残差替换原生 MOSS 缓冲区、导致张量跨越 AIMDO 内存编译区生命周期的问题：原生推理改为原地写入图像 token，保留动态显存管理，残差公式不变。已有 adapter 权重和工作流可继续使用，无需重新转换权重。

旧HF目录加载节点仅用于兼容已有工作流，不出现在新节点菜单中。使用原生加载请导入本仓库当前示例，并选择对应的单文件模型。
