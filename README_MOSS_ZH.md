# MOSS Base：画师名称与参考图生图

使用 `examples/release/Kaloscope_moss_artist_name.json` 或 `Kaloscope_moss_reference.json`。对应 `_api.json` 为HTTP队列输入。

需要官方 MOSS 插件提供的 `MOSSImageHFLoader`，加载 MOSS-Image-Base 完整目录并输出 MODEL、CLIP、VAE。将路径改为本机实际目录。旧 `KaloscopeMOSSBaseLoader` 包装节点保留兼容；更新版ComfyUI内置MOSS后端尚未验证与当前adapter补丁等价。

适配器选择 `Kaloscope-MOSS.safetensors`，风格库选择 `artist-v1.safetensors`。参考图可用新 Style Encoder Loader 或原反推包加载器，两者共用 `models/kaloscope/` 中训练匹配的artist-v1编码器。

| 参数 | 比较起点 |
| --- | --- |
| 分辨率 | 512×512，宽高64对齐 |
| 采样 | Euler，50步 |
| 时间表 | MOSS Sampling Schedule，shift=5 |
| 文本CFG / 风格CFG | 3 / 6 |
| adapter强度 | 1 |
| latent | `EmptyFlux2LatentImage`，128通道 |

Apply Style内置负分支skip22和独立风格CFG，不要再叠加SLG或另一套CFG补丁。风格强度或风格CFG设为0可生成同seed底模对照。Krea2与MOSS权重、latent、VAE和时间表不能混用。

模型卡、限制和下载见[公开模型仓库](https://huggingface.co/heathcliff01/Kaloscope-style-adapter)。当前MOSS更容易回到通用插画风格，不承诺与Krea2相同的画师区分能力。FP8/GGUF、编译、ControlNet及第三方补丁组合尚未全面验证。

维护者导出权重：

```bash
python scripts/export_adapter.py checkpoint.pt Kaloscope-MOSS.safetensors
```

权重读取严格检查架构、42层映射与有限值。既有权重无需转换；代码命名更新不改变模型数值。
