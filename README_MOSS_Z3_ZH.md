# MOSS Base：Z3 艺术家名称与参考图工作流

新增 `Kaloscope MOSS Base Loader`，默认读取 `ComfyUI/models/MOSS-Image-Base/`；也可以填写完整模型目录。输出标准 MODEL、CLIP、VAE，复用官方 `moss-image` 插件和 ComfyUI 显存管理。需要已安装提供 `MOSSImageHFLoader` 的官方 MOSS 插件。本扩展不下载模型。

原 `Kaloscope Z3 Checkpoint Loader` 和 `Kaloscope Z3 Apply Style` 现在同时支持 Krea2 与 MOSS。MOSS adapter 为42层、3072宽、4个共享bank，共7,017,216个参数；按训练实现 `floor(layer_index*4/42)` 分配bank，仅修改图像token。两种底模的adapter不可混用。

## 文件与工作流

```text
ComfyUI/models/
  MOSS-Image-Base/                       # 默认；包含 transformer/text_encoder/tokenizer/vae
  kaloscope_z3/moss_z3_step00008500.safetensors
  kaloscope/z3_artist_v1/config.json
  kaloscope/z3_artist_v1/model.safetensors
  kaloscope_style_banks/z3_artist_v1_recent.safetensors
```

示例默认使用全量训练第8500步的固定快照，避免追踪正在覆盖的训练断点。导出来源：
`<training-output>/adapter_step00008500.pt`。
导出文件位于同服务器：
`ComfyUI/models/kaloscope_z3/moss_z3_step00008500.safetensors`。

将以下 UI 工作流 JSON 拖入 ComfyUI。带 `_api.json` 后缀的是 HTTP `/prompt` 格式：

- `examples/moss/Kaloscope_MOSS_Base_Z3_artist_name.json`：在 Artist Name Style 节点填写规范艺术家名称。默认 `chahan_(fried_rice0614)`。沿用现有44,129位画师原型库；名称清单在 `models/kaloscope_style_banks/z3_artist_v1_recent.artists.txt`。
- `examples/moss/Kaloscope_MOSS_Base_Z3_reference.json`：在 Load Image 中选择自己的风格参考图。默认沿用现有 `input/kaloscope_z3_reference.png`。Reference Encode 使用训练匹配的 `z3_artist_v1`，不需要知道艺术家名称。

名称触发是“名称查询风格原型，再注入 adapter”，不是把画师名作为新文本 token 训练进 MOSS。正向提示词用于描述新画面，不需要再写画师名。参考图也不作为 img2img 内容图；两套示例都从纯噪声生成新图。

## 默认推理参数

| 参数 | 默认值 |
|---|---|
| 底模 | MOSS-Image-Base，bf16 |
| 尺寸 | 512×512，batch 1；可改为64对齐的宽高 |
| 采样 | Euler，50步，MOSS专用离散时间表 |
| Flow shift | 5，由 Kaloscope MOSS Z3 Evaluation Sigmas 设置 |
| 文本 CFG | 3，在 SamplerCustom 设置 |
| 风格 CFG | 6，在 Apply Style 设置 |
| adapter strength | 1 |
| seed | 17，固定 |
| 负向提示词 | 空；负分支内部跳过第22层 |

MOSS Apply 自动实现：`v = v_negative_skip22 + text_cfg*(v_text-v_negative_skip22) + style_cfg*(v_style-v_text)`。**不要额外连接 SkipLayerGuidance 或其他自定义 CFG 节点**。该节点已处理负分支 skip22；重复的冲突补丁会报错。

使用 `EmptyFlux2LatentImage` 的128通道 latent，不使用Krea2的单帧视频latent或Krea2 Z3 Sigmas节点。风格 CFG 或 adapter strength 设为0可生成同seed Base对照，负分支 skip22仍保留。可用已有 Blend Styles 融合多图或名称原型。

当前验证目标是官方原生 MOSS BF16；未验证 FP8/GGUF、ControlNet、编译或第三方模型替换组合。ComfyUI文本编码与训练环境的数值路径不同，不保证同seed逐像素等同于训练评估；长提示词在原生ComfyUI中的截断策略也与训练的128 token不同。

## 更新权重

```bash
python scripts/export_z3.py /path/to/adapter_stepXXXXXXXX.pt /path/to/ComfyUI/models/kaloscope_z3/moss_z3_stepXXXXXXXX.safetensors
```

权重读取使用 `weights_only=True` 或 safetensors，严格验证架构、42层bank映射、state dict与有限值；无需携带优化器。刷新文件列表后切换到新adapter即可。节点代码更新后需重启ComfyUI。

## 验证

数值测试覆盖MOSS导出回读、42层映射、仅图像token残差、负分支skip22、独立三分支CFG、零风格Base对照，以及原Krea2回归。实机验证记录在 `evidence/moss/`。

实机名称分支使用训练服务器已有的48位demo画师库 `moss_z3_demo_artists.safetensors`，与示例默认大库使用同一artist-v1特征空间。默认大库保留在本机，未为测试上传；两者按画师平均的样本集不同，因此同名输出可能不同。参考图实机测试使用服务器已有demo图，而非本机默认参考图。

实测结果：14项单元/节点接口测试通过；两个不同文本长度的双样本微型MOSS模型上，ComfyUI风格分支与训练实现最大绝对误差3.58e-7，关闭adapter后与原生分支逐位一致；12组steps/shift时间表与原生MOSS调度器逐位一致。两条工作流与同seed Base对照均通过真实HTTP队列生成512×512图像，UI端口/链接也已按该版本节点schema校验。这些验证证明接口与数值路径可用，不代表大规模风格质量评估。

部署状态：用户授权后，8500步推理权重已复制到本机 `models/kaloscope_z3/moss_z3_step00008500.safetensors`，SHA256与服务器一致，已通过本机adapter实际加载校验（MOSS、42层、7,017,216参数）。两套工作流需要的Base、大画师库、DINO和参考图均已确认存在。重启ComfyUI后即可导入示例运行。
