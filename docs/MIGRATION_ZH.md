# 从合并扩展版迁移到独立生图包

## 已安装原版 comfyui-kaloscope

直接安装本包即可。两者使用不同节点 ID、前端路由和 Python 命名空间；本包不注册 `KaloscopeModelLoader`、LSNet 或画师分类/分析节点。

模型继续放在 `ComfyUI/models/kaloscope/<原目录名>/`。在新 Style Encoder Loader 选择已有 v1 模型目录即可。若该目录有 `config.json`、兼容的 v1 classifier 权重和 class mapping，新加载器复用同一个权重文件并跳过分类头。不要求改目录名，不要求第二份权重。

已有“原加载器 → Z3 Reference Encode”连接仍可运行；共享同一个原加载器节点可以在反推与生成分支之间复用已加载的模型。新加载器只提供风格投影，不能代替原加载器向画师反推节点提供分类头。

## 已安装此前的合并扩展版

旧扩展版同时注册分类与 adapter 节点；必须避免同名节点被两个包覆盖。

1. 将旧包完整备份到 `custom_nodes` 之外，保留工作流与模型。
2. 要继续反推/分析：使用原版 `comfyui-kaloscope`；要保留本地修改，可只删除旧根 `__init__.py` 中导入 `kaloscope_style`、合并 `STYLE_NODES/STYLE_NAMES`、注册其搜索路由的附加段，并停用旧 adapter 搜索 JS/CSS。其他分类与分析代码保持原样。不要批量删除原插件自己的前端。
3. 将本包安装到 `custom_nodes/Kaloscope-Style-adapter-Comfyui/`，安装本包 requirements。
4. 重启 ComfyUI，刷新浏览器。生图节点显示在 `Kaloscope Adapter` 分类。

只有生图需求时，可以把旧包整体移出 `custom_nodes`；新包可独立运行。名称生图不需要编码器，参考图分支需要训练匹配的 v1 编码器。

## 工作流迁移

当前 `examples/release/` 已使用独立加载器及官方 VAE 文件名。也可转换自己的旧 UI/API JSON：

```bash
python scripts/migrate_workflow.py old_workflow.json new_workflow.json
```

脚本保留原文件，拒绝覆盖已有输出。只有加载器的消费者全是 `KaloscopeZ3EncodeReference` 时才替换类型，保留模型目录、设备、连接和 `KALOSCOPE_MODEL` socket；若同一加载器还用于反推，则保持原加载器。这类混合工作流仍需安装原插件。

新包不自动转换LSNet模型或创建符号链接。仅加载器接口兼容不代表所有 Kaloscope 编码器的特征都与本 adapter 匹配。
