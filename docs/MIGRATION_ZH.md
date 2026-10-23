# 更新节点与迁移工作流

## 安装与模型复用

当前组织仓库为 [Chenkin-x/Kaloscope-Style-adapter-Comfyui](https://github.com/Chenkin-x/Kaloscope-Style-adapter-Comfyui)。更新到3.0.0后，重启ComfyUI并刷新浏览器。

本包独立提供生图功能，可与原版 `comfyui-kaloscope` 同时安装。旧混合扩展版需先备份，再移除其中重复的adapter节点注册及搜索前端，或换回原版；保留反推和分析功能。备份放在 `custom_nodes` 之外。

新适配器目录为 `models/kaloscope_adapters/`，也会扫描旧安装目录。下拉框显示公开模型名称并映射到原文件，无需移动、改名或重新下载权重。若新旧目录中存在同名选项，新目录优先。风格库继续使用 `models/kaloscope_style_banks/`。

编码器仍读取 `models/kaloscope/<已有目录>/`，支持匹配的纯风格版与v1 classifier版。旧发行目录在新加载器中显示为 `artist-v1`；其他已有目录保留各自名称。原反推包仍读取原路径。原加载器的 `KALOSCOPE_MODEL` 输出能直接连接本包 Reference Encode；同时反推与生图时共用它即可复用同一个实例。新精简加载器不提供分类头，不能反向供画师分类节点使用。

## UI 工作流

拖入旧UI JSON时，前端自动转换旧生图节点ID、adapter连线类型及模型下拉值，并清理旧默认标题。提示词、随机种子、数值参数和图片输入不变。保存后得到当前命名的工作流。

若旧原加载器只用于参考图生图，则迁移为独立编码器加载器；若还连接画师反推或分析分支，则保留原加载器和原目录名。混合工作流仍需安装原插件。

## API 工作流

API请求不会经过浏览器迁移。提交旧API JSON前执行：

```bash
python scripts/migrate_workflow.py old_workflow.json new_workflow.json
```

同一脚本也可处理UI JSON。它拒绝覆盖原文件或已有输出；转换只写入指定的新文件。已迁移的工作流再次执行不会改变含义。也可直接使用 `examples/release/` 的当前示例。

节点菜单只注册当前功能名称。底层模型格式标记在内部兼容读取中保留，因此无需重新导出已下载权重。

## MOSS 原生加载

当前示例使用 ComfyUI 内置的 Load Diffusion Model、CLIPLoader（type=`moss_image`）和 VAELoader。下载官方单文件Base、文本编码器和VAE后，在示例中选择对应文件即可。

旧HF目录加载节点作为兼容入口保留并隐藏，不会把旧目录自动改造成单文件模型。原工作流仍需原HF插件；切换到原生加载请导入当前MOSS示例。旧工作流中的独立采样补丁不要照搬：核心MOSS的时间步倍率为1，直接使用默认设置；Apply Style已处理负分支skip22。
