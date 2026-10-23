# 发布工作流

UI：拖入不带 `_api` 的JSON。API：提交对应 `_api.json`。Raw63006与Turbo33000各有画师名称、参考图和加权融合入口；MOSS8500有名称与参考入口。

先选择本机已安装的底模、文本编码器和VAE。Raw/Turbo复用官方VAELoader，默认 `qwen_image_vae.safetensors`，无需创建别名或复制VAE。参考图示例需要在LoadImage中选择自己的图片。

Raw：25步Euler、text CFG2、style CFG3.5；Turbo：8步Euler、text/style CFG1、固定μ1.15。adapter strength默认1，可在固定种子下比较0.8。MOSS：50步、text CFG3、style CFG6、shift5、原生HF插件Base权重；路径改为本机模型目录。

风格注入与时间表使用Kaloscope节点，其余使用标准接口；MOSS由官方HF Loader输出标准MODEL/CLIP/VAE。不要为MOSS重复叠加SLG。原生ComfyUI新版MOSS后端尚未验证与当前adapter补丁兼容。
