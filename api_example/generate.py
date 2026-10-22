import requests
import base64
import json
import os
from PIL import Image
from io import BytesIO
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置
API_URL = "http://127.0.0.1:7871/kaloscope/v1/infer"
USERNAME = "user"  # 替换为你的用户名，如果未启用认证可留空
PASSWORD = "password"  # 替换为你的密码，如果未启用认证可留空
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 Base64 字符串"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        raise

def perform_inference(image_path: str, model_name='Kaloscope', **kwargs):
    """调用 /infer 端点进行推理"""
    try:
        input_image_base64 = encode_image_to_base64(image_path)

        # 准备请求数据
        data = {
            "input_image": input_image_base64,
            "model_name": model_name,
            **kwargs
        }

        # 设置认证
        auth = None
        if USERNAME and PASSWORD:
            auth = (USERNAME, PASSWORD)

        # 发送请求
        response = requests.post(API_URL, json=data, auth=auth)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Inference completed: {result['info']}")
        return result['results']

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

# 示例使用
if __name__ == "__main__":
    # 示例参数，请根据你的模型调整
    image_path = "test_image.png"  # 替换为你的测试图片路径
    model_name = "Kaloscope"  # 替换为你的模型文件夹名

    try:
        results = perform_inference(
            image_path=image_path,
            model_name=model_name,
            top_k=5
        )
        print("Inference Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

        # 保存结果
        output_file = os.path.join(OUTPUT_DIR, "inference_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
