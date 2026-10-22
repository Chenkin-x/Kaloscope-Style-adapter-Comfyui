import requests
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置
API_URL = "http://127.0.0.1:7871/kaloscope/v1/cancel"
USERNAME = "user"  # 替换为你的用户名，如果未启用认证可留空
PASSWORD = "password"  # 替换为你的密码，如果未启用认证可留空

def cancel_inference():
    """调用 /cancel 端点取消推理"""
    try:
        # 设置认证
        auth = None
        if USERNAME and PASSWORD:
            auth = (USERNAME, PASSWORD)

        # 发送请求
        response = requests.post(API_URL, auth=auth)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Cancel result: {result['info']}")
        return result['info']

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Cancel failed: {str(e)}")
        raise

# 示例使用
if __name__ == "__main__":
    try:
        result = cancel_inference()
        print(f"Cancel Result: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
