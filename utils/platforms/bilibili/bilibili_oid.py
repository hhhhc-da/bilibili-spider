import requests
from requests.exceptions import RequestException, JSONDecodeError

def get_oid_from_bilibili_url(video_url: str) -> int:
    try:
        # 提取BV号
        if "/video/" not in video_url:
            raise ValueError("无效的B站视频链接")
            
        bvid = video_url.split("/video/")[-1].split("?")[0].strip('/')
        if not bvid.startswith("BV"):
            raise ValueError("BV号格式错误")

        # 构造API请求
        api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": video_url
        }

        # 发送请求（禁用代理和SSL验证）
        response = requests.get(
            api_url,
            headers=headers,
            proxies={},
            timeout=10,
            verify=False
        )
        
        # 检查HTTP状态码
        if response.status_code != 200:
            raise RequestException(f"HTTP错误: {response.status_code}")

        # 解析JSON
        data = response.json()
        if data.get("code") != 0:
            raise ValueError(f"API返回错误: {data.get('message')}")

        return bvid, data["data"]["aid"]
        
    except JSONDecodeError:
        print("响应内容不是JSON格式, 原始内容:")
        print(response.text[:1000])
        return bvid, None
    except Exception as e:
        print(f"操作失败: {str(e)}")
        return bvid, None

# 使用示例
if __name__ == '__main__':
    oid = get_oid_from_bilibili_url("https://www.bilibili.com/video/BV1kQFUeaEeW")
    print("视频OID:", oid)