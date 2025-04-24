import os

# 随便存的 Cookie, 不要泄露这一段
def get_cookie(path=os.path.join("cookie", "cookie.txt"), encoding='utf-8'):
    cookie = None
    with open(path, 'r', encoding=encoding) as f:
        cookie = f.read().strip()
    return cookie