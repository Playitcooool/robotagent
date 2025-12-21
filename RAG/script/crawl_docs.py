import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from markdownify import markdownify
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

visited = set()

def clean_filename(name):
    """清理文件名，去除非法字符"""
    name = re.sub(r"[^\w\-_ .]", "_", name)
    return name[:150]  # 避免太长

def normalize_url(url):
    """规范化URL，去除末尾的斜杠"""
    if url.endswith('/'):
        return url[:-1]
    return url

def get_domain_from_url(url):
    """从URL提取域名"""
    parsed = urlparse(url)
    return parsed.netloc

def extract_main_content(soup):
    """提取正文内容，支持多种选择器"""
    # 常见的文档正文选择器
    selectors = [
        "main",
        "#content",
        ".content",
        ".documentation",
        ".docs-content",
        ".markdown-body",
        "article",
        ".main-content",
        "#main-content",
        ".doc-content"
    ]
    
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element
    
    # 如果没有找到特定选择器，返回整个body
    return soup.find("body") or soup

def crawl_and_convert(url, domain, save_dir="docs_md", delay=1, max_depth=5, current_depth=0):
    """
    爬取整个开源文档并转为 Markdown
    
    Args:
        url: 要爬取的URL
        domain: 限制的域名
        save_dir: 保存目录
        delay: 请求延迟（秒）
        max_depth: 最大爬取深度
        current_depth: 当前深度
    """
    if current_depth > max_depth:
        return
        
    url = normalize_url(url)
    
    if url in visited:
        return
    visited.add(url)

    logger.info(f"正在爬取 ({current_depth}/{max_depth}): {url}")

    try:
        time.sleep(delay)  # 延迟，避免被封
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"爬取失败 {url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # 提取正文区域
    main = extract_main_content(soup)
    if not main:
        logger.warning(f"未找到正文内容: {url}")
        return
        
    html_content = str(main)

    # 转成 markdown
    try:
        md = markdownify(html_content, heading_style="ATX")
    except Exception as e:
        logger.error(f"转换Markdown失败 {url}: {e}")
        md = ""

    # 标题作为文件名
    title = soup.title.text.strip() if soup.title else "document"
    filename = clean_filename(title) + ".md"

    # 创建子目录保存
    if save_dir:
        # 使用域名创建子目录
        domain_dir = domain.replace('.', '_')
        sub_dir = os.path.join(save_dir, domain_dir)
        os.makedirs(sub_dir, exist_ok=True)
        filepath = os.path.join(sub_dir, filename)
    else:
        filepath = filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"源URL: {url}\n\n")
            f.write("---\n\n")
            f.write(md)
        logger.info(f"已保存: {filepath}")
    except Exception as e:
        logger.error(f"保存文件失败 {filepath}: {e}")

    # 递归爬取站内链接
    if current_depth < max_depth:
        links_to_crawl = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            
            # 跳过锚点链接
            if href.startswith('#'):
                continue
                
            # 处理相对链接和绝对链接
            if href.startswith("/"):
                next_url = urljoin(url, href)
            elif href.startswith("http"):
                next_url = href
            else:
                # 处理相对路径
                next_url = urljoin(url, href)
            
            # 规范化URL
            next_url = normalize_url(next_url)
            
            # 只爬特定域名的链接
            parsed_next = urlparse(next_url)
            if parsed_next.netloc != domain:
                continue
            
            # 屏蔽文件、PDF、图片等
            blocked_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", 
                                 ".tar", ".gz", ".exe", ".mp4", ".mp3", ".avi"]
            if any(parsed_next.path.lower().endswith(ext) for ext in blocked_extensions):
                continue
            
            # 检查URL是否已访问
            if next_url not in visited:
                links_to_crawl.append(next_url)
        
        # 去重
        links_to_crawl = list(set(links_to_crawl))
        logger.info(f"找到 {len(links_to_crawl)} 个新链接待爬取")
        
        # 递归爬取
        for next_url in links_to_crawl:
            crawl_and_convert(next_url, domain, save_dir, delay, max_depth, current_depth + 1)

def batch_crawl(urls, save_dir="docs_md", delay=1, max_depth=3, max_workers=3):
    """
    批量爬取多个网站
    
    Args:
        urls: URL列表，可以是字符串或字典
        save_dir: 保存目录
        delay: 请求延迟
        max_depth: 最大爬取深度
        max_workers: 最大并发数
    """
    # 处理不同的URL格式
    tasks = []
    for item in urls:
        if isinstance(item, dict):
            # 字典格式：{"url": "https://...", "domain": "example.com", "depth": 2}
            url = item.get("url", "")
            domain = item.get("domain") or get_domain_from_url(url)
            depth = item.get("depth", max_depth)
            if url and domain:
                tasks.append((url, domain, depth))
        else:
            # 字符串格式
            url = str(item)
            domain = get_domain_from_url(url)
            if url and domain:
                tasks.append((url, domain, max_depth))
    
    logger.info(f"开始批量爬取 {len(tasks)} 个站点")
    
    # 使用线程池并发爬取
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for url, domain, depth in tasks:
            # 为每个站点重置已访问集合
            visited.clear()
            future = executor.submit(
                crawl_and_convert, 
                url, 
                domain, 
                save_dir, 
                delay, 
                depth,
                0  # 起始深度
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"任务执行失败: {e}")

if __name__ == "__main__":
    # 示例1：字符串列表
    url_list = [
        "https://gazebosim.org/home",
    ]
    
    
    # 使用方式1：直接传URL列表
    batch_crawl(url_list, save_dir="my_docs", delay=1, max_depth=2)
    