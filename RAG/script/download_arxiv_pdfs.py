import arxiv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_paper_pdf(paper, save_dir):
    paper_id = paper.get_short_id()
    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    # 跳过已存在且大于 100KB 的文件
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 100 * 1024:
        return f"[Skip] {pdf_path}"

    try:
        paper.download_pdf(filename=pdf_path)  # arxiv 官方方法
        return f"[OK] {pdf_path}"
    except Exception as e:
        return f"[Failed] {pdf_path}: {e}"


def download_arxiv_pdfs(keywords, max_results=1000, save_dir="arxiv_pdfs", workers=16):
    os.makedirs(save_dir, exist_ok=True)

    # 构造搜索
    query = " OR ".join(keywords)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for paper in search.results():  # 每拿到一篇立即提交任务
            futures.append(executor.submit(download_paper_pdf, paper, save_dir))

        for f in as_completed(futures):
            print(f.result())

    print("All done.")


if __name__ == "__main__":
    download_arxiv_pdfs(
        keywords=["robotics", "robot simulation", "SLAM", "manipulation"],
        max_results=5000,
        workers=16
    )
