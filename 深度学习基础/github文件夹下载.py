import requests
import os
def download_github_folder(github_url, output_path=None):
    if "github.com" not in github_url:
        print("❌ 无效的 GitHub URL")
        return
    parts = github_url.rstrip('/').split('/')
    try:
        owner = parts[3]
        repo = parts[4]
        if 'tree' in parts:
            tree_index = parts.index('tree')
            branch = parts[tree_index + 1]
            folder_path = '/'.join(parts[tree_index + 2:])
        else:
            branch = 'main'
            folder_path = '/'.join(parts[5:])
    except IndexError:
        print("❌ 无法解析 URL，请确保格式正确")
        return
    download_api = f"https://downgit.zxq.co/download"
    params = {
        'url': f'https://github.com/{owner}/{repo}/tree/{branch}/{folder_path}'
    }
    response = requests.get(download_api, params=params, stream=True)
    if response.status_code == 200:
        # 推测文件夹名
        folder_name = folder_path.split('/')[-1] if folder_path else repo
        filename = f"{folder_name}.zip"
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, filename)
        else:
            filepath = filename
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ 成功下载到: {filepath}")
    else:
        print(f"❌ 下载失败，状态码: {response.status_code}")
if __name__ == "__main__":
    # 修改下面这行的 URL 为你想下载的 GitHub 文件夹链接
    github_url = "https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/pairwise_cls_similarity_afqmc"

    # 可选：指定下载到哪个本地文件夹，不填则下载到当前目录
    output_path = "../resource/"  # 例如 "./my_folder" 或 "C:/Users/Name/Downloads"

    # 开始下载
    download_github_folder(github_url, output_path)