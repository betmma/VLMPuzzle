# generate_video_output(input_image_path, prompt_text) -> output_directory_path
# generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=3) -> output_directory_path
# generate_video_outputs_multiprocess(image_paths_list, prompt_texts, processes=None, attempts=1, chunksize=1) -> list of output_directory_path
import os
os.environ['NO_PROXY'] = '*'
import base64
import re
import requests
import time
import json
from datetime import datetime
import multiprocessing as _mp
from openai import OpenAI
import cv2

# ====================================
# 用户配置变量 - 请根据需要修改以下设置
# ====================================

# API配置
API_KEY = 'api_key.txt'
with open(API_KEY, 'r', encoding='utf-8') as f:
    API_KEY = f.read().split('\n')[0].strip()
MODEL_NAME = 'sora_video2'#"veo3-frames"  # 使用的模型名称
BASE_URL = 'https://jyapi.ai-wx.cn/v1' if MODEL_NAME=='sora_video2'else "https://api.sydney-ai.com/v1"  
# BASE_URL = 'https://api.chatanywhere.tech/v1/'

DEFAULT_INPUT_IMAGE = r"data/mirror/puzzles/c92274c1-deae-4f22-ab1a-ae5a8039694f_puzzle.png"
DEFAULT_PROMPT = "Instantly reflect this pattern along the central, vertical axis while keeping the existing colored pattern without modification. Static camera perspective, no zoom or pan."

# 重试设置
MAX_RETRIES = 1  # 最大重试次数
RETRY_DELAY = 0  # 重试延迟时间（秒），0表示立即重试

# API调用超时设置
API_TIMEOUT = 600  # API调用超时时间（秒），建议120秒以等待图片生成
USE_STREAM = True  # 必须使用流式响应才能获取完整的图片数据！

# ====================================
# 以下为功能代码，一般情况下无需修改
# ====================================

def prepare_image_data(image_path):
    """准备图片数据，转换为base64格式"""
    try:
        with open(image_path, "rb") as img_file:
            encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
            return "data:image/png;base64," + encoded_data
    except Exception as e:
        print(f"准备图片数据时出错: {image_path} - {e}")
        raise

def create_output_directory():
    """创建输出目录"""
    # Include microseconds to avoid collisions in multiprocessing
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # trim to milliseconds for shorter names
    timestamp = ts[:-3]
    base_dir = os.path.join("data", "output")
    os.makedirs(base_dir, exist_ok=True)
    candidate = os.path.join(base_dir, f"output_{timestamp}")
    if os.path.exists(candidate):
        counter = 1
        while True:
            suffix = f"_{counter:02d}"
            next_candidate = candidate + suffix
            if not os.path.exists(next_candidate):
                candidate = next_candidate
                break
            counter += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate

def save_base64_image(base64_data, output_dir, image_index):
    """保存base64图片到本地"""
    try:
        # 移除data:image/png;base64,前缀（如果存在）
        if base64_data.startswith('data:image/'):
            base64_data = base64_data.split(',', 1)[1]

        # 解码base64数据
        image_data = base64.b64decode(base64_data)

        # 保存图片
        image_filename = f"image_{image_index}.png"
        image_path = os.path.join(output_dir, image_filename)

        with open(image_path, "wb") as img_file:
            img_file.write(image_data)

        print(f"已保存base64图片: {image_path}")
        return image_path
    except Exception as e:
        print(f"保存base64图片时出错: {e}")
        return None

def download_image_from_url(url, output_dir, image_index):
    """从URL下载图片到本地"""
    try:
        response = requests.get(url, stream=True, proxies={})
        response.raise_for_status()

        # 获取文件扩展名
        content_type = response.headers.get('content-type', '')
        if 'png' in content_type.lower():
            ext = 'png'
        elif 'jpg' in content_type.lower() or 'jpeg' in content_type.lower():
            ext = 'jpg'
        elif 'gif' in content_type.lower():
            ext = 'gif'
        else:
            ext = 'png'  # 默认扩展名

        # 保存图片
        image_filename = f"image_url_{image_index}.{ext}"
        image_path = os.path.join(output_dir, image_filename)

        with open(image_path, "wb") as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                img_file.write(chunk)

        print(f"已下载URL图片: {image_path}")
        return image_path
    except Exception as e:
        print(f"下载URL图片时出错: {e}")
        return None

def download_video_from_url(url, output_dir, video_index):
    """从URL下载视频到本地"""
    try:
        response = requests.get(url, stream=True, proxies={})
        response.raise_for_status()

        url_path = url.split("?", 1)[0]
        ext = os.path.splitext(url_path)[1].lstrip('.').lower() or 'mp4'

        content_type = response.headers.get('content-type', '').lower()
        if 'webm' in content_type:
            ext = 'webm'
        elif 'mov' in content_type:
            ext = 'mov'
        elif 'mp4' in content_type:
            ext = 'mp4'

        video_filename = f"video_{video_index}.{ext}"
        video_path = os.path.join(output_dir, video_filename)

        with open(video_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    video_file.write(chunk)

        print(f"已下载视频: {video_path}")
        return video_path
    except Exception as e:
        print(f"下载视频时出错: {e}")
        return None

def extract_last_frame(video_path, output_dir):
    """提取视频的最后一帧并保存为result.png"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法读取视频: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        print(f"视频帧数为0: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print(f"读取最后一帧失败: {video_path}")
        return None

    result_path = os.path.join(output_dir, "result.png")
    saved = cv2.imwrite(result_path, frame)
    cap.release()

    if saved:
        print(f"已保存视频最后一帧: {result_path}")
        return result_path

    print(f"保存最后一帧失败: {result_path}")
    return None

def save_mixed_content(content, output_dir):
    """保存混合内容（文字、base64图片、URL图片）"""
    try:
        # 查找base64图片
        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
        base64_matches = re.finditer(base64_pattern, content)

        # 查找URL链接
        url_pattern = r'https?://[^\s<>"]+\.(png|jpg|jpeg|gif)'
        url_matches = re.finditer(url_pattern, content, re.IGNORECASE)

        # 查找视频链接
        video_pattern = r'https?://[^\s<>"]+\.(?:mp4|webm|mov|mkv)(?:\?[^\s<>"]*)?'
        video_matches = list(re.finditer(video_pattern, content, re.IGNORECASE))

        # 保存文字内容到文件
        text_content = content
        image_index = 1

        # 处理base64图片
        for match in base64_matches:
            full_match = match.group(0)
            base64_data = match.group(1)

            # 保存base64图片
            saved_path = save_base64_image(base64_data, output_dir, image_index)
            if saved_path:
                # 在文本中替换base64数据为文件路径
                text_content = text_content.replace(full_match, f"[保存的图片: {saved_path}]")
                image_index += 1

        # 处理URL图片
        for match in url_matches:
            url = match.group(0)

            # 下载URL图片
            saved_path = download_image_from_url(url, output_dir, image_index)
            if saved_path:
                # 在文本中替换URL为文件路径
                text_content = text_content.replace(url, f"[下载的图片: {saved_path}]")
                image_index += 1

        # 处理视频链接，优先下载链接
        video_urls = []
        for match in video_matches:
            url = match.group(0)
            if url not in video_urls:
                video_urls.append(url)

        video_urls.sort(key=lambda u: 0 if 'download' in u.lower() else 1)

        last_frame_path = None

        for idx, url in enumerate(video_urls, start=1):
            saved_path = download_video_from_url(url, output_dir, idx)
            if saved_path:
                text_content = text_content.replace(url, f"[下载的视频: {saved_path}]")
                if last_frame_path is None:
                    frame_path = extract_last_frame(saved_path, output_dir)
                    if frame_path:
                        last_frame_path = frame_path

        if last_frame_path:
            text_content += f"\n提取的最后一帧: {last_frame_path}"

        # 保存处理后的文字内容
        text_filename = os.path.join(output_dir, "content.txt")
        with open(text_filename, "w", encoding="utf-8") as text_file:
            text_file.write(text_content)

        print(f"已保存文字内容: {text_filename}")

        # 同时保存原始内容
        original_filename = os.path.join(output_dir, "original_content.txt")
        with open(original_filename, "w", encoding="utf-8") as original_file:
            original_file.write(content)

        print(f"已保存原始内容: {original_filename}")

    except Exception as e:
        print(f"保存混合内容时出错: {e}")

def is_quota_exceeded_error(error_message):
    """检查是否为配额超出错误"""
    quota_keywords = [
        "exceeded your current quota",
        "quota exceeded",
        "billing details",
        "plan and billing"
    ]
    error_str = str(error_message).lower()
    return any(keyword in error_str for keyword in quota_keywords)

def call_api_raw(api_key, base_url, model, messages, timeout=API_TIMEOUT, use_stream=False, output_dir=None):
    """使用原始HTTP请求调用API，获取完整响应"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream
    }

    url = f"{base_url}/chat/completions"

    try:
        print(f"发送原始HTTP请求到: {url}")
        if use_stream:
            print("使用流式响应模式...")

        response = requests.post(url, headers=headers, json=data, timeout=timeout, stream=use_stream, proxies={})
        response.raise_for_status()

        if use_stream:
            # 处理流式响应
            full_content = ""
            all_chunks = []

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str != '[DONE]':
                            try:
                                chunk = json.loads(data_str)
                                all_chunks.append(chunk)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        full_content += delta['content']
                            except json.JSONDecodeError:
                                pass

            # 保存所有流式数据（调试用）
            if output_dir:
                debug_path = os.path.join(output_dir, "stream_chunks.json")
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

            print(f"流式响应: 接收到 {len(all_chunks)} 个数据块")
            if len(full_content) > 1000:
                print(f"获取到完整数据: {len(full_content)} 字符（包含图片）")
            else:
                print(f"获取到文本内容: {len(full_content)} 字符")

            # 构造标准响应格式
            json_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    }
                }],
                "stream_chunks": all_chunks
            }
        else:
            # 获取完整的JSON响应
            json_response = response.json()

        # 保存原始JSON响应用于调试
        if output_dir:
            debug_path = os.path.join(output_dir, "raw_api_response.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(json_response, f, ensure_ascii=False, indent=2)
            print(f"原始API响应已保存到: {debug_path}")

        return json_response
    except requests.exceptions.RequestException as e:
        print(f"HTTP请求失败: {e}")
        raise

def call_openai_with_retry(client, model, messages, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY, timeout=API_TIMEOUT):
    """带重试功能的OpenAI API调用"""
    for attempt in range(max_retries):
        try:
            print(f"第 {attempt + 1} 次尝试调用API...")
            if timeout > 60:
                print(f"设置超时时间: {timeout}秒 (等待图片生成)")

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout
            )

            print("API调用成功！")
            return completion

        except Exception as e:
            error_message = str(e)
            print(f"API调用失败: {error_message}")

            # 检查是否为配额超出错误或超时错误
            if is_quota_exceeded_error(error_message):
                if attempt < max_retries - 1:  # 还有重试机会
                    if retry_delay > 0:
                        print(f"检测到配额超出错误，将在 {retry_delay} 秒后进行第 {attempt + 2} 次重试...")
                        time.sleep(retry_delay)
                    else:
                        print(f"检测到配额超出错误，立即进行第 {attempt + 2} 次重试...")
                    continue
                else:
                    print("已达到最大重试次数，仍然配额超出，请检查账户余额和计费设置。")
                    raise
            elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
                if attempt < max_retries - 1:  # 还有重试机会
                    print(f"API调用超时，可能图片生成需要更长时间，立即进行第 {attempt + 2} 次重试...")
                    continue
                else:
                    print("已达到最大重试次数，API仍然超时。建议增加API_TIMEOUT设置或检查网络连接。")
                    raise
            else:
                # 非配额/超时错误，直接抛出
                print("非配额/超时相关错误，不进行重试。")
                raise

    # 如果所有重试都失败了
    raise Exception(f"经过 {max_retries} 次重试后仍然失败")

DEBUG_MODE = False  # 设置为True以查看详细调试信息

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def generate_video_output(input_image_path, prompt_text):
    """根据输入图片和提示词生成视频，并返回输出目录路径"""
    if isinstance(input_image_path, (list, tuple)):
        image_paths = [str(path) for path in input_image_path if path]
    elif isinstance(input_image_path, str):
        image_paths = [input_image_path]
    else:
        raise TypeError("input_image_path must be a string or a sequence of strings")

    if not image_paths:
        raise ValueError("No input image paths provided")

    # 准备所有图片数据
    image_contents = []
    for i, image_path in enumerate(image_paths):
        try:
            print(f"处理第 {i+1} 张图片: {image_path}")
            image_data = prepare_image_data(image_path)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": image_data,
                },
            })
        except Exception as e:
            print(f"处理图片时出错: {image_path} - {e}")
            continue

    if not image_contents:
        raise RuntimeError("没有成功处理任何图片")

    # 构建消息内容（包含所有图片）
    content_list = [{"type": "text", "text": prompt_text}]
    content_list.extend(image_contents)

    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]

    print(f"\n共发送 {len(image_contents)} 张图片到API")

    # 创建输出目录（提前创建以保存调试文件）
    output_directory = create_output_directory()
    print(f"创建输出目录: {output_directory}")

    # 先尝试使用原始HTTP请求获取完整响应
    print("\n使用原始HTTP请求调用API...")
    use_raw_response = False
    completion = None
    try:
        raw_response = call_api_raw(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            messages=messages,
            timeout=API_TIMEOUT,
            use_stream=USE_STREAM,
            output_dir=output_directory
        )

        completion = raw_response
        use_raw_response = True
    except Exception as e:
        print(f"原始HTTP请求失败: {e}")
        print("\n回退到OpenAI客户端...")
        completion = call_openai_with_retry(
            client=client,
            model=MODEL_NAME,
            messages=messages
        )

    if DEBUG_MODE:
        print("\n=== 调试信息 ===")
        if use_raw_response:
            content = completion['choices'][0]['message']['content']
            print(f"Content长度: {len(content)} 字符")

            debug_path = os.path.join(output_directory, "raw_content_debug.txt")
            with open(debug_path, "wb") as f:
                f.write(content.encode('utf-8'))
            print(f"完整content已保存到: {debug_path}")

            import re
            base64_pattern = r'[A-Za-z0-9+/]{100,}={0,2}'
            matches = re.findall(base64_pattern, content)
            if matches:
                print(f"发现base64图片数据: {len(matches)} 个")
        else:
            print(f"响应类型: OpenAI客户端对象")
            print("\n消息对象:", completion.choices[0].message)

            if hasattr(completion.choices[0].message, '__dict__'):
                print("\n消息对象所有属性:")
                for key, value in completion.choices[0].message.__dict__.items():
                    if value:
                        print(f"  {key}: {str(value)[:200]}...")

    if use_raw_response:
        response_content = completion['choices'][0]['message'].get('content', '')
        message_dict = completion['choices'][0]['message']
        possible_image_fields = ['images', 'image', 'attachments', 'media', 'files', 'data']

        for field_name in possible_image_fields:
            if field_name in message_dict and message_dict[field_name]:
                print(f"\n发现图片字段 '{field_name}'!")
                images_data = message_dict[field_name]

                if isinstance(images_data, list):
                    for idx, img in enumerate(images_data):
                        if isinstance(img, str):
                            if not img.startswith('data:'):
                                response_content += f"\ndata:image/png;base64,{img}"
                            else:
                                response_content += f"\n{img}"
                        elif isinstance(img, dict):
                            if 'data' in img:
                                response_content += f"\ndata:image/png;base64,{img['data']}"
                            elif 'url' in img:
                                response_content += f"\n{img['url']}"
                            elif 'base64' in img:
                                response_content += f"\ndata:image/png;base64,{img['base64']}"
                elif isinstance(images_data, str):
                    if not images_data.startswith('data:'):
                        response_content += f"\ndata:image/png;base64,{images_data}"
                    else:
                        response_content += f"\n{images_data}"
    else:
        response_content = completion.choices[0].message.content

    if not use_raw_response:
        if hasattr(completion.choices[0].message, 'images'):
            print("\n发现images字段!")
            images = completion.choices[0].message.images
            if images:
                print(f"包含 {len(images)} 张图片")
                for idx, img in enumerate(images):
                    if isinstance(img, str):
                        response_content += f"\ndata:image/png;base64,{img}"
                    elif hasattr(img, 'data'):
                        response_content += f"\ndata:image/png;base64,{img.data}"

    print("\nAI响应内容:")
    print(response_content[:500] + "..." if len(response_content) > 500 else response_content)

    save_mixed_content(response_content, output_directory)
    print(f"\n所有内容已保存到目录: {output_directory}")

    return output_directory

def generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=3):
    """retry multiple times to generate video output"""
    result = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"\n=== 第 {attempt} 次尝试生成视频 ===")
            result = generate_video_output(input_image_path, prompt_text)
            result_png = os.path.join(result, "result.png")
            if not os.path.exists(result_png):
                raise FileNotFoundError(f"Expected result frame not found at {result_png}")
            break
        except Exception as e:
            print(f"尝试失败: {e}")
            if attempt == attempts:
                print("已达到最大尝试次数，停止尝试。")
                raise
            else:
                print("稍后重试...")
                time.sleep(2)
    return result

def _mp_worker_generate(arg_tuple):
    """Worker function for multiprocessing.
    arg_tuple: (input_image_path_or_list, prompt_text, attempts)
    Returns output directory path.
    """
    input_image_path, prompt_text, attempts = arg_tuple
    if attempts and attempts > 1:
        return generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=attempts)
    return generate_video_output(input_image_path, prompt_text)

def generate_video_outputs_multiprocess(image_paths_list, prompt_texts, processes=None, attempts=1, chunksize=1):
    """Generate multiple video outputs in parallel using multiprocessing.

    Args:
        image_paths_list: list where each item is either a string path or a list of image paths
                          for a single request. Length must match `prompt_texts`.
        prompt_texts: list of prompt strings aligned with `image_paths_list`.
        processes: number of worker processes to use. Defaults to CPU count.
        attempts: per-item retry attempts (>=1). If >1, uses generate_video_output_multiple_tries.
        chunksize: chunksize passed to Pool.map for throughput tuning.

    Returns:
        List of output directory paths in the same order as inputs.
    """
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(prompt_texts, (list, tuple)):
        raise TypeError("image_paths_list and prompt_texts must be lists/tuples")
    if len(image_paths_list) != len(prompt_texts):
        raise ValueError("image_paths_list and prompt_texts must have the same length")
    if attempts is None or attempts < 1:
        attempts = 1

    tasks = [(image_paths_list[i], prompt_texts[i], attempts) for i in range(len(prompt_texts))]

    # Use 'spawn' for cross-platform safety (Windows/macOS/Linux)
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        results = pool.map(_mp_worker_generate, tasks, chunksize=chunksize)

    return results

def main():
    output_dir = generate_video_output(DEFAULT_INPUT_IMAGE, DEFAULT_PROMPT)
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
