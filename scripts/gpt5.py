from veo3 import call_api_raw,prepare_image_data,create_output_directory,API_KEY,BASE_URL,API_TIMEOUT,USE_STREAM,call_openai_with_retry,client,save_mixed_content
import os,time,json,multiprocessing as _mp

MODEL_NAME = "gemini-2.5-pro"

def generate(input_image_path, prompt_text)->str:
    """use gpt-5 to process images with prompt_text, return response content and output directory"""
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

    content_list = [{"type": "text", "text": prompt_text}]
    content_list.extend(image_contents)

    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]
    
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
    if use_raw_response:
        response_content = completion['choices'][0]['message'].get('content', '')
    else:
        response_content = completion.choices[0].message.content
    with open(output_directory+'/input.txt','w') as file:
        file.write(f'Model name:\n{MODEL_NAME}\nInput image path:\n{input_image_path}\nPrompt:\n{prompt_text}')
        messages_text = json.dumps(messages, indent=2, ensure_ascii=False)
        file.write(f'\n\nMessages:\n{messages_text}')
    save_mixed_content(response_content, output_directory)
    return output_directory

def generate_multiple_tries(input_image_path, prompt_text, attempts=3):
    """retry multiple times to generate output"""
    result = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"\n=== 第 {attempt} 次尝试生成 ===")
            result = generate(input_image_path, prompt_text)
            result_txt = os.path.join(result, "content.txt")
            if not os.path.exists(result_txt):
                raise FileNotFoundError(f"Expected output not found at {result_txt}")
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
    input_image_path, prompt_text, attempts = arg_tuple
    if attempts and attempts > 1:
        return generate_multiple_tries(input_image_path, prompt_text, attempts=attempts)
    return generate(input_image_path, prompt_text)

def generate_outputs_multiprocess(image_paths_list, prompt_texts, processes=None, attempts=1, chunksize=1):
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(prompt_texts, (list, tuple)):
        raise TypeError("image_paths_list and prompt_texts must be lists/tuples")
    if len(image_paths_list) != len(prompt_texts):
        raise ValueError("image_paths_list and prompt_texts must have the same length")
    if attempts is None or attempts < 1:
        attempts = 1

    tasks = [(image_paths_list[i], prompt_texts[i], attempts) for i in range(len(prompt_texts))]

    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        results = pool.map(_mp_worker_generate, tasks, chunksize=chunksize)

    return results

if __name__ == "__main__":
    pass