import google.generativeai as genai
import PIL.Image
import numpy as np
import time
import random
import os
import torch
import base64
import json
from io import BytesIO
import tempfile

class GeminiProNode:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
                    if self.api_key:
                        genai.configure(api_key=self.api_key)
                        print("[Gemini Pro] 成功加载配置文件中的 API Key")
        except Exception as e:
            print(f"[Gemini Pro] 加载配置文件失败: {str(e)}")

    def save_config(self, api_key):
        """保存配置文件"""
        try:
            config = {'api_key': api_key}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print("[Gemini Pro] 成功保存 API Key 到配置文件")
        except Exception as e:
            print(f"[Gemini Pro] 保存配置文件失败: {str(e)}")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze the situation in details.", "multiline": True}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": True}),
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "model": (["gemini-2.5-pro-exp-03-25","gemini-2.5-flash-preview-05-20"], {"default": "gemini-2.5-flash-preview-05-20"}),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""}),
                "delay_time": (["0", "1", "2", "3", "5", "10"], {"default": "0"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "max_output_tokens": ("INT", {"default": 6000, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Gemini Pro"

    def generate(self, prompt, system_prompt, input_type, model, api_key, proxy, 
                delay_time="0", image=None, video=None, audio=None, 
                max_output_tokens=8192, temperature=0.4):
        try:
            # 在 API 调用前添加强制延迟 (这部分逻辑保留)
            delay_seconds = int(delay_time)
            if delay_seconds > 0:
                print(f"[Gemini Pro] 等待 {delay_seconds} 秒...")
                time.sleep(delay_seconds)

            # 处理 API Key (这部分逻辑保留)
            if api_key.strip():
                if api_key != self.api_key:
                    self.api_key = api_key
                    genai.configure(api_key=api_key)
                    self.save_config(api_key)
            elif not self.api_key:
                return ("错误: 请先输入 API Key",)
            
            # 处理代理设置 (这部分逻辑保留)
            if proxy and proxy.strip():
                proxy = proxy.strip()
                if not (proxy.startswith('http://') or proxy.startswith('https://')):
                    proxy = f"http://{proxy}"
                os.environ['http_proxy'] = proxy
                os.environ['https_proxy'] = proxy
                print(f"[Gemini Pro] 使用代理: {proxy}")
            else:
                if 'http_proxy' in os.environ:
                    del os.environ['http_proxy']
                if 'https_proxy' in os.environ:
                    del os.environ['https_proxy']
                print("[Gemini Pro] 不使用代理")

            print(f"\n[Gemini Pro] 正在调用 API...")
            print(f"[Gemini Pro] 模型: {model}")
            print(f"[Gemini Pro] 输入类型: {input_type}")
            print(f"[Gemini Pro] 系统引导词: {system_prompt}")
            print(f"[Gemini Pro] 提示词: {prompt}")
            
            # --- 准备输入内容 (这部分核心逻辑保留) ---
            content = [prompt] if prompt else []
            media_content = None
            if input_type == "text":
                if any([image, video, audio]):
                    return ("错误: 文本模式下不能提供多媒体输入",)
            elif input_type == "image" and image is not None:
                try:
                    if isinstance(image, torch.Tensor):
                        image = image.cpu()
                        if len(image.shape) == 4: image = image.squeeze(0)
                        if len(image.shape) == 3 and image.shape[0] == 3: image = image.permute(1, 2, 0)
                        image = (image * 255).round().clamp(0, 255).numpy().astype(np.uint8)
                    
                    if isinstance(image, np.ndarray):
                        if image.shape[-1] == 4: image = image[..., :3]
                        media_content = PIL.Image.fromarray(image)
                        content = [prompt, media_content] if prompt else [media_content]
                    else:
                        raise ValueError("Image must be tensor or numpy array")
                except Exception as e:
                    return (f"错误: 图片处理失败 - {str(e)}",)
            elif input_type == "video" and video is not None:
                # 此处视频和音频处理逻辑保持不变
                try:
                    if isinstance(video, str):
                        file_upload = GeminiFileUpload()
                        uploaded_file_result = file_upload.file_upload(video)
                        if isinstance(uploaded_file_result, tuple) and uploaded_file_result[0].startswith("错误"):
                            return uploaded_file_result
                        content = [prompt, uploaded_file_result[0]]
                    else:
                        return ("错误: 视频输入必须是文件路径",)
                except Exception as e:
                    return (f"错误: 视频处理失败 - {str(e)}",)
            elif input_type == "audio" and audio is not None:
                content = [prompt, audio] if prompt else [audio]
            elif input_type != "text":
                return (f"错误: {input_type} 类型需要提供对应的输入",)

            # --- 修改开始: 使用新的超时和重试逻辑 ---

            # 改进System Prompt的处理方式，并创建模型实例
            model_instance = genai.GenerativeModel(model, system_instruction=system_prompt)
            
            response = None
            last_error = "未知错误"
            max_retries = 2  # 总共尝试次数 = 1次初次尝试 + 1次重试

            for attempt in range(max_retries):
                try:
                    print(f"[Gemini Pro] 尝试第 {attempt + 1}/{max_retries} 次 API 调用...")
                    start_time = time.time()
                    
                    response = model_instance.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            candidate_count=1
                        ),
                        request_options={'timeout': 15}  # 核心修改：设置15秒网络超时
                    )
                    
                    end_time = time.time()
                    print(f"[Gemini Pro] API 调用成功，耗时: {end_time - start_time:.2f} 秒")
                    break  # 如果成功，立即跳出重试循环

                except Exception as e:
                    last_error = str(e)
                    print(f"[Gemini Pro] 第 {attempt + 1} 次尝试失败: {last_error}")
                    if attempt < max_retries - 1:
                        print("[Gemini Pro] 网络连接超时或错误，1秒后将进行最后一次重试...")
                        time.sleep(1)
                    else:
                        print("[Gemini Pro] 已达到最大重试次数，将返回错误。")
            
            # --- 修改结束 ---

            if response and response.text:
                print(f"[Gemini Pro] 生成完成，长度: {len(response.text)} 字符")
                return (response.text,)
            else:
                # 如果所有尝试都失败，返回最后一次捕获到的错误信息
                error_message = f"错误: API 未返回任何内容。最后一次错误: {last_error}"
                print(f"[Gemini Pro] {error_message}")
                return (error_message,)

        except Exception as e:
            # 兜底的异常捕获
            error_str = f"错误: 插件发生意外错误 - {str(e)}"
            print(f"[Gemini Pro] {error_str}")
            return (error_str,)

        finally:
            # 确保资源被正确释放 (这部分逻辑保留)
            if media_content and hasattr(media_content, 'close'):
                try:
                    media_content.close()
                except:
                    pass
            
            # 显式地进行一次垃圾收集
            import gc
            gc.collect()

class GeminiFileUpload:
    def __init__(self):
        self.api_key = None
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.temp_dir = os.path.join("I:", "ComfyUI_windows_portable", "ComfyUI", "temp")
        self.load_config()
        
        # 确保临时目录存在
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
                print(f"[Gemini文件上传] 创建临时目录: {self.temp_dir}")
            except Exception as e:
                print(f"[Gemini文件上传] 创建临时目录失败: {str(e)}")
                return ("错误: 无法创建临时目录",)
        
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
                    if self.api_key:
                        genai.configure(api_key=self.api_key)
                        print("[Gemini文件上传] 成功加载配置文件中的 API Key")
                    else:
                        print("[Gemini文件上传] 警告: 配置文件中没有找到有效的 API Key")
            else:
                print(f"[Gemini文件上传] 警告: 配置文件不存在 - {self.config_path}")
        except Exception as e:
            print(f"[Gemini文件上传] 加载配置文件失败: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "./sample.mp3", "multiline": False}),
            }
        }

    RETURN_TYPES = ("GEMINI_FILE",)
    RETURN_NAMES = ("file",)
    FUNCTION = "file_upload"

    CATEGORY = "Gemini Pro"

    def file_upload(self, file_path):
        try:
            if not self.api_key:
                return ("错误: 未在配置文件中找到有效的 API Key，请先在 config.json 中配置",)
                
            print(f"[Gemini文件上传] 正在处理文件: {file_path}")
            
            # 检查原始文件是否存在
            if not os.path.exists(file_path):
                print(f"[Gemini文件上传] 错误: 文件不存在 - {file_path}")
                return ("错误: 文件不存在",)
            
            # 生成随机文件名
            original_filename = os.path.basename(file_path)
            file_ext = os.path.splitext(original_filename)[1]
            random_filename = f"{int(time.time())}_{random.randint(1000, 9999)}{file_ext}"
            temp_file_path = os.path.join(self.temp_dir, random_filename)
            
            print(f"[Gemini文件上传] 创建临时文件: {temp_file_path}")
            
            # 复制文件到临时目录
            try:
                import shutil
                shutil.copy2(file_path, temp_file_path)
            except Exception as e:
                print(f"[Gemini文件上传] 复制文件到临时目录失败: {str(e)}")
                return (f"错误: 复制文件失败 - {str(e)}",)
            
            # 上传临时文件
            try:
                print(f"[Gemini文件上传] 上传临时文件: {temp_file_path}")
                uploaded_file = genai.upload_file(temp_file_path)
                print(f"[Gemini文件上传] 文件上传成功，ID: {uploaded_file.id if hasattr(uploaded_file, 'id') else '未知'}")
                
                # 保存原始文件路径信息（可选，如果后续需要）
                uploaded_file.original_filepath = file_path
                
                # 删除临时文件
                try:
                    os.remove(temp_file_path)
                    print(f"[Gemini文件上传] 临时文件已删除: {temp_file_path}")
                except Exception as e:
                    print(f"[Gemini文件上传] 警告: 删除临时文件失败 - {str(e)}")
                
                return (uploaded_file,)
            except Exception as e:
                # 上传失败，也尝试删除临时文件
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        print(f"[Gemini文件上传] 上传失败，临时文件已删除: {temp_file_path}")
                except:
                    pass
                
                print(f"[Gemini文件上传] 错误: 上传文件失败 - {str(e)}")
                return (f"错误: 上传文件失败 - {str(e)}",)
            
        except Exception as e:
            print(f"[Gemini文件上传] 错误: {str(e)}")
            return (f"错误: {str(e)}",)


class GeminiFileProcessing:
    def __init__(self):
        self.api_key = None
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
                    if self.api_key:
                        genai.configure(api_key=self.api_key)
                        print("[Gemini文件处理] 成功加载配置文件中的 API Key")
                    else:
                        print("[Gemini文件处理] 警告: 配置文件中没有找到有效的 API Key")
            else:
                print(f"[Gemini文件处理] 警告: 配置文件不存在 - {self.config_path}")
        except Exception as e:
            print(f"[Gemini文件处理] 加载配置文件失败: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("GEMINI_FILE",),
                "prompt": ("STRING", {"default": "分析这个文件内容并提供摘要。", "multiline": True}),
                "user_prompt": ("STRING", {"default": "你是一个专业的文件分析助手，请以专业、清晰的方式分析文件内容。", "multiline": True}),
                "model": (["gemini-2.5-pro-exp-03-25”,", "gemini-2.0-flash-exp","gemini-2.5-flash-preview-05-20"], {"default": "gemini-2.5-flash-preview-05-20"}),
                "stream": ("BOOLEAN", {"default": False}),
                "max_output_tokens": ("INT", {"default": 65536, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "proxy": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_content"
    OUTPUT_NODE = True
    CATEGORY = "Gemini Pro"

    def generate_content(self, file, prompt, user_prompt, model, stream, max_output_tokens=65536, temperature=0.4, proxy=""):
        try:
            if not self.api_key:
                return ("错误: 未在配置文件中找到有效的 API Key，请先在 config.json 中配置",)
                
            print(f"[Gemini文件处理] 正在处理文件...")
            print(f"[Gemini文件处理] 系统提示词: {user_prompt}")
            print(f"[Gemini文件处理] 提示词: {prompt}")
            print(f"[Gemini文件处理] 模型: {model}")
            
            # 处理代理设置 (逻辑保留)
            if proxy and proxy.strip():
                proxy = proxy.strip()
                if not (proxy.startswith('http://') or proxy.startswith('https://')):
                    proxy = f"http://{proxy}"
                os.environ['http_proxy'] = proxy
                os.environ['https_proxy'] = proxy
                print(f"[Gemini文件处理] 使用代理: {proxy}")
            else:
                if 'http_proxy' in os.environ:
                    del os.environ['http_proxy']
                if 'https_proxy' in os.environ:
                    del os.environ['https_proxy']
                print("[Gemini文件处理] 不使用代理")
                
            # --- 修改开始: 使用新的超时和重试逻辑 ---

            # 使用更现代的方式处理系统提示词，并创建模型实例
            model_instance = genai.GenerativeModel(model, system_instruction=user_prompt)
            
            # 准备请求内容
            content = [prompt, file]
            
            textoutput = ""
            last_error = "未知错误"
            max_retries = 2  # 总共尝试2次 (1次初次尝试 + 1次重试)

            for attempt in range(max_retries):
                try:
                    print(f"[Gemini文件处理] 尝试第 {attempt + 1}/{max_retries} 次 API 调用...")
                    start_time = time.time()
                    
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens
                    )
                    request_options = {'timeout': 15} # <--- 核心修改：为API调用设置15秒超时

                    if stream:
                        response_iterator = model_instance.generate_content(
                            content, 
                            stream=True,
                            generation_config=generation_config,
                            request_options=request_options
                        )
                        # 高效地处理流式响应
                        textoutput = "".join(chunk.text for chunk in response_iterator if hasattr(chunk, 'text'))
                    else:
                        response = model_instance.generate_content(
                            content,
                            generation_config=generation_config,
                            request_options=request_options
                        )
                        if hasattr(response, 'text'):
                            textoutput = response.text
                    
                    end_time = time.time()
                    print(f"[Gemini文件处理] API 调用成功，耗时: {end_time - start_time:.2f} 秒")
                    
                    if textoutput:
                        break # 如果成功获取到内容，立即跳出重试循环

                except Exception as e:
                    last_error = str(e)
                    print(f"[Gemini文件处理] 第 {attempt + 1} 次尝试失败: {last_error}")
                    # 针对文件状态错误给出特定提示并立即返回
                    if "not in an ACTIVE state" in last_error:
                        return ("错误: 文件状态无效。这可能是由于文件已被使用过，请重新上传文件。",)
                    
                    if attempt < max_retries - 1:
                        print("[Gemini文件处理] 网络连接超时或错误，1秒后将进行最后一次重试...")
                        time.sleep(1)
                    else:
                        print("[Gemini文件处理] 已达到最大重试次数，将返回错误。")
            
            # --- 修改结束 ---

            if textoutput:
                print(f"[Gemini文件处理] 生成完成，长度: {len(textoutput)} 字符")
                return (textoutput,)
            else:
                error_message = f"错误: API 未返回有效输出。最后一次错误: {last_error}"
                print(f"[Gemini文件处理] {error_message}")
                return (error_message,)
                
        except Exception as e:
            error_msg = f"错误: 插件发生意外错误 - {str(e)}"
            print(f"[Gemini文件处理] {error_msg}")
            return (error_msg,)
        
        finally:
            # 资源清理 (逻辑保留)
            import gc
            gc.collect()
            print("[Gemini文件处理] 资源清理完成")