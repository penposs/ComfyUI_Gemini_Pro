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
                "model": (["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"], {"default": "gemini-1.5-flash"}),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""}),
                "delay_time": (["0", "1", "2", "3", "5", "10"], {"default": "0"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "max_output_tokens": ("INT", {"default": 1000, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Gemini Pro"

    def generate(self, prompt, system_prompt, input_type, model, api_key, proxy, 
                delay_time="0", image=None, video=None, audio=None, 
                max_output_tokens=1000, temperature=0.4):
        try:
            # 在 API 调用前添加强制延迟
            delay_seconds = int(delay_time)
            if delay_seconds > 0:
                print(f"[Gemini Pro] 等待 {delay_seconds} 秒...")
                time.sleep(delay_seconds)

            # 处理 API Key
            if api_key.strip():  # 如果用户输入了新的 API Key
                if api_key != self.api_key:  # 如果与当前的不同
                    self.api_key = api_key
                    genai.configure(api_key=api_key)
                    self.save_config(api_key)  # 保存到配置文件
            elif not self.api_key:  # 如果没有输入且没有保存的 API Key
                return ("错误: 请先输入 API Key",)
            
            # 设置代理
            if proxy:
                os.environ['http_proxy'] = proxy
                os.environ['https_proxy'] = proxy
                print(f"[Gemini Pro] 使用代理: {proxy}")

            print(f"\n[Gemini Pro] 正在调用 API...")
            print(f"[Gemini Pro] 模型: {model}")
            print(f"[Gemini Pro] 输入类型: {input_type}")
            print(f"[Gemini Pro] 系统引导词: {system_prompt}")
            print(f"[Gemini Pro] 提示词: {prompt}")
            
            start_time = time.time()

            # 准备输入内容
            content = [prompt] if prompt else []
            
            # 根据输入类型处理
            media_content = None
            if input_type == "text":
                if any([image, video, audio]):
                    return ("错误: 文本模式下不能提供多媒体输入",)
            elif input_type == "image" and image is not None:
                try:
                    # 转换图片格式
                    if isinstance(image, torch.Tensor):
                        # 确保图像是在 CPU 上
                        image = image.cpu()
                        # 如果是 4D tensor，去掉第一个维度
                        if len(image.shape) == 4:
                            image = image.squeeze(0)
                        # 如果是 3D tensor，调整维度顺序
                        if len(image.shape) == 3:
                            # 确保通道顺序是 (H, W, C)
                            if image.shape[0] == 3:
                                image = image.permute(1, 2, 0)
                        # 转换为 uint8 格式
                        image = (image * 255).round().clamp(0, 255)
                        # 转换为 numpy array
                        image = image.numpy().astype(np.uint8)
                    
                    # 如果是 numpy array
                    if isinstance(image, np.ndarray):
                        # 确保形状正确
                        if len(image.shape) == 3:
                            # 确保是 RGB 格式
                            if image.shape[-1] == 4:  # RGBA
                                image = image[..., :3]  # 只保留 RGB
                            elif image.shape[-1] != 3:
                                raise ValueError(f"Unexpected image shape: {image.shape}")
                        else:
                            raise ValueError(f"Unexpected image dimensions: {len(image.shape)}")
                        
                        # 转换为 PIL Image
                        media_content = PIL.Image.fromarray(image)
                        
                        print(f"[Gemini Pro] 图片处理成功: {media_content.size}")
                    else:
                        raise ValueError("Image must be tensor or numpy array")
                        
                    # 如果有媒体内容，将其与提示词组合
                    if media_content:
                        content = [prompt, media_content] if prompt else [media_content]
                        
                except Exception as e:
                    print(f"[Gemini Pro] 图片处理错误: {str(e)}")
                    print(f"[Gemini Pro] 图片类型: {type(image)}")
                    print(f"[Gemini Pro] 图片形状: {image.shape if hasattr(image, 'shape') else 'unknown'}")
                    return (f"错误: 图片处理失败 - {str(e)}",)
            elif input_type == "video" and video is not None:
                try:
                    # 处理视频文件
                    if isinstance(video, str):  # 视频文件路径
                        # 使用GeminiFileUpload类上传视频文件
                        file_upload = GeminiFileUpload()
                        uploaded_file = file_upload.file_upload(video)
                        
                        if isinstance(uploaded_file, tuple) and uploaded_file[0].startswith("错误"):
                            return uploaded_file
                            
                        # 构建视频分析提示
                        video_prompt = f"这是一个视频文件。请分析视频内容，注意视频中的动作、场景变化和关键事件：\n{prompt}"
                        content = [video_prompt, uploaded_file[0]]
                        
                    else:
                        return ("错误: 视频输入必须是文件路径",)
                        
                    print(f"[Gemini Pro] 视频处理成功")
                except Exception as e:
                    print(f"[Gemini Pro] 视频处理错误: {str(e)}")
                    return (f"错误: 视频处理失败 - {str(e)}",)
            elif input_type == "audio" and audio is not None:
                media_content = audio
                content = [prompt, media_content] if prompt else [media_content]
            else:
                return (f"错误: {input_type} 类型需要提供对应的输入",)

            # 创建模型实例并生成内容
            model_instance = genai.GenerativeModel(model)
            
            # 添加系统提示词（如果有）
            if system_prompt:
                print(f"[Gemini Pro] 使用系统提示词: {system_prompt}")
                content.insert(0, f"System: {system_prompt}\nUser: ")
            # 添加基础延迟和重试逻辑
            base_delay = 3  # 增加基础延迟到 3 秒
            max_retries = 5  # 增加最大重试次数到 5 次
            
            # 每次 API 调用前添加基础延迟
            time.sleep(base_delay)
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = base_delay * (2 ** attempt)  # 增加退避时间
                        print(f"[Gemini Pro] 配额限制，等待 {delay} 秒后重试...")
                        time.sleep(delay)
                    
                    response = model_instance.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            candidate_count=1
                        )
                    )
                    break  # 如果成功，跳出重试循环
                except Exception as e:
                    if ("429" in str(e) or "Resource has been exhausted" in str(e)) and attempt < max_retries - 1:
                        print(f"[Gemini Pro] 配额限制，正在重试 ({attempt + 1}/{max_retries})")
                        continue
                    raise  # 如果是其他错误或已达到最大重试次数，则抛出异常
            max_retries = 3
            base_delay = 2  # 基础延迟时间（秒）
            # 修改重试逻辑，添加总体超时控制
            max_retries = 3
            base_delay = 2
            start_time = time.time()
            timeout = 30  # 设置30秒超时
            for attempt in range(max_retries):
                try:
                    current_time = time.time()
                    if current_time - start_time > timeout:
                        raise TimeoutError("API 调用超时（超过30秒）")
                    if attempt > 0:
                        delay = min(base_delay * (2 ** (attempt - 1)), timeout - (current_time - start_time))
                        if delay <= 0:
                            raise TimeoutError("API 调用超时（超过30秒）")
                        print(f"[Gemini Pro] 等待 {delay:.1f} 秒后重试...")
                        time.sleep(delay)
                    
                    response = model_instance.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            candidate_count=1
                        )
                    )
                    break  # 如果成功，跳出重试循环
                except TimeoutError as te:
                    print(f"[Gemini Pro] {str(te)}")
                    return (f"错误: {str(te)}",)
                except Exception as e:
                    if "ResourceExhausted" in str(e) and attempt < max_retries - 1:
                        print(f"[Gemini Pro] 资源不足，正在重试 ({attempt + 1}/{max_retries})")
                        continue
                    raise  # 如果是其他错误或已达到最大重试次数，则抛出异常

            end_time = time.time()
            print(f"[Gemini Pro] API 调用耗时: {end_time - start_time:.2f} 秒")

            if response.text:
                print(f"[Gemini Pro] 生成完成，长度: {len(response.text)} 字符")
                return (response.text,)
            else:
                print("[Gemini Pro] 错误: API 返回为空")
                return ("错误: API 返回为空",)

        except Exception as e:
            print(f"[Gemini Pro] 错误: {str(e)}")
            return (f"错误: {str(e)}",)

class GeminiFileUpload:
    def __init__(self):
        self.api_key = None
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        # 在插件根目录下创建临时文件夹
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
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
                "model": (["gemini-1.5-pro", "gemini-2.0-pro-exp", "gemini-2.0-flash-exp"], {"default": "gemini-1.5-pro"}),
                "stream": ("BOOLEAN", {"default": False}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_content"
    OUTPUT_NODE = True
    CATEGORY = "Gemini Pro"

    def generate_content(self, file, prompt, user_prompt, model, stream, max_output_tokens=8192, temperature=0.4):
        try:
            if not self.api_key:
                return ("错误: 未在配置文件中找到有效的 API Key，请先在 config.json 中配置",)
                
            print(f"[Gemini文件处理] 正在处理文件")
            print(f"[Gemini文件处理] 系统提示词: {user_prompt}")
            print(f"[Gemini文件处理] 提示词: {prompt}")
            print(f"[Gemini文件处理] 模型: {model}")
                
            # 创建模型实例
            model_instance = genai.GenerativeModel(model)
            
            # 添加基础延迟和重试逻辑
            base_delay = 3  # 基础延迟3秒
            max_retries = 5  # 最大重试次数5次
            
            # 调用API前的延迟
            time.sleep(base_delay)
            
            # 合并用户提示词和提示词
            combined_prompt = f"System: {user_prompt}\nUser: {prompt}"
            
            # 发送请求并处理流式/非流式响应
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = base_delay * (2 ** attempt)  # 指数退避
                        print(f"[Gemini文件处理] 配额限制，等待 {delay} 秒后重试...")
                        time.sleep(delay)
                    
                    start_time = time.time()
                    
                    if stream:
                        print("[Gemini文件处理] 使用流式输出模式")
                        response = model_instance.generate_content(
                            [combined_prompt, file], 
                            stream=True,
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=max_output_tokens
                            )
                        )
                        textoutput = "\n".join([chunk.text for chunk in response])
                    else:
                        print("[Gemini文件处理] 使用标准输出模式")
                        response = model_instance.generate_content(
                            [combined_prompt, file],
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=max_output_tokens
                            )
                        )
                        textoutput = response.text
                    
                    end_time = time.time()
                    print(f"[Gemini文件处理] API 调用耗时: {end_time - start_time:.2f} 秒")
                    print(f"[Gemini文件处理] 生成完成，长度: {len(textoutput)} 字符")
                    
                    return (textoutput,)
                except Exception as e:
                    if "ResourceExhausted" in str(e) and attempt < max_retries - 1:
                        print(f"[Gemini文件处理] 资源不足，正在重试 ({attempt + 1}/{max_retries})")
                        continue
                    elif attempt < max_retries - 1:
                        print(f"[Gemini文件处理] 错误: {str(e)}, 正在重试 ({attempt + 1}/{max_retries})")
                        continue
                    raise  # 其他错误或已达到最大重试次数，抛出异常
            
            # 如果所有重试都失败
            return ("错误: API调用失败，已重试最大次数",)
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Gemini文件处理] 错误: {error_msg}")
            
            # 如果出现文件状态错误，给用户更清晰的提示
            if "not in an ACTIVE state" in error_msg:
                return ("错误: 文件状态无效。这可能是由于文件已被使用过，请重新上传文件。",)
            
            return (f"错误: {error_msg}",)