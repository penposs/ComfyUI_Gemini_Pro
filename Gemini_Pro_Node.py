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
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "max_output_tokens": ("INT", {"default": 1000, "min": 1, "max": 5124}),
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
                    # 处理视频帧
                    if isinstance(video, torch.Tensor):
                        if len(video.shape) == 4 and video.shape[0] > 1:  # 多帧视频
                            frame_count = video.shape[0]
                            step = max(1, frame_count // 10)  # 最多采样10帧
                            frames = []
                            for i in range(0, frame_count, step):
                                frame = video[i]
                                # 转换单帧为PIL图像
                                if frame.shape[0] == 3:
                                    frame = frame.permute(1, 2, 0)
                                frame = (frame * 255).round().clamp(0, 255).cpu().numpy().astype(np.uint8)
                                pil_frame = PIL.Image.fromarray(frame)
                                # 调整帧大小到256x256
                                width, height = pil_frame.size
                                if width > height:
                                    if width > 256:
                                        height = int(256 * height / width)
                                        width = 256
                                else:
                                    if height > 256:
                                        width = int(256 * width / height)
                                        height = 256
                                pil_frame = pil_frame.resize((width, height), PIL.Image.LANCZOS)
                                frames.append(pil_frame)
                            
                            # 更新提示词和内容
                            content = [f"这是一个包含 {frame_count} 帧的视频。分析视频内容，注意帧之间的任何变化或动作："]
                            content.extend(frames)
                            if prompt:
                                content.append(prompt)
                        else:  # 单帧视频
                            frame = video.squeeze(0) if len(video.shape) == 4 else video
                            if frame.shape[0] == 3:
                                frame = frame.permute(1, 2, 0)
                            frame = (frame * 255).round().clamp(0, 255).cpu().numpy().astype(np.uint8)
                            pil_frame = PIL.Image.fromarray(frame)
                            
                            # 调整大小到最大1024像素
                            width, height = pil_frame.size
                            if width > height:
                                if width > 1024:
                                    height = int(1024 * height / width)
                                    width = 1024
                            else:
                                if height > 1024:
                                    width = int(1024 * width / height)
                                    height = 1024
                            pil_frame = pil_frame.resize((width, height), PIL.Image.LANCZOS)
                            
                            content = ["这是视频中的单个帧。分析图像内容：", pil_frame]
                            if prompt:
                                content.append(prompt)
                        
                        print(f"[Gemini Pro] 视频处理成功: {len(content)-1} 帧")
                    else:
                        raise ValueError("视频必须是tensor格式")
                        
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