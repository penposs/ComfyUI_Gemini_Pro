import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import tempfile
from io import BytesIO
from google import genai
from google.genai import types
import time
import traceback

class GeminiProEditimage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp“,”models/gemini-2.5-flash-preview-05-20"], {"default": "models/gemini-2.5-flash-preview-05-20"}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 88888, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Gemini-Pro"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        # 默认图像尺寸
        self.default_width = 1024
        self.default_height = 1024
        
        # 检查google-genai版本
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"当前google-genai版本: {genai_version}")
            
            # 检查版本是否满足最低要求
            from packaging import version
            if version.parse(genai_version) < version.parse('0.8.0'):  # 用实际需要的最低版本替换
                self.log("警告: google-genai版本过低，建议升级到最新版本")
                self.log("建议执行: pip install -q -U google-genai")
        except Exception as e:
            self.log(f"无法检查google-genai版本: {e}")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥，其次从config.json读取"""
        # 如果用户输入了有效的密钥，使用该密钥
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到config.json文件中
            try:
                config_data = {}
                if os.path.exists(self.config_file):
                    with open(self.config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                
                config_data["api_key"] = user_input_key
                
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                    
                self.log("已保存API密钥到config.json")
            except Exception as e:
                self.log(f"保存API密钥到config.json失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从config.json文件读取
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    
                if "api_key" in config_data and config_data["api_key"] and len(config_data["api_key"]) > 10:
                    self.log("使用config.json中的API密钥")
                    return config_data["api_key"]
            except Exception as e:
                self.log(f"读取config.json中的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥，请在节点中输入API密钥或在config.json中配置")
        return ""
    
    def generate_empty_image(self, width=None, height=None):
        """生成标准格式的空白RGB图像张量 - 确保ComfyUI兼容格式 [B,H,W,C]"""
        # 使用默认尺寸（如果未指定）
        if width is None:
            width = self.default_width
        if height is None:
            height = self.default_height
            
        # 创建一个符合ComfyUI标准的图像张量
        # ComfyUI期望 [batch, height, width, channels] 格式!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def resize_keep_aspect_ratio(self, img, target_width, target_height, mode='fit'):
        """
        调整图像大小，同时保持纵横比
        
        参数:
        - img: PIL Image对象
        - target_width: 目标宽度
        - target_height: 目标高度
        - mode: 调整模式，'fit'确保图像完全适合目标尺寸，'fill'确保填满目标尺寸，'long_edge'调整长边
        
        返回: 调整后的PIL Image对象
        """
        # 获取原始图像的宽度和高度
        original_width, original_height = img.width, img.height
        self.log(f"原始图像尺寸: {original_width}x{original_height}, 目标尺寸: {target_width}x{target_height}")
        
        # 计算原始图像的宽高比
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        # 如果原始和目标宽高比非常接近，直接调整为目标尺寸
        aspect_threshold = 0.01  # 1%的宽高比差异
        if abs(original_aspect - target_aspect) < aspect_threshold:
            self.log(f"宽高比接近匹配，直接调整为目标尺寸")
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        if mode == 'fit':
            # 确保图像完全适合目标框而不裁剪
            if original_aspect > target_aspect:  # 原图更宽
                # 按宽度调整
                new_width = target_width
                new_height = int(new_width / original_aspect)
                self.log(f"按宽度适配，新尺寸: {new_width}x{new_height}")
            else:  # 原图更高
                # 按高度调整
                new_height = target_height
                new_width = int(new_height * original_aspect)
                self.log(f"按高度适配，新尺寸: {new_width}x{new_height}")
        
        elif mode == 'fill':
            # 确保填满目标框，可能裁剪部分图像
            if original_aspect > target_aspect:  # 原图更宽
                # 按高度调整
                new_height = target_height
                new_width = int(new_height * original_aspect)
                self.log(f"按高度填充，新尺寸: {new_width}x{new_height}，需裁剪宽度")
            else:  # 原图更高
                # 按宽度调整
                new_width = target_width
                new_height = int(new_width / original_aspect)
                self.log(f"按宽度填充，新尺寸: {new_width}x{new_height}，需裁剪高度")
        
        elif mode == 'long_edge':
            # 按长边调整
            if original_width >= original_height:
                # 宽度是长边
                new_width = target_width
                new_height = int(new_width / original_aspect)
                self.log(f"调整长边(宽)，新尺寸: {new_width}x{new_height}")
            else:
                # 高度是长边
                new_height = target_height
                new_width = int(new_height * original_aspect)
                self.log(f"调整长边(高)，新尺寸: {new_width}x{new_height}")
        
        # 调整图像大小
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 如果是fill模式并且尺寸不匹配，需要居中裁剪
        if mode == 'fill' and (new_width != target_width or new_height != target_height):
            left = (new_width - target_width) // 2 if new_width > target_width else 0
            top = (new_height - target_height) // 2 if new_height > target_height else 0
            right = left + target_width if new_width > target_width else new_width
            bottom = top + target_height if new_height > target_height else new_height
            
            # 裁剪并填充（如果需要）
            crop_img = resized_img.crop((left, top, right, bottom))
            if crop_img.width != target_width or crop_img.height != target_height:
                # 需要填充
                final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                paste_x = (target_width - crop_img.width) // 2
                paste_y = (target_height - crop_img.height) // 2
                final_img.paste(crop_img, (paste_x, paste_y))
                self.log(f"裁剪并填充，最终尺寸: {final_img.width}x{final_img.height}")
                return final_img
            
            self.log(f"裁剪后尺寸: {crop_img.width}x{crop_img.height}")
            return crop_img
        
        # 对于fit和long_edge模式，或者正好匹配的fill模式，直接返回调整后的图像
        return resized_img
    
    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None
                
            self.log(f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}")
            
            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None
                
            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()
                
            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None
    
    def save_tensor_as_image(self, image_tensor, file_path):
        """将图像张量保存为文件"""
        try:
            # 转换为numpy数组
            if torch.is_tensor(image_tensor):
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]  # 获取批次中的第一张图像
                
                # [C, H, W] -> [H, W, C]
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor
            
            # 缩放到0-255
            image_np = (image_np * 255).astype(np.uint8)
            
            # 创建PIL图像
            pil_image = Image.fromarray(image_np)
            
            # 保存到文件
            pil_image.save(file_path, format="PNG")
            self.log(f"已保存图像到: {file_path}")
            return True
        except Exception as e:
            self.log(f"图像保存错误: {str(e)}")
            return False
    
    def process_image_data(self, image_data, width, height):
        """处理API返回的图像数据，返回ComfyUI格式的图像张量 [B,H,W,C]"""
        try:
            # 打印图像数据类型和大小以便调试
            self.log(f"图像数据类型: {type(image_data)}")
            self.log(f"图像数据长度: {len(image_data) if hasattr(image_data, '__len__') else '未知'}")
            
            # 尝试直接转换为PIL图像
            try:
                pil_image = Image.open(BytesIO(image_data))
                self.log(f"成功打开图像, 尺寸: {pil_image.width}x{pil_image.height}, 模式: {pil_image.mode}")
            except Exception as e:
                self.log(f"无法直接打开图像数据: {e}")
                
                # 尝试其他方式解析，例如base64解码
                try:
                    # 检查是否是base64编码的字符串
                    if isinstance(image_data, str):
                        # 尝试移除base64前缀
                        if "base64," in image_data:
                            image_data = image_data.split("base64,")[1]
                        decoded_data = base64.b64decode(image_data)
                        pil_image = Image.open(BytesIO(decoded_data))
                    else:
                        # 如果是向量或其他格式，生成一个占位图像
                        self.log("无法解析图像数据，创建一个空白图像")
                        return self.generate_empty_image()
                except Exception as e2:
                    self.log(f"备用解析方法也失败: {e2}")
                    return self.generate_empty_image()
            
            # 确保图像是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                self.log(f"图像已转换为RGB模式")
            
            # 调整图像大小
            if pil_image.width != width or pil_image.height != height:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                self.log(f"图像已调整为目标尺寸: {width}x{height}")
            
            # 关键修复: 使用ComfyUI兼容的格式 [batch, height, width, channels]
            # 而不是PyTorch标准的 [batch, channels, height, width]
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self.log(f"生成的图像张量格式: 形状={img_tensor.shape}, 类型={img_tensor.dtype}")
            return (img_tensor,)
            
        except Exception as e:
            self.log(f"处理图像数据时出错: {e}")
            traceback.print_exc()
            return self.generate_empty_image()
    
    def generate_image(self, prompt, api_key, model, temperature, seed=66666666, image=None):
        """生成图像 - 使用简化的API密钥管理"""
        temp_img_path = None
        response_text = ""
        
        # 使用默认尺寸
        width = self.default_width
        height = self.default_height
        
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(), full_text)
            
            # 创建客户端实例
            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 构建简单提示
            simple_prompt = f"Create a detailed image of: {prompt}"
            
            # 配置生成参数，使用用户指定的温度值
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}")
            
            # 处理参考图像
            contents = []
            has_reference = False
            
            if image is not None:
                try:
                    # 确保图像格式正确
                    if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] 格式
                        # 获取第一帧图像
                        input_image = image[0].cpu().numpy()
                        
                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        # 保存为临时文件
                        temp_img_path = os.path.join(tempfile.gettempdir(), f"reference_{int(time.time())}.png")
                        pil_image.save(temp_img_path)
                        
                        self.log(f"参考图像处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 读取图像数据
                        with open(temp_img_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # 添加图像部分和文本部分
                        img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                        txt_part = {"text": simple_prompt + " Use this reference image as style guidance."}
                        
                        # 组合内容(图像在前，文本在后)
                        contents = [img_part, txt_part]
                        has_reference = True
                        self.log("参考图像已添加到请求中")
                    else:
                        self.log(f"参考图像格式不正确: {image.shape}")
                        contents = simple_prompt
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    contents = simple_prompt
            else:
                # 没有参考图像，只使用文本
                contents = simple_prompt
            
            # 打印请求信息
            self.log(f"请求Gemini API生成图像，种子值: {seed}, 包含参考图像: {has_reference}")
            
            # 调用API
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=gen_config
            )
            
            # 响应处理
            self.log("API响应接收成功，正在处理...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("API响应中没有candidates")
                # 合并日志和返回值
                full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                return (self.generate_empty_image(), full_text)
            
            # 检查响应中是否有图像
            image_found = False
            
            # 遍历响应部分
            for part in response.candidates[0].content.parts:
                # 检查是否为文本部分
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API返回文本片段，长度: {len(text_content)} 字符")
                    # 只在日志中截断显示，不影响实际返回的文本
                    self.log(f"文本内容预览: {text_content[:100]}..." if len(text_content) > 100 else f"文本内容: {text_content}")
                
                # 检查是否为图像部分
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("API返回数据解析处理")
                    try:
                        # 记录图像数据信息以便调试
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                        self.log(f"图像数据类型: {type(image_data)}, MIME类型: {mime_type}, 数据长度: {len(image_data) if image_data else 0}")
                        
                        # 确认数据不为空且长度足够
                        if not image_data or len(image_data) < 100:
                            self.log("警告: 图像数据为空或太小")
                            continue
                        
                        # 尝试检查数据的前几个字节，确认是否为有效的图像格式
                        is_valid_image = False
                        if len(image_data) > 8:
                            # 检查常见图像格式的魔法字节
                            magic_bytes = image_data[:8]
                            self.log(f"图像魔法字节(十六进制): {magic_bytes.hex()[:16]}...")
                            # PNG 头部是 \x89PNG\r\n\x1a\n
                            if magic_bytes.startswith(b'\x89PNG'):
                                self.log("检测到有效的PNG图像格式")
                                is_valid_image = True
                            # JPEG 头部是 \xff\xd8
                            elif magic_bytes.startswith(b'\xff\xd8'):
                                self.log("检测到有效的JPEG图像格式")
                                is_valid_image = True
                            # GIF 头部是 GIF87a 或 GIF89a
                            elif magic_bytes.startswith(b'GIF87a') or magic_bytes.startswith(b'GIF89a'):
                                self.log("检测到有效的GIF图像格式")
                                is_valid_image = True
                        
                        if not is_valid_image:
                            self.log("警告: 数据可能不是有效的图像格式")
                        
                        # 多种方法尝试打开图像
                        pil_image = None
                        
                        # 方法1: 直接用PIL打开
                        try:
                            pil_image = Image.open(BytesIO(image_data))
                            self.log(f"方法1成功: 直接使用PIL打开图像, 尺寸: {pil_image.width}x{pil_image.height}")
                        except Exception as e1:
                            self.log(f"方法1失败: {str(e1)}")
                            
                            # 方法2: 保存到临时文件再打开
                            try:
                                temp_file = os.path.join(tempfile.gettempdir(), f"gemini_image_{int(time.time())}.png")
                                with open(temp_file, "wb") as f:
                                    f.write(image_data)
                                self.log(f"已保存图像数据到临时文件: {temp_file}")
                                
                                pil_image = Image.open(temp_file)
                                self.log(f"方法2成功: 通过临时文件打开图像")
                            except Exception as e2:
                                self.log(f"方法2失败: {str(e2)}")
                                
                                # 方法3: 尝试修复数据头再打开
                                try:
                                    # 如果MIME类型是PNG但数据头不正确，尝试添加正确的PNG头
                                    if mime_type == "image/png" and not image_data.startswith(b'\x89PNG'):
                                        self.log("尝试修复PNG头部")
                                        fixed_data = b'\x89PNG\r\n\x1a\n' + image_data[8:] if len(image_data) > 8 else image_data
                                        pil_image = Image.open(BytesIO(fixed_data))
                                        self.log("方法3成功: 通过修复头部打开PNG图像")
                                    # 如果MIME类型是JPEG但数据头不正确，尝试添加正确的JPEG头
                                    elif mime_type == "image/jpeg" and not image_data.startswith(b'\xff\xd8'):
                                        self.log("尝试修复JPEG头部")
                                        fixed_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00' + image_data[20:] if len(image_data) > 20 else image_data
                                        pil_image = Image.open(BytesIO(fixed_data))
                                        self.log("方法3成功: 通过修复头部打开JPEG图像")
                                except Exception as e3:
                                    self.log(f"方法3失败: {str(e3)}")
                                    
                                    # 方法4: 尝试使用base64解码再打开
                                    try:
                                        if isinstance(image_data, bytes):
                                            # 尝试将bytes转换为字符串并进行base64解码
                                            str_data = image_data.decode('utf-8', errors='ignore')
                                            if 'base64,' in str_data:
                                                base64_part = str_data.split('base64,')[1]
                                                decoded_data = base64.b64decode(base64_part)
                                                pil_image = Image.open(BytesIO(decoded_data))
                                                self.log("方法4成功: 通过base64解码打开图像")
                                    except Exception as e4:
                                        self.log(f"方法4失败: {str(e4)}")
                                        
                                        # 所有方法都失败，跳过这个数据
                                        self.log("所有图像处理方法都失败，无法处理返回的数据")
                                        continue
                        
                        # 确保图像已成功加载
                        if pil_image is None:
                            self.log("无法打开图像，跳过")
                            continue
                            
                        # 确保图像是RGB模式
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                            self.log(f"图像已转换为RGB模式")
                        
                        # 调整图像尺寸，保持纵横比
                        pil_image = self.resize_keep_aspect_ratio(pil_image, width, height, mode='fit')
                        self.log(f"图像已调整为自适应尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 转换为ComfyUI格式
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        
                        self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                        image_found = True
                        
                        # 合并日志和API返回文本
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                        return (img_tensor, full_text)
                    except Exception as e:
                        self.log(f"图像处理错误: {e}")
                        traceback.print_exc()  # 添加详细的错误追踪信息
            
            # 没有找到图像数据，但可能有文本
            if not image_found:
                self.log("API响应中未找到图像数据，仅返回文本")
                if not response_text:
                    response_text = "API未返回任何图像或文本"
                else:
                    self.log(f"返回文本总长度: {len(response_text)} 字符")
            
            # 合并日志和API返回文本
            # 确保API返回部分显示完整的文本
            full_text = "## API返回 (长度: " + str(len(response_text)) + " 字符)\n" + response_text + "\n\n## 处理日志\n" + "\n".join(self.log_messages)
            
            # 记录最终返回文本的长度
            self.log(f"最终返回文本总长度: {len(full_text)} 字符")
            
            return (self.generate_empty_image(), full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")
            
            # 合并日志和错误信息
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (self.generate_empty_image(), full_text)

class GeminiProChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash", "models/gemini-2.0-pro", "models/gemini-1.5-pro"], {"default": "models/gemini-2.0-flash"}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "context": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "chat"
    CATEGORY = "Gemini-Pro"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        # 检查google-genai版本
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"当前google-genai版本: {genai_version}")
            
            # 检查版本是否满足最低要求
            from packaging import version
            if version.parse(genai_version) < version.parse('0.8.0'):
                self.log("警告: google-genai版本过低，建议升级到最新版本")
                self.log("建议执行: pip install -q -U google-genai")
        except Exception as e:
            self.log(f"无法检查google-genai版本: {e}")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥，其次从config.json读取"""
        # 如果用户输入了有效的密钥，使用该密钥
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到config.json文件中
            try:
                config_data = {}
                if os.path.exists(self.config_file):
                    with open(self.config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                
                config_data["api_key"] = user_input_key
                
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                    
                self.log("已保存API密钥到config.json")
            except Exception as e:
                self.log(f"保存API密钥到config.json失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从config.json文件读取
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    
                if "api_key" in config_data and config_data["api_key"] and len(config_data["api_key"]) > 10:
                    self.log("使用config.json中的API密钥")
                    return config_data["api_key"]
            except Exception as e:
                self.log(f"读取config.json中的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥，请在节点中输入API密钥或在config.json中配置")
        return ""
    
    def generate_empty_image(self, width=512, height=512):
        """生成标准格式的空白RGB图像张量 - 确保ComfyUI兼容格式 [B,H,W,C]"""
        # 创建一个符合ComfyUI标准的图像张量
        # ComfyUI期望 [batch, height, width, channels] 格式!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def process_image_parts(self, parts):
        """处理API返回的多个图像部分，返回图像列表"""
        image_tensors = []
        
        for part in parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                try:
                    # 获取图像数据
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                    
                    if not image_data or len(image_data) < 100:
                        self.log("警告: 图像数据为空或太小，跳过")
                        continue
                    
                    # 尝试打开图像
                    try:
                        pil_image = Image.open(BytesIO(image_data))
                        self.log(f"成功打开图像, 尺寸: {pil_image.width}x{pil_image.height}, 模式: {pil_image.mode}")
                    except Exception as e:
                        self.log(f"无法打开图像: {e}")
                        continue
                    
                    # 确保图像是RGB模式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # 转换为ComfyUI格式 [B,H,W,C]
                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    
                    self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                    image_tensors.append(img_tensor)
                    
                except Exception as e:
                    self.log(f"处理图像部分时出错: {e}")
                    traceback.print_exc()
        
        return image_tensors
    
    def combine_image_tensors(self, image_tensors):
        """将多个图像张量合并为一个批次"""
        if not image_tensors:
            self.log("没有有效的图像，返回空白图像")
            return self.generate_empty_image()
        
        # 如果只有一张图像，直接返回
        if len(image_tensors) == 1:
            return image_tensors[0]
        
        # 合并多张图像为一个批次
        try:
            # 确保所有图像具有相同的高度和宽度
            heights = [img.shape[1] for img in image_tensors]
            widths = [img.shape[2] for img in image_tensors]
            
            self.log(f"合并 {len(image_tensors)} 张图像，高度: {heights}, 宽度: {widths}")
            
            # 使用torch.cat沿批次维度合并
            combined = torch.cat(image_tensors, dim=0)
            self.log(f"合并后的张量形状: {combined.shape}")
            return combined
            
        except Exception as e:
            self.log(f"合并图像张量时出错: {e}")
            traceback.print_exc()
            # 出错时返回第一张图像
            return image_tensors[0] if image_tensors else self.generate_empty_image()
    
    def chat(self, prompt, api_key, model, temperature, seed=66666666, context=None, image=None):
        """与Gemini模型聊天，处理返回的文本和图像"""
        temp_img_path = None
        response_text = ""
        
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(), full_text)
            
            # 创建客户端实例
            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 配置生成参数
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}，模型: {model}")
            
            # 处理内容
            contents = []
            
            # 处理上下文
            if context and context.strip():
                self.log("添加上下文到对话")
                contents.append({"text": context})
            
            # 处理参考图像
            if image is not None:
                try:
                    # 确保图像格式正确
                    if len(image.shape) == 4:  # [B, H, W, 3] 格式
                        # 处理每一帧图像
                        for i in range(image.shape[0]):
                            # 获取单帧图像
                            input_image = image[i].cpu().numpy()
                            
                            # 转换为PIL图像
                            input_image = (input_image * 255).astype(np.uint8)
                            pil_image = Image.fromarray(input_image)
                            
                            # 保存为临时文件
                            temp_img_path = os.path.join(tempfile.gettempdir(), f"reference_{int(time.time())}_{i}.png")
                            pil_image.save(temp_img_path)
                            
                            self.log(f"参考图像 {i+1} 处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                            
                            # 读取图像数据
                            with open(temp_img_path, "rb") as f:
                                image_bytes = f.read()
                            
                            # 添加图像部分
                            img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                            contents.append(img_part)
                    else:
                        self.log(f"参考图像格式不正确: {image.shape}")
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
            
            # 添加用户提示
            contents.append({"text": prompt})
            
            # 打印请求信息
            self.log(f"请求Gemini API，模型: {model}, 种子值: {seed}")
            
            # 调用API
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )
            
            # 响应处理
            self.log("API响应接收成功，正在处理...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("API响应中没有candidates")
                full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                return (self.generate_empty_image(), full_text)
            
            # 处理文本和图像
            text_parts = []
            image_tensors = []
            
            # 遍历响应部分
            for part in response.candidates[0].content.parts:
                # 处理文本部分
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    text_parts.append(text_content)
                    self.log(f"API返回文本: {text_content[:100]}..." if len(text_content) > 100 else text_content)
            
            # 处理所有图像部分
            image_tensors = self.process_image_parts(response.candidates[0].content.parts)
            
            # 合并所有文本
            response_text = "\n".join(text_parts)
            
            # 合并所有图像
            combined_images = self.combine_image_tensors(image_tensors)
            
            # 检查是否有图像
            if not image_tensors:
                self.log("API响应中未找到图像数据")
                if not response_text:
                    response_text = "API未返回任何图像或文本"
            else:
                self.log(f"成功处理 {len(image_tensors)} 张图像")
            
            # 合并日志和API返回文本
            full_text = "## API返回\n" + response_text + "\n\n## 处理日志\n" + "\n".join(self.log_messages)
            return (combined_images, full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini聊天错误: {str(e)}")
            traceback.print_exc()
            
            # 合并日志和错误信息
            full_text = "## 错误\n" + error_message + "\n\n## 处理日志\n" + "\n".join(self.log_messages)
            return (self.generate_empty_image(), full_text)

# 更新节点注册
NODE_CLASS_MAPPINGS = {
    "Gemini-Pro-Editimage": GeminiProEditimage,
    "Gemini-Pro-Chat": GeminiProChat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini-Pro-Editimage": "Gemini Pro Editimage",
    "Gemini-Pro-Chat": "Gemini Pro Chat"
} 
