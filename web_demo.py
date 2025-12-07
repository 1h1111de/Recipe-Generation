import copy
import warnings
import os
import json
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
from transformers import TextIteratorStreamer
from threading import Thread

# 解决torch.dtype JSON序列化问题的兜底配置
class TorchDtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(obj)
original_dumps = json.dumps
def custom_dumps(*args, **kwargs):
    kwargs['cls'] = TorchDtypeEncoder
    return original_dumps(*args, **kwargs)
json.dumps = custom_dumps

# 日志初始化
logger = logging.get_logger(__name__)

# 配置项
MODEL_PATH = "./InternVL2-2B-Receipe1"  # 模型路径
LOGO_PATH = "logo.png"                 # Logo路径
DEFAULT_IMAGE_SIZE = 448               # 图像预处理尺寸
DEFAULT_MAX_TILES = 4                  # 默认图像块数量

# Logo图片兜底处理
if os.path.exists(LOGO_PATH):
    logo = Image.open(LOGO_PATH)
else:
    logo = Image.new('RGB', (200, 80), color=(240, 240, 240))
    draw = ImageDraw.Draw(logo)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((10, 30), "InternVL2-2B", font=font, fill=(0, 0, 0))

# 图像预处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """构建图像预处理变换"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """寻找最接近的图像比例"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态预处理图像（InternVL2核心逻辑）"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """加载并预处理图像"""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_upload_file_and_show(uploaded_file, max_num=DEFAULT_MAX_TILES):
    """处理上传的图片并返回模型输入的pixel_values（统一float16类型）"""
    pixel_values = None
    if uploaded_file is not None:
        # 核心修复：张量类型改为float16，与模型保持一致
        pixel_values = load_image(uploaded_file, max_num=max_num).to(torch.float16).cuda()
    return pixel_values

@dataclass
class GenerationConfig:
    """生成配置类"""
    max_length: int = 2048
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000

@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    pixel_values,
    generation_config=None,
):
    """交互式生成回答（流式输出，延长超时时间）"""
    if generation_config is None:
        generation_config = {}
    
    # 核心修复：延长streamer超时时间到5分钟，避免推理超时
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True, 
        timeout=300
    )
    
    # 生成配置
    gen_config = dict(
        max_new_tokens=generation_config.get('max_length', 1024),
        do_sample=generation_config.get('do_sample', False),
        temperature=generation_config.get('temperature', 0.1),
        top_p=generation_config.get('top_p', 0.75),
        repetition_penalty=generation_config.get('repetition_penalty', 1.0),
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    )
    
    # 启动线程执行模型推理
    thread = Thread(target=model.chat, kwargs=dict(
        tokenizer=tokenizer, 
        pixel_values=pixel_values, 
        question=prompt,
        history=None, 
        return_history=False, 
        generation_config=gen_config
    ))
    thread.start()
    
    generated_text = ''
    try:
        for new_text in streamer:
            # 检查对话结束标记
            if hasattr(model, 'conv_template') and new_text == model.conv_template.sep:
                break
            generated_text += new_text
            yield generated_text
    except Exception as e:
        yield f"流式输出出错：{str(e)}"

def on_btn_click():
    """清空聊天历史的回调函数"""
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'pixel_values' in st.session_state:
        st.session_state.pixel_values = None
    if 'uploaded_image' in st.session_state:
        del st.session_state.uploaded_image
    st.session_state.uploader_key += 1

@st.cache_resource
def load_model():
    """加载单模型（缓存资源，统一float16类型）"""
    # 核心修复：使用torch_dtype参数，模型类型改为float16
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    
    # 强制模型所有参数转为float16，避免类型残留
    model = model.to(dtype=torch.float16)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    return model, tokenizer

def main():
    """主函数：构建Streamlit界面"""
    # 初始化会话状态
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'pixel_values' not in st.session_state:
        st.session_state.pixel_values = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # 加载模型和分词器
    model, tokenizer = load_model()

    # 侧边栏配置
    with st.sidebar:
        st.image(logo, caption='InternVL2-2B-Receipe')
        st.divider()
        
        # 语言选择
        lan = st.selectbox(
            '#### 界面语言 / Language', 
            ['中文', 'English'], 
            help='仅切换UI显示语言 / Only switch UI display language'
        )
        
        # 高级生成选项
        with st.expander('🔥 高级生成选项 / Advanced Options'):
            temperature = st.slider('温度 / Temperature', 0.0, 1.0, 0.7, 0.1)
            top_p = st.slider('Top-P', 0.0, 1.0, 0.95, 0.05)
            repetition_penalty = st.slider('重复惩罚 / Repetition Penalty', 1.0, 1.5, 1.1, 0.02)
            max_length = st.slider('最大输出长度 / Max New Tokens', 512, 4096, 1024, 128)
            # 核心修复：减少图像块数范围，默认4
            max_input_tiles = st.slider('图像块数 / Max Input Tiles', 1, 8, 4, 1)
        
        # 清空历史按钮
        st.button('清空聊天历史 / Clear History', on_click=on_btn_click, type='primary')
        st.divider()

        # 图片上传器
        uploaded_image = st.file_uploader(
            '上传食物图片 / Upload Image',
            type=['png', 'jpg', 'jpeg', 'webp'],
            key=f'uploader_{st.session_state.uploader_key}',
            help='上传你想查询的食物图片，支持PNG/JPG/WEBP格式' if lan == '中文' else 'Upload food image (PNG/JPG/WEBP)'
        )

        # 处理上传的图片（传递动态图像块数）
        if uploaded_image is not None:
            st.session_state.pixel_values = load_upload_file_and_show(uploaded_image, max_num=max_input_tiles)
            st.session_state.uploaded_image = uploaded_image
            # 预览上传的图片
            st.image(uploaded_image, caption='已上传图片 / Uploaded Image', use_column_width=True)

    # 主界面标题和提示语
    if lan == "中文":
        st.title('🍲 食谱大模型 - InternVL2-2B')
        sys_prompt = "您好，我是食谱大模型🍲，您可以上传食物图片并输入问题，我会为您分析制作方法！"
        chat_placeholder = st.chat_input('请输入你的问题（如：这道菜怎么做？）...')
    else:
        st.title('🍲  Recipe Generation - InternVL2-2B')
        sys_prompt = "Hello, I am the Cuisine Recipe Model 🍲. Upload an image and ask a question, I will analyze the making method for you!"
        chat_placeholder = st.chat_input('Type your question (e.g., How to make this dish?)...')

    # 初始化聊天历史
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'role': 'robot',
            'content': sys_prompt
        }]
    else:
        st.session_state.messages[0]["content"] = sys_prompt

    # 生成配置
    generation_config = {
        'max_length': max_length,
        'top_p': top_p,
        'temperature': temperature,
        'do_sample': temperature > 0,
        'repetition_penalty': repetition_penalty
    }

    # 展示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if "image" in message.keys():
                st.image(message['image'], caption='', use_column_width=True)

    # 处理用户文本输入
    if chat_placeholder:
        user_prompt = chat_placeholder.strip()
        if not user_prompt:
            st.warning('请输入有效问题！' if lan == '中文' else 'Please enter a valid question!')
            st.stop()

        # 检查是否有图片
        pixel_values = st.session_state.pixel_values
        if pixel_values is None:
            st.warning('请先上传食物图片！' if lan == '中文' else 'Please upload a food image first!')
            st.stop()

        # 展示用户消息
        with st.chat_message('user'):
            st.markdown(user_prompt)
            if st.session_state.uploaded_image is not None:
                st.image(st.session_state.uploaded_image, caption='', use_column_width=True)

        # 构建用户消息字典
        user_message = {'role': 'user', 'content': user_prompt}
        if st.session_state.uploaded_image is not None:
            user_message['image'] = st.session_state.uploaded_image

        # 添加用户消息到历史
        st.session_state.messages.append(user_message)

        # 模型推理并展示回答
        with st.chat_message('robot'):
            message_placeholder = st.empty()
            final_response = ""
            # 流式输出回答
            try:
                for cur_response in generate_interactive(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=user_prompt,
                        pixel_values=pixel_values,
                        generation_config=generation_config
                ):
                    final_response = cur_response
                    message_placeholder.markdown(final_response + '▌')
                # 最终展示回答
                message_placeholder.markdown(final_response)
            except Exception as e:
                error_msg = f"生成回答时出错: {str(e)}" if lan == '中文' else f"Error generating response: {str(e)}"
                st.error(error_msg)
                final_response = error_msg

        # 添加机器人消息到历史
        st.session_state.messages.append({
            'role': 'robot',
            'content': final_response
        })

if __name__ == '__main__':
    # 设置页面配置
    st.set_page_config(
        page_title='食谱大模型-InternVL2-2B',
        page_icon='🍲',
        layout='wide'
    )
    # 禁用PyTorch的CUDA内存分配警告（可选）
    torch.cuda.empty_cache()
    main()