import gradio as gr
import requests
import json

# API 服务器地址（与你的 LMDeploy 一致）
API_URL = "http://localhost:23333/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def model_chat(message, history, max_tokens=512, temperature=0.7):
    """调用 API 服务器实现对话，history 为 Gradio 聊天历史"""
    # 构造请求参数
    payload = {
        "model": "InternVL2-2B-Receipe1",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        # 发送请求并获取回复
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()  # 捕获 HTTP 错误
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"请求失败：{str(e)}"

# 构建 Gradio 界面
with gr.Blocks(title="InternVL2 菜谱模型") as demo:
    gr.Markdown("# InternVL2-2B-Receipe 菜谱生成模型")
    chatbot = gr.Chatbot(height=600, label="对话窗口")
    msg = gr.Textbox(label="输入问题", placeholder="请输入你想查询的菜谱问题...")
    with gr.Row():
        send_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空历史")
    # 可调节的生成参数
    max_tokens = gr.Slider(128, 1024, 512, label="最大生成Token数")
    temperature = gr.Slider(0.1, 1.0, 0.7, label="生成温度（值越大越随机）")

    # 绑定按钮事件
    send_btn.click(model_chat, [msg, chatbot, max_tokens, temperature], chatbot)
    msg.submit(model_chat, [msg, chatbot, max_tokens, temperature], chatbot)  # 按回车发送
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# 启动 Gradio 服务
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部机器访问
        server_port=6006,       # 前端端口，与你原计划一致
        share=False             # 若需公网临时链接，设为 True
    )