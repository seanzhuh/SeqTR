import os
import argparse
import gradio as gr

try:
    from constants import *
    from api import ReferenceProxy, scan_dir, STATIC_PATH
except ImportError:
    from .constants import *
    from .api import ReferenceProxy, scan_dir, STATIC_PATH

# static variable
CHECKPOINT_DIR = "ckpt"
CONFIG_DIR = "configs/seqtr"
CONFIG_MAP = {
    'Detection': 'detection',
    'Segmentation': 'segmentation',
    'Mixed': 'multi-task'
}
MODE_MAP = {
    'Detection': 'det',
    'Segmentation': 'seg',
    'Mixed': 'mixed'
}
MODE_MAP_REVERSE = {
    'det': 'Detection',
    'seg': 'Segmentation',
    'mixed': 'Mixed'
}
LANG_EMBED_DIR = STATIC_PATH

# global variable
proxy = ReferenceProxy(args={'max_token': 20, 'ema': False})
## default
default_lang_embed_path = scan_dir(LANG_EMBED_DIR, '')
default_config_list = scan_dir(os.path.join(CONFIG_DIR, CONFIG_MAP["Detection"]), '.py')
default_ckpt_list = scan_dir(CHECKPOINT_DIR, '.pth')
## dynamic
working_mode = MODE_MAP_REVERSE[proxy.get_mode()]
model_config_file = 'seqtr_det_refcocog-umd.py'
model_ckpt_file = default_ckpt_list[0]
lang_embed_path = default_lang_embed_path[-1]

# event handler / callback
def change_working_mode(choice):
    global working_mode
    working_mode = choice
    model_config.choices = scan_dir(os.path.join(CONFIG_DIR, CONFIG_MAP[working_mode]), '.py')
    return gr.Dropdown(choices=model_config.choices, label="Select config", value=model_config.choices[0], interactive=True, scale=2)

change_config_file = lambda choice: globals().update({'model_config_file': choice})
change_ckpt_file = lambda choice: globals().update({'model_ckpt_file': choice})
change_lang_embed = lambda choice: globals().update({'lang_embed_path': choice})

def load_model_callback():
    proxy.load_config(os.path.join(CONFIG_DIR, CONFIG_MAP[working_mode], model_config_file),
                      os.path.join(CHECKPOINT_DIR, model_ckpt_file),
                      lang_embed_path,
                      MODE_MAP[working_mode]
                      )
    gr.Info(message="Model loaded successfully!")
    gr.Info(message="Current config: {}".format(model_config_file))
    gr.Info(message="Current checkpoint: {}".format(model_ckpt_file))

def inference_button_callback(image_input=None, text_input=None):
    if image_input is None:
        raise gr.Error("Please upload image first!")
    if text_input is None:
        raise gr.Error("Please input referring text first!")
    if not proxy.check_health():
        raise gr.Error("Model has an unsolved error yet! ({})".format(proxy.check_health()))
    gr.Info(message='Inferencing on "{}" mode'.format(working_mode))
    return proxy.inference(image_input, text_input)
    
def reset_button_callback():
    return [None, '', None]
    
with gr.Blocks(theme=gr.themes.Soft(), head=HEAD) as demo:
    gr.HTML(value=HEADER)

    with gr.Group():
        gr.HTML('<div style="background-color: #6668E9; padding: 10px;"><h3 style="color: #ffffff; text-align: center; margin: 0px; ">Model Selection</h6></div>')
        with gr.Row():
            work_mode = gr.Radio(choices=['Detection', 'Segmentation', 'Mixed'], label="Working mode", value=working_mode, interactive=True, scale=1)
            model_config = gr.Dropdown(choices=default_config_list, label="Select model config", value=model_config_file, interactive=True, scale=2)
        with gr.Row():
            model_ckpt = gr.Dropdown(choices=default_ckpt_list, label="Select checkpoint", value=default_ckpt_list[0], interactive=True, scale=2)
            word_embed = gr.Dropdown(choices=default_lang_embed_path, label="Select word embedding", value=default_lang_embed_path[-1], interactive=True, scale=2)
            load_button = gr.Button("Reload model", variant="secondary")
    # bind event handler
    work_mode.change(fn=change_working_mode, inputs=work_mode, outputs=model_config)
    model_config.change(fn=change_config_file, inputs=model_config)
    model_ckpt.change(fn=change_ckpt_file, inputs=model_ckpt)
    word_embed.change(fn=change_lang_embed, inputs=word_embed)
    load_button.click(fn=load_model_callback)
    
    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.HTML('<div style="background-color: #6668E9; padding: 10px;"><h3 style="color: #ffffff; text-align: center; margin: 0px; ">Input Control</h6></div>')
                image_input = gr.Image(type="pil", label="Upload Image", height="400px")
                text_input = gr.Textbox(label="Referring text", placeholder="Please input your descrption...", lines=7)
        with gr.Column():
            with gr.Group():
                gr.HTML('<div style="background-color: #6668E9; padding: 10px;"><h3 style="color: #ffffff; text-align: center; margin: 0px; ">Inference Result</h6></div>')
                output_image = gr.Image(type="pil", label="Result", interactive=False, height="570px")
            with gr.Row():
                infer_button = gr.Button("Run inference", variant="primary")
                reset_button = gr.Button("Reset", variant="stop", size="lg")
    # bind event handler
    infer_button.click(fn=inference_button_callback, inputs=[image_input, text_input], outputs=output_image)
    reset_button.click(fn=reset_button_callback, outputs=[image_input, text_input, output_image])



def parse_args():
    parser = argparse.ArgumentParser(description='SeqTR demo')
    parser.add_argument('--config_dir', default='configs/seqtr/', help='config directory path')
    parser.add_argument('--ckpt_dir', default='ckpt/', help='checkpoint directory path')
    parser.add_argument('--device', default='cuda', help='device used for inference')
    parser.add_argument('--host', default='0.0.0.0', help='listening host address')
    parser.add_argument('--port', type=int, default=7869, help='port number for webui')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    CONFIG_DIR = args.config_dir
    CHECKPOINT_DIR = args.ckpt_dir
    load_model_callback()
    demo.launch(share=True, server_name=args.host, server_port=args.port)
    
