import os, argparse
import sys
import gradio as gr
from scripts.gradio.i2v_test_application import Image2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))


i2v_examples_interp_512 = [
    ['prompts/512_interp/74906_1462_frame1.png', 'walking man', 50, 7.5, 1.0, 10, 123, 'prompts/512_interp/74906_1462_frame3.png'],
    ['prompts/512_interp/Japan_v2_2_062266_s2_frame1.png', 'an anime scene', 50, 7.5, 1.0, 10, 789, 'prompts/512_interp/Japan_v2_2_062266_s2_frame3.png'],
    ['prompts/512_interp/Japan_v2_3_119235_s2_frame1.png', 'an anime scene', 50, 7.5, 1.0, 10, 123, 'prompts/512_interp/Japan_v2_3_119235_s2_frame3.png'],
]




def dynamicrafter_demo(result_dir='./tmp/', res=512):
    if res == 1024:
        resolution = '576_1024'
        css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height:576px}"""
    elif res == 512:
        resolution = '320_512'
        css = """#input_img {max-width: 512px !important} #output_vid {max-width: 512px; max-height: 320px} #input_img2 {max-width: 512px !important} #output_vid {max-width: 512px; max-height: 320px}"""
    elif res == 256:
        resolution = '256_256'
        css = """#input_img {max-width: 256px !important} #output_vid {max-width: 256px; max-height: 256px}"""
    else:
        raise NotImplementedError(f"Unsupported resolution: {res}")
    image2video = Image2Video(result_dir, resolution=resolution)
    with gr.Blocks(analytics_enabled=False, css=css) as dynamicrafter_iface:



        with gr.Tab(label='ToonCrafter_320x512'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image = gr.Image(label="Input Image1",elem_id="input_img")
                        with gr.Row():
                            i2v_input_text = gr.Text(label='Prompts')
                        with gr.Row():
                            i2v_seed = gr.Slider(label='Random Seed', minimum=0, maximum=50000, step=1, value=123)
                            i2v_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="i2v_eta")
                            i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5, elem_id="i2v_cfg_scale")
                        with gr.Row():
                            i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                            i2v_motion = gr.Slider(minimum=5, maximum=30, step=1, elem_id="i2v_motion", label="FPS", value=10)
                        i2v_end_btn = gr.Button("Generate")
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image2 = gr.Image(label="Input Image2",elem_id="input_img2")
                        with gr.Row():
                            i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)

                gr.Examples(examples=i2v_examples_interp_512,
                            inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed, i2v_input_image2],
                            outputs=[i2v_output_video],
                            fn = image2video.get_image,
                            cache_examples=False,
                )
            i2v_end_btn.click(inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed, i2v_input_image2],
                            outputs=[i2v_output_video],
                            fn = image2video.get_image
            )


    return dynamicrafter_iface

def get_parser():
    parser = argparse.ArgumentParser()
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_dir = os.path.join('./', 'results')
    dynamicrafter_iface = dynamicrafter_demo(result_dir)
    dynamicrafter_iface.queue(max_size=12)
    dynamicrafter_iface.launch(max_threads=1)
    # dynamicrafter_iface.launch(server_name='0.0.0.0', server_port=80, max_threads=1)