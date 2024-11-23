import gradio as gr
from pydub import AudioSegment
from time import sleep

from pydub.utils import which

AudioSegment.converter = which(r"D:\Anaconda\envs\yolo_env\Lib\site-packages\ffmpeg\bin\ffmpeg.exe")
with gr.Blocks() as demo:
    input_audio = gr.Audio(label="Input Audio", type="filepath", format="mp3")
    with gr.Row():
        with gr.Column():
            stream_as_bytes_btn = gr.Button("Stream as Bytes")
            stream_as_bytes_output = gr.Audio(
                    label="Output Audio",
                    value=None,
                    streaming=True,
                    format="wav",
                    autoplay=True,
                    interactive=False,
                    waveform_options=gr.WaveformOptions(
                        sample_rate=24000
                    ))


            def stream_bytes(audio_file):
                chunk_size = 20_000

                with open(audio_file, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if chunk:
                            print(chunk)
                            yield chunk
                            sleep(1)
                        else:
                            break


            stream_as_bytes_btn.click(stream_bytes, input_audio, stream_as_bytes_output)

if __name__ == "__main__":
    demo.launch()
