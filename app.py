import os

import gradio as gr
import uvicorn

from large_wav import create_app
from processor import AudioProcessor


def process_audio(input_file, progress=gr.Progress()):
    if input_file is None:
        return None, "音声ファイルをアップロードしてください。"

    try:
        processor = AudioProcessor()
        output_file = processor.process(
            input_file,
            progress_callback=lambda value, description: progress(value, desc=description),
        )
        return output_file, "処理が完了しました。WAVファイルを再生・ダウンロードできます。"
    except Exception as exc:
        return None, f"処理に失敗しました: {exc}"


with gr.Blocks(title="Spectral Lifter", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# Spectral Lifter (v1.2)")
    gr.Markdown(
        "Suno AIなどの生成音源に含まれる高域の欠落や、"
        "シュワシュワしたシマーノイズを補正します。"
    )
    gr.Markdown(
        "**使い方:** 音声をアップロードして「音声を処理」を押してください。"
        "6分以内の音源に対応しています。処理には曲の長さに応じて数分かかる場合があります。"
        " **大容量WAV:** [専用アップロード画面](/large-wav)をご利用ください。"
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="入力音声")
            process_btn = gr.Button("音声を処理", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(type="filepath", label="処理後の音声（WAV）")
            status_text = gr.Textbox(label="処理状況", interactive=False)

    process_btn.click(fn=process_audio, inputs=audio_input, outputs=[audio_output, status_text])
    gr.Markdown(
        "アップロードされた音声と生成ファイルは一時保存され、定期的に削除されます。"
        "大切な原本は必ず手元に保管してください。"
    )

if __name__ == "__main__":
    queued_demo = demo.queue(default_concurrency_limit=1, max_size=5)
    server_app = create_app(queued_demo)
    uvicorn.run(
        server_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
    )
