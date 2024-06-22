from ultralytics import YOLO
import gradio as gr
import numpy as np
import supervision as sv
from tqdm import tqdm
import subprocess
import cv2
import os
import shutil
import time

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
DEFAULT_OUTPUT_PATH = os.getcwd()+"/tmp/gradio/"

def ui_update(btn):
    return gr.update(interactive=True)

def open_directory(path=None):
    if path is None:
        return
    try:
        os.startfile(path)
    except:
        subprocess.Popen(["xdg-open", path])

def process_categories(model, categories):
    input_classes = [category.strip() for category in categories.split(',')]
    model_classes = model.names
    classes_list = []
    for i in range(len(input_classes)):
        try:
            classes_list.append(list(model_classes.keys())[list(model_classes.values()).index(input_classes[i])])
        except:
            raise gr.Error("Enter some classes or class is not in model")
    return classes_list

def annotate_image(model, image, detections):
    labels = [
        (
            f"{model.names[class_id]}: {confidence:.3f}"
        )
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image

def calculate_end_frame_index(source_video_path):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return video_info.total_frames

def convert_mpeg_to_h264(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libopenh264',
        '-y',
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

def process_image(source_image, model, classes, conf_thresh, iou_thresh, progress=gr.Progress(track_tqdm=True)):
    start_time = time.time()
    model = YOLO(model)
    classes = process_categories(model, classes)
    with sv.CSVSink(DEFAULT_OUTPUT_PATH+"results.csv") as csv_sink:
        result = model.predict(source_image, conf=conf_thresh, iou=iou_thresh, classes=classes)[0]
        detections = sv.Detections.from_ultralytics(result)
        output_image = annotate_image(model, source_image, detections)
        cv2.imwrite(DEFAULT_OUTPUT_PATH+"output.png", output_image)
        csv_sink.append(detections, {})
    print(f"Время на исполнение: {(time.time() - start_time)}")
    return output_image

def process_video(source_video, model, classes, conf_thresh, iou_thresh, progress=gr.Progress(track_tqdm=True)):
    start_time = time.time()
    model = YOLO(model)
    classes = process_categories(model, classes)
    video_info = sv.VideoInfo.from_video_path(source_video)
    total = int(video_info.total_frames)
    frame_generator = sv.get_video_frames_generator(source_path=source_video, end=total)
    result_file_path = DEFAULT_OUTPUT_PATH+"demo.mp4"
    with sv.VideoSink(result_file_path, video_info=video_info) as sink, sv.CSVSink(DEFAULT_OUTPUT_PATH+"results.csv") as csv_sink:
        for _ in tqdm(range(total), desc="Processing video..."):
            frame = next(frame_generator)
            results = model.predict(frame, conf=conf_thresh, iou=iou_thresh, classes=classes)[0]
            detections = sv.Detections.from_ultralytics(results)
            frame = annotate_image(model, frame, detections)
            sink.write_frame(frame)
            csv_sink.append(detections, {})
    result_file_path = convert_mpeg_to_h264(result_file_path, DEFAULT_OUTPUT_PATH+"demo1.mp4")
    os.remove(DEFAULT_OUTPUT_PATH+"demo.mp4")
    print(f"Время на исполнение: {(time.time() - start_time)}")
    return result_file_path


css = """

div.gradio-container{
    max-width: unset !important;
}

footer{
    display:none !important
}

#slider_row {
display: flex;
flex-wrap: wrap;
justify-content: space-between;
}

#refresh_slider {
flex: 0 1 20%;
display: flex;
align-items: center;
}

#frame_slider {
flex: 1 0 80%;
display: flex;
align-items: center;
}

"""

with gr.Blocks(css=css, theme=gr.themes.Default()) as interface:
    gr.Markdown("Выпускная квалификационная работа")
    gr.Markdown("Программа для анализа потока машин")

    with gr.Tabs():
        with gr.TabItem("Input Image"):
            with gr.Row():
                with gr.Column(scale=0.35):
                    source_image = gr.Image(
                        label="Source Image",
                        interactive=True,
                        type='numpy'
                    )
                    classes_image = gr.Textbox(
                        lines=2,
                        label='Enter the classes to be detected, '
                        'separated by comma',
                        elem_id='textbox'
                    )
                    with gr.Row():
                        submit_image = gr.Button('Submit')
                        clear_image = gr.Button('Clear')
                with gr.Column(scale=0.65):
                    output_image = gr.Image(
                        label="Output Image",
                        interactive=False
                    )
                    with gr.Row():
                            output_image_button = gr.Button(
                                "📂", interactive=False)
                            output_image_button.click(
                                lambda: open_directory(path=DEFAULT_OUTPUT_PATH),
                                inputs=None,
                                outputs=None,
                            )


        with gr.TabItem("Input Video"):
            with gr.Row():
                with gr.Column(scale=0.35):
                    source_video = gr.Video(
                        label="Source Video",
                        interactive=True,
                    )
                    classes_video = gr.Textbox(
                        lines=2,
                        label='Enter the classes to be detected, '
                        'separated by comma',
                        elem_id='textbox'
                        )
                    with gr.Row():
                        submit_video = gr.Button('Submit')
                        clear_video = gr.Button('Clear')

                with gr.Column(scale=0.65):
                        output_video = gr.Video(
                                    label="Output Video"
                                )
                        with gr.Row():
                            output_video_button = gr.Button(
                                "📂", interactive=False)
                            output_video_button.click(
                                lambda: open_directory(path=DEFAULT_OUTPUT_PATH),
                                inputs=None,
                                outputs=None,
                            )


        with gr.TabItem("Preferences"):
            models = gr.Dropdown(
                label="Model",
                choices=[
                    "yolov8n.pt",
                    "yolov8s.pt",
                    "yolov8m.pt",
                    "yolov8l.pt",
                    "yolov8x.pt"
                ],
                value="yolov8n.pt"
            )
            max_num_boxes = gr.Slider(minimum=1,
                            maximum=300,
                            value=100,
                            step=1,
                            interactive=True,
                            label='Maximum Number Boxes')
            conf_thr = gr.Slider(minimum=0,
                                maximum=1.0,
                                value=0.5,
                                step=0.01,
                                interactive=True,
                                label='Confidence Threshold')
            IoU_thr = gr.Slider(minimum=0,
                                maximum=1.0,
                                value=0.5,
                                step=0.01,
                                interactive=True,
                                label='IoU Threshold')

        submit_image.click(fn=process_image, inputs=[source_image, models, classes_image, conf_thr, IoU_thr], outputs=[output_image]).then(
            fn=ui_update, inputs=submit_image, outputs=output_image_button)
        clear_image.click(lambda: [None, '', None], None, [source_image, classes_image, output_image])
        submit_video.click(fn=process_video, inputs=[source_video, models, classes_video, conf_thr, IoU_thr], outputs=[output_video]).then(
            fn=ui_update, inputs=submit_video, outputs=output_video_button)
        clear_video.click(lambda: [None, '', None], None, [source_video, classes_video, output_video])

def clear_temp_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))

folder = 'tmp/gradio'  # Замените на ваш путь к папке

clear_temp_folder(folder)
if __name__ == "__main__":
    interface.launch(share=False, debug=True)