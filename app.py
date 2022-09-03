import gradio as gr
from model import Pipeline

pipeline = Pipeline()
labels = {"plastic": "플라스틱", "glass": "유리", "metal": "금속", "paper": "종이", "cardboard": "골판지", "trash": "쓰레기"}
models = {"ResNet152 (Pretrained, 2022-09-02)": "resnet152_0902", "ResNet152 (Pretrained, 2022-08-13)": "resnet152_0813", "ResNet50 (Pretrained, 2022-08-10)": "resnet50_0810"}
is_webcam = False

def predict(model, file_input, webcam_input):
    image = webcam_input if is_webcam else file_input
    if image == None:
        return [None, None, None]
    pred = pipeline.predict_image(models[model], image)
    return [{labels[key]: value for key, value in pred.data.items()}, f"{pred.duration}ms ({pred.duration / 1000}s)", pred.heatmap]

def set_input_type(value):
    global is_webcam
    if value == "파일":
        is_webcam = False
        return [gr.update(visible=True), gr.update(visible=False)]
    else:
        is_webcam = True
        return [gr.update(visible=False), gr.update(visible=True)]

with gr.Blocks(title="🌿 Trash AI") as demo:
    gr.Markdown('<h1 align="center">🌿 Trash AI</h1>')
    gr.Markdown('<p align="center">딥러닝 기반 쓰레기 이미지 분류 모델 데모</p>')

    with gr.Row():
        with gr.Column():
            model_select = gr.Dropdown(label="모델 선택", choices=list(models.keys()), value=list(models.keys())[0])
            input_select = gr.Dropdown(label="입력 유형", choices=["파일", "웹캠"], value="파일")
            file_input = gr.Image(label="입력 이미지", type="pil", source="upload")
            webcam_input = gr.Image(label="입력 이미지", type="pil", source="webcam", visible=False)
            with gr.Row():
                classify = gr.Button("분류")
            gr.Examples(["images/cocacola.jpg", "images/samdasoo.jpg", "images/sprite.jpg", "images/box.jpg", "images/tissue.jpg"], file_input)
        with gr.Column():
            output = gr.Label(label="결과")
            duration = gr.Label(label="소요 시간")
            heatmap = gr.Image(label="히트맵")

    input_select.change(set_input_type, [input_select], [file_input, webcam_input])
    classify.click(predict, [model_select, file_input, webcam_input], [output, duration, heatmap])

if __name__ == "__main__":
    demo.launch(share=True)