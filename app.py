import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('male_female_cnn_model.h5')

# Prediction function
def predict_gender(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    if prediction[0] > 0.5:
        label = "🧑 Male! "
        confidence = prediction[0][0] * 100
    else:
        label = "👩 Female! "
        confidence = (1 - prediction[0][0]) * 100

    return label, f"{confidence:.2f}% Confidence 🔍"

# Build the Gradio app with updated CSS
with gr.Blocks(css="""
    body {
        background-color: #FAFAFA;
        color: #222;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        margin-top: 35px;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #2c3e50;
    }

    p {
        text-align: center;
        color: #333;
        font-size: 1rem;
        margin-bottom: 10px;
    }

    .gr-button {
        background-color: #4B8BBE !important;
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .gr-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 12px #4B8BBE;
    }

    #output-label, #output-confidence {
        color: #1c1c1c !important;
        border: 2px solid #4B8BBE;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 12px;
        text-align: center;
    }


label:has(span:contains("Prediction 🎯")),
label:has(span:contains("Confidence 🔍")) {
    color: #2c5282; /* Custom blue color */
}

    .gr-image {
        background-color: #fff;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 10px;
    }

    .gr-image-preview {
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    hr {
        border: 1px solid #ccc;
        margin-bottom: 30px;
    }
""") as demo:
    gr.Markdown(
        """
        <h1>🕵️ Gender Prediction Model</h1>
        <p>Upload a face image and let our AI guess the gender — Male or Female!</p>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📤 Upload an Image", image_mode="RGB")
            predict_btn = gr.Button("🚀 Predict Gender")

        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Prediction 🎯", interactive=False, elem_id="output-label")
            output_confidence = gr.Textbox(label="Confidence 🔍", interactive=False, elem_id="output-confidence")

    predict_btn.click(fn=predict_gender, inputs=image_input, outputs=[output_label, output_confidence])

demo.launch()