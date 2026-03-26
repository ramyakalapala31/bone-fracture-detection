
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Bone Fracture Detection")

# Load trained model
model = YOLO("/content/runs/detect/bone_fracture_model/weights/best.pt")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    results = model(temp_file.name)

    for r in results:
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            st.success("Fracture Detected")

            for box in boxes:
                conf = float(box.conf[0])
                st.write(f"Confidence: {conf:.2f}")
        else:
            st.success("No Fracture Detected")

        im_array = r.plot()
        st.image(im_array, caption="Detection Result", width="stretch")
