import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    # Preprocess the image and generate caption
    inputs = processor(images=image, return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

def main():
    st.title("Image Captioning with BLIP Model")

    # File Upload
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Generate caption
        description = generate_caption(image)

        # Display the caption
        st.write("Description:", description)

if __name__ == '__main__':
    main()
