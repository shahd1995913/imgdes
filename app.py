import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
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
    st.title("BLIP Image Captioning App")
    st.write("Upload an image, and the model will generate a caption.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the PIL image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

        # Generate caption
        description = generate_caption(image_bytes.getvalue())

        # Display the generated caption
        st.subheader("Generated Caption:")
        st.write(description)

if __name__ == "__main__":
    main()
