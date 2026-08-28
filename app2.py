import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Sayfa ayarları
st.set_page_config(
    page_title="Meyve mi Sebze mi?",
    layout="centered"
)

st.title("Meyve mi  Sebze mi?")
st.write("Bir görsel yükleyin, modelimiz bunun meyve mi yoksa sebze mi olduğunu tahmin etsin!")

# 2. Modeli yükleme
@st.cache_resource
def load_fruit_veg_model():
    return tf.keras.models.load_model("fruit_vegetable_model.keras")

try:
    model = load_fruit_veg_model()
except Exception as e:
    st.error("Model yüklenemedi. Lütfen 'fruit_vegetable_model.keras' dosyasının aynı dizinde olduğundan emin olun.")
    st.stop()

# 3. Dosya yükleyici
uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi görüntüleme
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)
    
    st.write("Tahmin yapılıyor...")

    # 4. Ön işleme (Preprocessing)
    # Modelimizin eğitimdeki standartları: 128x128 boyut, RGB, [0, 1] arası normalizasyon
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    # 5. Tahmin alma
    prediction = model.predict(img_batch, verbose=0)[0][0]

    # Sigmoid çıktısı: 0.5 üzerindeyse Fruit (1), altındaysa Vegetables (0)
    if prediction > 0.5:
        label = "Meyve (Fruit)"
        confidence = prediction * 100
        st.success(f"**Tahmin:** {label}")
        st.info(f"**Güven Oranı:** %{confidence:.2f}")
    else:
        label = "Sebze (Vegetable)"
        confidence = (1 - prediction) * 100
        st.success(f"**Tahmin:** {label}")
        st.info(f"**Güven Oranı:** %{confidence:.2f}")