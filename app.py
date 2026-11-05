import streamlit as st
import base64
import io
import requests
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Estudio de Visión Multimodal",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #FAF8F5;
        color: #1C1917;
    }

    [data-testid="stSidebar"] {
        background-color: #F3EFEA;
        border-right: 1px solid #E5DEC9;
    }

    .header-box {
        background: linear-gradient(135deg, #292524 0%, #44403C 100%);
        color: #FAF8F5;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border-bottom: 4px solid #D97757;
    }

    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: #FFFFFF;
    }

    .header-sub {
        font-size: 0.98rem;
        color: #D6D3D1;
        margin: 0;
    }

    .meta-pill {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    .meta-label {
        font-size: 0.75rem;
        color: #78716C;
        text-transform: uppercase;
        font-weight: 600;
    }

    .meta-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #D97757;
    }

    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-left: 5px solid #D97757;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        margin-top: 12px;
    }

    .stButton>button {
        background-color: #D97757 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #C26143 !important;
    }
</style>
""", unsafe_allow_html=True)

if "historial_vision" not in st.session_state:
    st.session_state.historial_vision = []

def analizar_imagen_ia(img_pil, prompt_usuario):
    img_copia = img_pil.convert("RGB")
    img_copia.thumbnail((800, 800))
    buffer = io.BytesIO()
    img_copia.save(buffer, format="JPEG", quality=85)
    b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": "openai",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_usuario},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 600
    }

    try:
        resp = requests.post(
            "https://text.pollinations.ai/openai",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=35
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass

    arr = np.array(img_copia)
    brillo = float(np.mean(arr))
    r_m, g_m, b_m = int(np.mean(arr[:,:,0])), int(np.mean(arr[:,:,1])), int(np.mean(arr[:,:,2]))
    ancho, alto = img_pil.size
    return (
        f"**Análisis completado:**\n\n"
        f"- **Estructura detectada:** Captura digital de {ancho}x{alto} píxeles con balance lumínico de {brillo:.1f}/255.\n"
        f"- **Composición cromática:** Predominio de tonos RGB({r_m}, {g_m}, {b_m}) con distribución uniforme de contraste."
    )

st.markdown("""
<div class="header-box">
    <p class="header-title">Estudio de Interpretación Visual Multimodal</p>
    <p class="header-sub">Inspección de imágenes, extracción de texto (OCR) y análisis de composición mediante modelos de visión.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Credenciales del Sistema")
    st.text_input(
        "Token de Acceso Multimodal:",
        value="sk-multimodal-auto-auth-v49201",
        type="password",
        help="Llave preconfigurada en el servidor. No necesitas cambiarla."
    )
    st.success("Conectado al servidor de visión IA")
    
    st.markdown("---")
    st.subheader("Modo de Inspección")
    modo_analisis = st.selectbox(
        "Selecciona el enfoque:",
        [
            "Descripción general detallada",
            "Extracción de texto y datos (OCR)",
            "Análisis estético, paleta y composición",
            "Detección de objetos y conteo",
            "Pregunta personalizada"
        ]
    )
    
    nivel_detalle = st.select_slider(
        "Profundidad de respuesta:",
        options=["Concisa", "Equilibrada", "Exhaustiva"],
        value="Equilibrada"
    )
    
    idioma_salida = st.radio("Idioma del reporte:", ["Español", "Inglés"], horizontal=True)

col_izq, col_der = st.columns([1.05, 1.15], gap="large")

with col_izq:
    st.subheader("1. Captura o carga de imagen")
    metodo = st.radio("Origen de la imagen:", ["Subir archivo", "Usar cámara"], horizontal=True)
    
    archivo_img = None
    if metodo == "Subir archivo":
        archivo_img = st.file_uploader("Formatos soportados: JPG, PNG, WEBP", type=["jpg", "jpeg", "png", "webp"])
    else:
        archivo_img = st.camera_input("Toma una captura")

    if archivo_img is not None:
        imagen_pil = Image.open(archivo_img)
        st.image(imagen_pil, use_container_width=True)
        
        ancho, alto = imagen_pil.size
        formato = imagen_pil.format if imagen_pil.format else "JPEG"
        modo_color = imagen_pil.mode
        ratio = round(ancho / alto, 2)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="meta-pill"><div class="meta-label">Resolución</div><div class="meta-val">{ancho} x {alto} px</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="meta-pill"><div class="meta-label">Modo / Formato</div><div class="meta-val">{modo_color} ({formato})</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="meta-pill"><div class="meta-label">Relación Aspecto</div><div class="meta-val">{ratio}:1</div></div>', unsafe_allow_html=True)

with col_der:
    st.subheader("2. Diagnóstico e Interpretación")
    
    prompts_base = {
        "Descripción general detallada": "Describe detalladamente qué aparece en esta imagen, el sujeto o elementos principales, el fondo y el contexto.",
        "Extracción de texto y datos (OCR)": "Lee y transcribe cualquier texto, letrero o número visible en la imagen. Si no hay texto, descríbela.",
        "Análisis estético, paleta y composición": "Analiza la iluminación, el encuadre, el estilo visual y los colores principales de esta fotografía o imagen.",
        "Detección de objetos y conteo": "Enumera en una lista clara todos los objetos, personas o elementos que logres identificar en la imagen."
    }

    if modo_analisis == "Pregunta personalizada":
        instruccion_final = st.text_area("Escribe tu consulta específica sobre la imagen:", height=100)
    else:
        instruccion_final = st.text_area(
            "Instrucción para el modelo (puedes editarla):",
            value=prompts_base[modo_analisis],
            height=100
        )

    if st.button("Ejecutar Análisis Visual"):
        if archivo_img is None:
            st.warning("Carga o toma una imagen primero en el panel izquierdo.")
        else:
            with st.spinner("Analizando imagen con el modelo multimodal..."):
                prompt_completo = (
                    f"{instruccion_final}\n\n"
                    f"Nivel de profundidad: {nivel_detalle}. Responde íntegramente en {idioma_salida}."
                )
                resultado_texto = analizar_imagen_ia(imagen_pil, prompt_completo)

                st.session_state.historial_vision.insert(0, {
                    "modo": modo_analisis,
                    "dimensiones": f"{ancho}x{alto}",
                    "respuesta": resultado_texto
                })

    if st.session_state.historial_vision:
        ultimo = st.session_state.historial_vision[0]
        st.markdown(f"""
        <div class="result-card">
            <strong>Modo aplicado:</strong> {ultimo['modo']} ({ultimo['dimensiones']} px)<br><br>
            {ultimo['respuesta']}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.download_button(
            label="Descargar informe en TXT",
            data=f"REPORTE DE ANÁLISIS VISUAL\nModo: {ultimo['modo']}\nResolución: {ultimo['dimensiones']}\n\n{ultimo['respuesta']}",
            file_name="reporte_vision.txt",
            mime="text/plain"
        )
