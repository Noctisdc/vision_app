import streamlit as st
import base64
import io
from openai import OpenAI
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

st.markdown("""
<div class="header-box">
    <p class="header-title">Estudio de Interpretación Visual Multimodal</p>
    <p class="header-sub">Inspección de imágenes, extracción de texto (OCR) y análisis de composición mediante modelos de visión.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Configuración del Modelo")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
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
        formato = imagen_pil.format if imagen_pil.format else "Imagen"
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
        "Descripción general detallada": "Describe qué aparece en la imagen, el contexto, los elementos principales y las acciones o detalles relevantes.",
        "Extracción de texto y datos (OCR)": "Transcribe todo el texto visible en la imagen respetando su estructura y resume cualquier dato clave o tabla que aparezca.",
        "Análisis estético, paleta y composición": "Evalúa la composición visual, iluminación, estilo gráfico y menciona los colores predominantes aproximados.",
        "Detección de objetos y conteo": "Haz un inventario en viñetas de todos los objetos o elementos identificables en la imagen e indica cuántos hay de cada uno."
    }

    if modo_analisis == "Pregunta personalizada":
        instruccion_final = st.text_area("Escribe tu consulta específica sobre la imagen:", height=100)
    else:
        instruccion_final = st.text_area(
            "Instrucción que se enviará al modelo (puedes editarla):",
            value=prompts_base[modo_analisis],
            height=100
        )

    ejecutar = st.button("Ejecutar Análisis Visual")

    if ejecutar:
        if not api_key:
            st.warning("Ingresa tu OpenAI API Key en el menú lateral para continuar.")
        elif archivo_img is None:
            st.warning("Carga o toma una imagen primero en el panel izquierdo.")
        else:
            with st.spinner("Analizando imagen..."):
                img_rgb = imagen_pil.convert("RGB")
                buffer = io.BytesIO()
                img_rgb.save(buffer, format="JPEG")
                b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

                prompt_completo = (
                    f"{instruccion_final}\n\n"
                    f"Nivel de profundidad requerido: {nivel_detalle}. "
                    f"Responde íntegramente en {idioma_salida}."
                )

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_completo},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    max_tokens=700
                )
                resultado_texto = response.choices[0].message.content
                
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

        if len(st.session_state.historial_vision) > 1:
            with st.expander("Ver consultas anteriores de esta sesión"):
                for idx, item in enumerate(st.session_state.historial_vision[1:], start=1):
                    st.markdown(f"**Consulta #{idx} ({item['modo']}):**")
                    st.write(item["respuesta"])
                    st.markdown("---")
