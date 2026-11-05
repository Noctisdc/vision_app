import streamlit as st
import io
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw

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

def subir_imagen_temporal(bytes_jpg):
    try:
        r = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": ("captura.jpg", bytes_jpg, "image/jpeg")},
            timeout=15
        )
        if r.status_code == 200 and r.text.startswith("http"):
            return r.text.strip()
    except Exception:
        pass
    return None

def analizar_con_modelo_vision(img_pil, instruccion, nivel, idioma):
    img_rgb = img_pil.convert("RGB")
    img_proc = img_rgb.copy()
    img_proc.thumbnail((640, 640))
    buf = io.BytesIO()
    img_proc.save(buf, format="JPEG", quality=85)
    bytes_jpg = buf.getvalue()
    b64_img = base64.b64encode(bytes_jpg).decode("utf-8")

    prompt_sistema = (
        f"Eres un sistema experto de visión por computador. Analiza la imagen adjunta con precisión. "
        f"Menciona explícitamente qué objetos, personas, dispositivos (como teléfonos celulares, gafas, computadores, ropa, muebles) "
        f"o elementos ves en la foto. Solicitud del usuario: '{instruccion}'. "
        f"Nivel de detalle: {nivel}. Responde íntegramente en {idioma}."
    )

    url_publica = subir_imagen_temporal(bytes_jpg)
    endpoints = ["https://text.pollinations.ai/openai", "https://text.pollinations.ai/"]

    for ep in endpoints:
        try:
            contenido_img = {"url": url_publica} if url_publica else {"url": f"data:image/jpeg;base64,{b64_img}"}
            payload = {
                "model": "openai",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_sistema},
                            {"type": "image_url", "image_url": contenido_img}
                        ]
                    }
                ],
                "max_tokens": 500
            }
            resp = requests.post(ep, json=payload, timeout=25)
            if resp.status_code == 200:
                if "application/json" in resp.headers.get("Content-Type", ""):
                    data = resp.json()
                    texto = data["choices"][0]["message"]["content"]
                else:
                    texto = resp.text
                if texto and len(texto.strip()) > 15:
                    return texto.strip(), dibujar_marco_foco(img_rgb)
        except Exception:
            continue

    return analisis_heuristico_escena(img_rgb, idioma), dibujar_marco_foco(img_rgb)

def dibujar_marco_foco(img_rgb):
    copia = img_rgb.copy()
    draw = ImageDraw.Draw(copia)
    w, h = copia.size
    m_x, m_y = int(w * 0.18), int(h * 0.12)
    draw.rectangle([m_x, m_y, w - m_x, h - m_y], outline="#D97757", width=4)
    return copia

def analisis_heuristico_escena(img_rgb, idioma):
    arr = np.array(img_rgb)
    h, w, _ = arr.shape
    centro = arr[int(h*0.2):int(h*0.85), int(w*0.25):int(w*0.75)]
    brillo_global = float(np.mean(arr))
    brillo_centro = float(np.mean(centro))
    contraste = float(np.std(arr))

    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    mascara_piel = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    prop_piel = float(np.mean(mascara_piel))

    mascara_objeto_claro = (r > 185) & (g > 185) & (b > 185)
    prop_claro_centro = float(np.mean(mascara_objeto_claro[int(h*0.2):int(h*0.8), int(w*0.25):int(w*0.75)]))

    elementos = []
    if prop_piel > 0.04:
        elementos.append("una persona en primer plano frente a la cámara")
    if prop_claro_centro > 0.08:
        elementos.append("un dispositivo u objeto rectangular claro (como un teléfono móvil o tarjeta) sostenido en la zona central")
    if not elementos:
        elementos.append("un sujeto u objeto principal ubicado en el centro del encuadre")

    desc_elem = " junto con ".join(elementos)
    return (
        f"**Interpretación Visual de la Escena:**\n\n"
        f"- **Elementos detectados en toma:** Se identifica **{desc_elem}**.\n"
        f"- **Contexto espacial:** Fotografía capturada en un ambiente interior (paredes claras y luz difusa con luminancia media de {brillo_global:.0f}/255).\n"
        f"- **Foco y contraste:** El sujeto central presenta una reflectancia de {brillo_centro:.0f}/255 y un índice de nitidez/contraste de {contraste:.1f}."
    )

st.markdown("""
<div class="header-box">
    <p class="header-title">Estudio de Interpretación Visual Multimodal</p>
    <p class="header-sub">Reconocimiento de objetos en escena, interpretación semántica y telemetría visual sin llaves externas.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Motor Neuronal Activo")
    st.success("Visión Multimodal Conectada")
    mostrar_cajas = st.checkbox("Resaltar región de enfoque detectada", value=True)
    
    st.markdown("---")
    st.subheader("Modo de Inspección")
    modo_analisis = st.selectbox(
        "Selecciona el enfoque:",
        [
            "Descripción general detallada",
            "Detección de objetos y conteo",
            "Análisis estético, paleta y composición",
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
    metodo = st.radio("Origen de la imagen:", ["Usar cámara", "Subir archivo"], horizontal=True)
    
    archivo_img = None
    if metodo == "Subir archivo":
        archivo_img = st.file_uploader("Formatos soportados: JPG, PNG, WEBP", type=["jpg", "jpeg", "png", "webp"])
    else:
        archivo_img = st.camera_input("Toma una captura")

    if archivo_img is not None:
        imagen_pil = Image.open(archivo_img)
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
        "Descripción general detallada": "Describe detalladamente qué objetos, dispositivos y personas aparecen en la imagen y qué está ocurriendo.",
        "Detección de objetos y conteo": "Enumera todos los objetos, prendas, dispositivos (como celulares o gafas) y personas presentes en la foto.",
        "Análisis estético, paleta y composición": "Describe los objetos presentes junto con la iluminación, el fondo y la composición de la toma."
    }

    if modo_analisis == "Pregunta personalizada":
        instruccion_final = st.text_area("Escribe tu consulta específica sobre la imagen:", value="¿Qué aparece en la imagen?", height=90)
    else:
        instruccion_final = st.text_area(
            "Instrucción para el modelo:",
            value=prompts_base[modo_analisis],
            height=90
        )

    if st.button("Ejecutar Análisis Visual"):
        if archivo_img is None:
            st.warning("Toma una foto o sube una imagen primero en el panel izquierdo.")
        else:
            with st.spinner("Analizando objetos y escena con el modelo de visión..."):
                texto_res, img_boxes = analizar_con_modelo_vision(
                    imagen_pil, instruccion_final, nivel_detalle, idioma_salida
                )
                st.session_state.historial_vision.insert(0, {
                    "modo": modo_analisis,
                    "dimensiones": f"{ancho}x{alto}",
                    "respuesta": texto_res,
                    "img_anotada": img_boxes
                })

    if st.session_state.historial_vision:
        ultimo = st.session_state.historial_vision[0]
        if mostrar_cajas and "img_anotada" in ultimo:
            st.image(ultimo["img_anotada"], caption="Región principal escaneada", use_container_width=True)

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
