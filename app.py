import streamlit as st
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

TRADUCCION_COCO = {
    "person": "una persona",
    "cell phone": "un teléfono celular (smartphone)",
    "laptop": "un computador portátil",
    "mouse": "un ratón (mouse)",
    "keyboard": "un teclado",
    "tv": "un monitor o pantalla",
    "Remote": "un control remoto",
    "remote": "un control remoto",
    "book": "un libro o cuaderno",
    "bottle": "una botella",
    "cup": "una taza o vaso",
    "chair": "una silla",
    "couch": "un sofá",
    "bed": "una cama",
    "backpack": "un morral o mochila",
    "handbag": "un bolso",
    "eyeglasses": "unas gafas",
    "clock": "un reloj",
    "scissors": "unas tijeras",
    "spoon": "una cuchara",
    "fork": "un tenedor",
    "bowl": "un recipiente",
    "apple": "una manzana",
    "banana": "un banano",
    "dog": "un perro",
    "cat": "un gato",
    "car": "un automóvil",
    "potted plant": "una planta en maceta"
}

@st.cache_resource
def cargar_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

if "historial_vision" not in st.session_state:
    st.session_state.historial_vision = []

def analizar_objetos_reales(img_pil, instruccion, nivel, idioma):
    modelo = cargar_detector()
    img_rgb = img_pil.convert("RGB")
    np_img = np.array(img_rgb)[..., ::-1]

    resultados = modelo(np_img, conf=0.22)
    res = resultados[0]
    img_anotada = res.plot()[:, :, ::-1]

    objetos_espanol = []
    detalles_confianza = []
    nombres_modelo = modelo.names

    if res.boxes is not None and len(res.boxes) > 0:
        conteo = {}
        conf_max = {}
        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            nombre_en = nombres_modelo[cls_id]
            nombre_es = TRADUCCION_COCO.get(nombre_en, f"un objeto tipo '{nombre_en}'")
            conteo[nombre_es] = conteo.get(nombre_es, 0) + 1
            conf_max[nombre_es] = max(conf_max.get(nombre_es, 0.0), conf)

        for obj, cant in conteo.items():
            if cant == 1:
                objetos_espanol.append(f"{obj}")
            else:
                objetos_espanol.append(f"{cant} elementos de tipo ({obj})")
            detalles_confianza.append(f"- **{obj.capitalize()}**: detectado en escena (confianza del {conf_max[obj]*100:.1f}%).")

    arr = np.array(img_rgb)
    brillo = float(np.mean(arr))
    r_m, g_m, b_m = int(np.mean(arr[:,:,0])), int(np.mean(arr[:,:,1])), int(np.mean(arr[:,:,2]))
    luz_desc = "iluminación interior equilibrada" if 80 <= brillo <= 175 else ("iluminación intensa/clara" if brillo > 175 else "ambiente de baja luz")

    if objetos_espanol:
        lista_str = ", ".join(objetos_espanol)
        prompt_llm = (
            f"Una red neuronal de visión acaba de analizar una fotografía y detectó exactamente esto en la escena: {lista_str}. "
            f"El ambiente tiene {luz_desc}. "
            f"Atiende esta solicitud del usuario: '{instruccion}'. "
            f"Redacta una descripción natural, fluida y directa en {idioma} (nivel de detalle: {nivel}) mencionando explícitamente los objetos detectados ({lista_str}) y cómo interactúan en la toma."
        )
        try:
            r = requests.post(
                "https://text.pollinations.ai/",
                json={"messages": [{"role": "user", "content": prompt_llm}], "model": "openai"},
                timeout=15
            )
            if r.status_code == 200 and len(r.text.strip()) > 20:
                narrativa = r.text.strip()
            else:
                raise ValueError("Fallback narrativo")
        except Exception:
            narrativa = (
                f"En la imagen capturada se observa claramente **{lista_str}** en primer plano. "
                f"La toma se encuentra en un entorno con {luz_desc} (brillo medio de {brillo:.0f}/255) y tonos predominantes RGB({r_m}, {g_m}, {b_m})."
            )

        resumen_final = (
            f"**Interpretación de la Escena:**\n\n"
            f"{narrativa}\n\n"
            f"**Elementos identificados por la Red Neuronal:**\n" + "\n".join(detalles_confianza)
        )
    else:
        resumen_final = (
            f"**Interpretación de la Escena:**\n\n"
            f"Se observa una toma en primer plano con {luz_desc} (luminancia de {brillo:.0f}/255 y balance cromático RGB {r_m}, {g_m}, {b_m}). "
            f"Acerca un poco más el objeto (como un teléfono, taza, libro o rostro) para resaltar sus bordes en el detector."
        )

    return resumen_final, Image.fromarray(img_anotada)

st.markdown("""
<div class="header-box">
    <p class="header-title">Estudio de Interpretación Visual Multimodal</p>
    <p class="header-sub">Reconocimiento neuronal de objetos en escena, interpretación semántica y telemetría visual sin llaves externas.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Motor Neuronal Activo")
    st.success("Red de Visión YOLOv8 + LLM Conectada")
    mostrar_cajas = st.checkbox("Mostrar recuadros de detección en la foto", value=True)
    
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
        "Descripción general detallada": "Describe detalladamente qué objetos y personas aparecen en la imagen y qué está ocurriendo.",
        "Detección de objetos y conteo": "Indica exactamente qué objetos lograste identificar en la foto y cuántos hay.",
        "Análisis estético, paleta y composición": "Describe los objetos presentes junto con la iluminación y composición de la toma."
    }

    if modo_analisis == "Pregunta personalizada":
        instruccion_final = st.text_area("Escribe tu consulta específica sobre la imagen:", value="¿Qué objetos tiene la persona en la foto?", height=90)
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
            with st.spinner("Escaneando objetos en la imagen con la red neuronal..."):
                texto_res, img_boxes = analizar_objetos_reales(
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
            st.image(ultimo["img_anotada"], caption="Objetos detectados por la red neuronal", use_container_width=True)

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
