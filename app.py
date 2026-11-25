"""Dashboard de mantenimiento predictivo + Orden de Trabajo con Groq."""
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
ESTADOS = {0: "Operación Normal", 1: "Mantenimiento Preventivo", 2: "Fallo Crítico"}


@st.cache_resource
def cargar_modelo():
    # Solo depende del .pkl, no del CSV.
    return joblib.load("modelo_mantenimiento.pkl")


modelo = cargar_modelo()

# ---------- Sidebar ----------
st.sidebar.header("🎛️ Lecturas de sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 120.0, 60.0, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, 0.1)
horas = st.sidebar.slider("Horas de operación", 0, 6000, 300, 10)
presion = st.sidebar.slider("Presión (PSI)", 70.0, 170.0, 92.0, 0.5)

# ---------- Cabecera y métricas ----------
st.title("🏭 Sistema Híbrido de Mantenimiento Predictivo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# ---------- Predicción ----------
entrada = pd.DataFrame([[temperatura, vibracion, horas, presion]], columns=FEATURES)
estado = int(modelo.predict(entrada)[0])
probas = dict(zip(modelo.classes_, modelo.predict_proba(entrada)[0]))
prob_estado = probas[estado]

st.subheader("🔎 Diagnóstico del modelo")
mensaje = f"**{ESTADOS[estado]}** — probabilidad: {prob_estado * 100:.1f}%"
if estado == 0:
    st.success(f"✅ {mensaje}")
elif estado == 1:
    st.warning(f"⚠️ {mensaje}")
else:
    st.error(f"🚨 {mensaje}")

with st.expander("Probabilidad por clase"):
    for clase, p in probas.items():
        st.write(f"{ESTADOS[int(clase)]}")
        st.progress(float(p), text=f"{p * 100:.1f}%")

# ---------- Groq ----------
SYSTEM_PROMPT = (
    "Eres el Jefe de Mantenimiento Industrial de una planta de manufactura automatizada. "
    "Respondes en español, con tono técnico, claro y accionable. No inventes datos "
    "fuera de las lecturas recibidas."
)


def construir_prompt(estado: int, folio: str) -> str:
    lecturas = (
        f"Temperatura: {temperatura} °C | Vibración: {vibracion} mm/s | "
        f"Horas de operación: {horas} | Presión: {presion} PSI"
    )
    if estado == 0:
        return (
            f"Lecturas: {lecturas}.\nEl modelo clasificó la máquina como OPERACIÓN NORMAL. "
            "Emite un dictamen BREVE (máximo 4 líneas) de monitoreo preventivo de rutina."
        )
    nivel = "PREVENTIVA (programar revisión)" if estado == 1 else "URGENTE (paro inmediato)"
    return (
        f"Lecturas: {lecturas}.\nEl modelo clasificó la máquina como "
        f"{ESTADOS[estado].upper()}. Genera una ORDEN DE TRABAJO {nivel} en Markdown "
        f"con exactamente estas secciones:\n"
        f"1. **Folio**: usa {folio}\n2. **Diagnóstico**\n"
        "3. **Acciones Técnicas** (exactamente 3, numeradas)\n"
        "4. **Equipo de Seguridad (EPP)** (lista)\n"
        + ("Incluye el protocolo de paro inmediato y bloqueo/etiquetado (LOTO)." if estado == 2 else "")
    )


EXCLUIR = ["whisper", "guard", "tts", "playai", "embed", "orpheus", "safeguard"]
PREFERIDOS = ["llama", "gpt-oss", "qwen", "allam"]


def modelos_chat(client: Groq) -> list[str]:
    """Detecta dinámicamente modelos de chat activos (evita errores 404), por preferencia."""
    activos = [
        m.id
        for m in client.models.list().data
        if not any(x in m.id.lower() for x in EXCLUIR)
    ]
    def prioridad(mid: str) -> int:
        return next((i for i, p in enumerate(PREFERIDOS) if p in mid.lower()), len(PREFERIDOS))
    return sorted(activos, key=lambda m: (prioridad(m), m))


def generar_orden(estado: int) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    folio = f"OT-{datetime.now():%Y%m%d-%H%M%S}"
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": construir_prompt(estado, folio)},
    ]
    ultimo_error = None
    for modelo_groq in modelos_chat(client):
        try:
            resp = client.chat.completions.create(model=modelo_groq, messages=mensajes, temperature=0.3)
            return f"{resp.choices[0].message.content}\n\n---\n*Generado con `{modelo_groq}`*"
        except Exception as e:  # modelo no apto: probar el siguiente
            ultimo_error = e
    raise ultimo_error or RuntimeError("Sin modelos de chat disponibles")


st.subheader("📋 Orden de Trabajo (Groq)")
etiqueta = "Generar dictamen" if estado == 0 else "Generar Orden de Trabajo"
if st.button(etiqueta, type="primary"):
    try:
        with st.spinner("Consultando a Groq..."):
            st.markdown(generar_orden(estado))
    except Exception as e:
        # Resiliencia: falta de clave, clave inválida, límite de peticiones, red...
        st.info(
            "⚙️ No se pudo generar la Orden de Trabajo con IA "
            f"({type(e).__name__}). El diagnóstico de ML sigue vigente: "
            f"**{ESTADOS[estado]}** ({prob_estado * 100:.1f}%)."
        )
