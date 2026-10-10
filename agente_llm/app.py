# app.py
import streamlit as st
import uuid
import re
from pathlib import Path
from agente_pisa import agente_pisa  # <-- Tu agente compilado

st.set_page_config(page_title="Agente PISA ", layout="wide")
st.title("📊 Asistente de Investigación PISA")

# 1. Mantener una sesión persistente para la memoria de LangGraph
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 2. Historial visual del chat en Streamlit
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = []

# --- FUNCIÓN PARA SEPARAR EL TEXTO Y PINTAR LA IMAGEN EN STREAMLIT ---
def renderizar_mensaje(contenido: str):
    # Detecta rutas locales Windows (C:/...) tanto si vienen con /gradio_api/file= como si no
    patron_md = r'!\[.*?\]\((?:/gradio_api/file=)?([A-Za-z]:[^\)]+)\)'
    patron_html = r'<img\s+[^>]*src=[\'"](?:/gradio_api/file=)?([A-Za-z]:[^\'"]+)[\'"][^>]*>'
    
    coincidencias = re.findall(patron_md, contenido) + re.findall(patron_html, contenido)
    
    # 1. Deduplicar rutas manteniendo el orden de aparición
    rutas_imagenes = list(dict.fromkeys(r.strip() for r in coincidencias))
    
    # Limpiamos las etiquetas de imagen del texto para evitar fallos de renderizado
    texto_limpio = re.sub(r'!\[.*?\]\([^\)]+\)', '', contenido)
    texto_limpio = re.sub(r'<img\s+[^>]*>', '', texto_limpio)
    
    # 2. Renderizar el texto explicativo
    if texto_limpio.strip():
        st.markdown(texto_limpio)
        
    # 3. Renderizar imágenes con tamaño controlado
    for ruta in rutas_imagenes:
        p = Path(ruta)
        if p.exists():
            # Opción A: ancho fijo controlado (ajusta entre 600 y 750 según prefieras)
            # st.image(str(p), caption="Gráfico generado", width=650)
            
            # Opción B (alternativa): si prefieres centrarlo en la columna del chat
            col_izq, col_centro, col_der = st.columns([1, 4, 1])
            with col_centro:
                st.image(str(p), caption="Gráfico generado", use_container_width=True)

# Dibujar el historial previo en pantalla
for msg in st.session_state.mensajes_chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            renderizar_mensaje(msg["content"])
        else:
            st.markdown(msg["content"])

# 3. Entrada de texto del usuario
if prompt := st.chat_input("Escribe tu consulta sobre el informe PISA..."):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.mensajes_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Ejecución del agente con streaming en pantalla
    with st.chat_message("assistant"):
        config_hilo = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        contenedor_texto = st.empty()
        respuesta_final = ""
        
        # Invocamos el stream del agente
        with st.spinner("Consultando datos y modelos PISA..."):
            for evento in agente_pisa.stream(
                {"messages": [("user", prompt)]}, 
                config=config_hilo, 
                stream_mode="updates"
            ):
                if "asistente" in evento:
                    msg = evento["asistente"]["messages"][-1]
                    if hasattr(msg, "content") and msg.content:
                        respuesta_final = msg.content
                        
                        # Mientras va escribiendo, filtramos el tag de la imagen para que no falle
                        texto_parcial = re.sub(r'!\[.*?\]\([^\)]+\)', '', respuesta_final)
                        texto_parcial = re.sub(r'<img\s+[^>]*>', '', texto_parcial)
                        contenedor_texto.markdown(texto_parcial)

        # Cuando el stream termina, mostramos el mensaje completo con su imagen
        if respuesta_final:
            contenedor_texto.empty()  # Limpia el borrador previo
            renderizar_mensaje(respuesta_final)
            st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta_final})