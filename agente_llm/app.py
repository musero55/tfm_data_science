# app.py
import streamlit as st
import uuid
from agente_pisa import agente_pisa  # <-- ¡Importas directamente tu agente ya compilado!

st.set_page_config(page_title="Agente PISA - TFM", layout="wide")
st.title("📊 Asistente de Investigación PISA")

# 1. Mantener una sesión persistente para la memoria de LangGraph
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 2. Historial visual del chat en Streamlit
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = []

# Dibujar el historial en pantalla
for msg in st.session_state.mensajes_chat:
    with st.chat_message(msg["role"]):
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
        
        # Invocamos el stream del agente ya configurado
        with st.spinner("Consultando datos y modelos PISA..."):
            for evento in agente_pisa.stream(
                {"messages": [("user", prompt)]}, 
                config=config_hilo, 
                stream_mode="updates"
            ):
                if "asistente" in evento:
                    respuesta_final = evento["asistente"]["messages"][-1].content
                    contenedor_texto.markdown(respuesta_final)

        # Guardar la respuesta final en el historial de la sesión
        st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta_final})