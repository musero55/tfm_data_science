import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, Annotated

import pandas as pd
import gradio as gr
from dotenv import find_dotenv, load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, trim_messages

#Configuramos las rutas
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from herramientas import PROMPT_SISTEMA, lista_herramientas_pisa
from src.utils_pisa import establecer_semilla

establecer_semilla(42)
pd.set_option("display.float_format", lambda valor: f"{valor:.2f}")

DIR_IMAGENES = os.path.join(tempfile.gettempdir(), "pisa_plots")
os.makedirs(DIR_IMAGENES, exist_ok=True)

# Credenciales y LLM
load_dotenv(find_dotenv(usecwd=True))
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Añade GROQ_API_KEY a tu archivo .env.")

llm = ChatGroq(temperature=0.0, model_name='openai/gpt-oss-120b')
llm_con_herramientas = llm.bind_tools(lista_herramientas_pisa)

# Estado y Memoria
class State(TypedDict):
    messages: Annotated[list, add_messages]

recortador = trim_messages(
    max_tokens=20,
    strategy="last",
    token_counter=len,
    include_system=True,
    start_on="human"
)

# 4. Nodos del grafo
def asistente(state: State):
    mensajes_completos = [SystemMessage(content=PROMPT_SISTEMA)] + state["messages"]
    mensajes_recortados = recortador.invoke(mensajes_completos)
    
    # Invocamos al LLM
    respuesta = llm_con_herramientas.invoke(mensajes_recortados)
    
    # Saneado defensivo contra canales de razonamiento corruptos
    if hasattr(respuesta, "tool_calls") and respuesta.tool_calls:
        for tool_call in respuesta.tool_calls:
            if "<|" in tool_call["name"]:
                tool_call["name"] = tool_call["name"].split("<|")[0].strip()

    return {"messages": [respuesta]}

def enrutador_asistente(state: State):
    ultimo_mensaje = state["messages"][-1]
    if not hasattr(ultimo_mensaje, "tool_calls") or not ultimo_mensaje.tool_calls:
        return END

    # Si ya generó un gráfico, cortamos el bucle hacia END
    for msg in reversed(state["messages"][:-1]):
        if getattr(msg, "type", "") == "tool":
            if "generar_grafico" in getattr(msg, "name", ""):
                return END
            break

    return "tools"

# 5. Compilación del Grafo LangGraph
workflow = StateGraph(State)
workflow.add_node("asistente", asistente)
workflow.add_node("tools", ToolNode(lista_herramientas_pisa))

workflow.add_edge(START, "asistente")
workflow.add_conditional_edges("asistente", enrutador_asistente)
workflow.add_edge("tools", "asistente")

memoria = MemorySaver()
agente_pisa = workflow.compile(checkpointer=memoria)

# 6. Conexión con el Front-end (Gradio)
def responder_chat(mensaje_usuario, historial):
    config_hilo = {"configurable": {"thread_id": "sesion_activa"}}
    
    # Ejecutamos el agente
    respuesta_final = ""
    for evento in agente_pisa.stream(
        {"messages": [("user", mensaje_usuario)]}, 
        config=config_hilo, 
        stream_mode="updates"
    ):
        if "asistente" in evento:
            respuesta_final = evento["asistente"]["messages"][-1].content

    return respuesta_final

demo = gr.ChatInterface(
    fn=responder_chat,
    title="Asistente PISA - TFM",
    description="Pregunta sobre factores socioeducativos y modelos predictivos PISA."
)

if __name__ == "__main__":
    demo.launch()