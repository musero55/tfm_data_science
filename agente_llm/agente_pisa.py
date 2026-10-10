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
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, trim_messages

#Configuramos las rutas
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ahora sí funcionará la importación
from src.utils_pisa import establecer_semilla

from herramientas import PROMPT_SISTEMA, lista_herramientas_pisa
from src.utils_pisa import establecer_semilla

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


# Compilación del Grafo LangGraph
workflow = StateGraph(State)
workflow.add_node("asistente", asistente)
workflow.add_node("tools", ToolNode(lista_herramientas_pisa))

workflow.add_edge(START, "asistente")
workflow.add_conditional_edges("asistente",tools_condition) #Si tools_condition devuelve tools, vamos a herramientas. Si no END

workflow.add_edge("tools", "asistente")

memoria = MemorySaver()
agente_pisa = workflow.compile(checkpointer=memoria)

# 6. Conexión con el Front-end (Streamlit)
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
    title="Asistente PISA (2012 - 2022)",
    description="Pregunta sobre factores socioeducativos y modelos predictivos PISA."
)

if __name__ == "__main__":
    demo.launch()