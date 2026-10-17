
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import Field,BaseModel

import re
import time
from typing import List, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pathlib import Path


from reference_test import CASOS_TEST_PISA


# Generamos un esquema estrucurado con pydantic de las respuestas que el juez debe mostrar a la salida de tal forma que se evita generar texto plano desestructurado

#Funciones publicas
__all__ = ["ejecutar_bateria_evaluacion"]

DIR_ACTUAL = Path(__file__).resolve().parent
RUTA_PROMPT_JUEZ = DIR_ACTUAL / "judge_prompt.md"

#Definimos la estructura del output que el juez debe seguir
class VeredictoJuez(BaseModel):
    sin_alucinaciones: bool = Field(
        description='True si los datos numéricos y conclusiones coinciden con la referencia real. False si inventa datos o incumple guardrails.'
    )
    puntuacion_calidad: int = Field(
        ge=1, le=5, # greather than 1 less than 5
        description='Puntuación del 1 al 5. Nivel de alineación y rigor con los datos y conclusiones reales (1=totalmente contradictorio, 5=perfectamente alineado).'
    )
    completitud: int = Field(
        description='Puntuación del 1 al 5. ¿El agente respondió a todo lo solicitado en la pregunta de referencia?',
        ge=1, le=5,
    )
    motivo: str = Field(
        description='Justificación breve en una frase de por qué se considera acierto o fallo.'
    )

# ## Funciónes de evaluacion


PALABRAS_PROHIBIDAS = [
    "xgboost", "escs_estrato", "brecha_escs", "media_math", 
    "media_read", "media_scie", "pib_per_capita", "gasto_alumno"
]


#Funcion para evaluar la inclusion de variables técnicas o variables de los dataframes
def _verificar_limpieza_tecnica(texto: str) -> bool:
    """Devuelve True si el texto está limpio (NO contiene variables técnicas ni 'XGBoost')."""
    texto_lower = texto.lower()
    for palabra in PALABRAS_PROHIBIDAS:
        if palabra in texto_lower:
            return False
    # Detectar patrones típicos de nombres de columnas con guiones bajos (ej. variable_1)
    if re.search(r"\b[a-z]+_[a-z_]+\b", texto_lower):
        return False
    return True


#Cargamos el prompt del juez 
with open(RUTA_PROMPT_JUEZ, "r", encoding="utf-8") as file:
    PROMPT_JUEZ = file.read()

# Función de evaluación que conecta el agente, las herramientas y el juez

#Función para evaluar un único caso pasando el agente como argumento. Permite seleccionar el modelo del juez a usar 
def _evaluar_caso_estudio(agente, caso: Dict[str, Any], modelo_juez: str = "openai/gpt-oss-120b") -> Dict[str, Any]:

    #Cogemos las variables del caso
    llm_juez = ChatGroq(temperature=0.0, model_name=modelo_juez)
    llm_juez_estructurado = llm_juez.with_structured_output(VeredictoJuez)

    test_id = caso["id"]
    pregunta = caso["pregunta"]
    herramientas_esperadas = caso.get("herramientas_imprescindibles", [])
    min_llamadas = caso.get("min_llamadas_herramienta", len(herramientas_esperadas))
    ref = caso["referencia"]
    
    print(f" Ejecutando caso :{test_id} -> {pregunta}")
    inicio = time.time()
    
    # Ejecutar el agente PISA en un hilo aislado (util para evaluar diferentes casos)
    config_run = {"configurable": {"thread_id": f"eval_{test_id}_{int(inicio)}"}, "recursion_limit": 10} #Configuración del hilo por caso
    salida = agente.invoke({"messages": [("user", pregunta)]}, config=config_run)
    tiempo_seg = round(time.time() - inicio, 2) #Calculamos el delay manualmente

    mensajes = salida["messages"] #Obtenemos todos los mensajes del agente
    respuesta_agente = mensajes[-1].content #Nos quedamos con la respuesta 
    
    herramientas_usadas = []
    for msg in mensajes:
        if hasattr(msg, "tool_calls") and msg.tool_calls: #Filtramos los obetos pydantic que tienen herramientas como atributo y con el campo de herramientas relleno
            for tc in msg.tool_calls:
                herramientas_usadas.append(tc["name"]) #Fuardamos las herramientas usadas
      
    # Comprobamos las herramientas 
    if not herramientas_esperadas:
        tools_ok = len(herramientas_usadas) == 0
    else:
        todas_presentes = all(h in herramientas_usadas for h in herramientas_esperadas)
        cantidad_suficiente = len(herramientas_usadas) >= min_llamadas
        tools_ok = todas_presentes and cantidad_suficiente

    # Evaluar limpieza de variables técnicas
    estilo_limpio = _verificar_limpieza_tecnica(respuesta_agente)
        
    # Preparamos el prompt de entrada para que lo evalue el juez
    datos_clave_str = "\n".join([f"- {d}" for d in ref.get("datos_clave", [])])
    conclusiones_str = "\n".join([f"- {c}" for c in ref.get("conclusiones_esperadas", [])])

    
    # Rellenamos el prompt del juez
    prompt_completo = PROMPT_JUEZ.format(
        pregunta=pregunta,
        datos_clave=datos_clave_str,
        conclusiones_esperadas=conclusiones_str,
        respuesta_agente=respuesta_agente
    )
    
    veredicto: VeredictoJuez = llm_juez_estructurado.invoke([HumanMessage(content=prompt_completo)])
    
    # Devolvemos los resultados en un diccionario
    return {
        "ID": test_id,
        "Categoría": caso.get("categoria", "General"),
        "Tools OK": "✅" if tools_ok else "❌",
        "Fidelidad OK": "✅" if veredicto.sin_alucinaciones else "❌",
        "Estilo Limpio": "✅" if estilo_limpio else "⚠️ (Tecnicismos)",
        "Calidad (1-5)": veredicto.puntuacion_calidad,
        "Completitud (1-5)": veredicto.completitud,
        "Tiempo (s)": tiempo_seg,
        "Tools Esperadas": ", ".join(herramientas_esperadas) if herramientas_esperadas else "Ninguna",
        "Tools Usadas": ", ".join(herramientas_usadas) if herramientas_usadas else "Ninguna",
        "Dictamen Juez": veredicto.motivo,
        "Respuesta Agente": respuesta_agente
    }

#Función para evaluar toda la batería de casos de prueba
def ejecutar_bateria_evaluacion(agente,  casos: List[Dict[str, Any]] = CASOS_TEST_PISA, modelo_juez: str = "qwen/qwen3.8-27b") -> pd.DataFrame:
    resultados = []
    for caso in casos:
        try:
            res = _evaluar_caso_estudio(agente, caso, modelo_juez=modelo_juez)
            resultados.append(res)
        except Exception as e:
            print(f"❌ Error ejecutando {caso['id']}: {e}")
            resultados.append({
                "ID": caso["id"],
                "Categoría": caso.get("categoria", "General"),
                "Tools OK": "💥 Error",
                "Fidelidad OK": "💥 Error",
                "Estilo Limpio": "-",
                "Calidad (1-5)": 0,
                "Completitud (1-5)": 0,
                "Tiempo (s)": 0,
                "Tools Esperadas": ", ".join(caso.get("herramientas_imprescindibles", [])),
                "Tools Usadas": "Error en ejecución",
                "Dictamen Juez": str(e),
                "Respuesta Agente": ""
            })
            
    df_resultados = pd.DataFrame(resultados)
    print("\n✅ Evaluación finalizada.")
    return df_resultados

