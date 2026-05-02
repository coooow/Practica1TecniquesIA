#CREWAI llibreria per instanciar i definir els agents
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool
import os, json
#LLAMA 3 com a llenguatge LLM instal·lat en local
llm_local = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434" # Port per defecte de Llama3
)

path_cv = "data/users/software_engineer_cv.tex" #Ruta al CV del usuario
path_example = "outputs/cv_estructurado_ejemplo.md" #Ruta al ejemplo de CV estructurado para el agente extractor de datos.

#Funciones & Herramientas
#-------------------------------------------------
def cargar_usuarios(UserId):
    
    path= os.path.join("data", "users", f"{UserId}.json") #Construimos la ruta al archivo JSON del usuario.
    if not os.path.exists(path):    #comparamos si el archivo existe, si no existe lanzamos un error.
        raise FileNotFoundError("Usuario no encontrado")
    with open(path, "r", encoding="utf-8") as f:
        data= json.load(f)
    requisitos = data["requisitos"]
    cv = data["cv"] #Obtenemos el CV del usuario i requisitos, que se encuentran en el archivo JSON.

    # Convertir a texto para el LLM
    requisitos_txt = "\n".join([f"{k}: {v}" for k, v in requisitos.items()]) # Convertimos los requisitos a un formato de texto legible para el LLM.
    cv_txt = "\n".join([
        f"{k}: {', '.join(v) if isinstance(v, list) else v}" 
        for k, v in cv.items()
    ])

    return requisitos_txt, cv_txt #Devolvemos los requisitos y el CV en formato de texto para que puedan ser utilizados por los agentes.

# Read the file using standard Python
with open(path_cv, "r", encoding="utf-8") as file:
    latex_text = file.read()

with open(path_example, "r", encoding="utf-8") as file:
    ejemplo_cv = file.read()

#Inicialitzem els agents amb el model local.
#------------------------------------------------------------------------

#Agente para buscar ofertas de trabajo abiertas que cumplan estrictamente los requisitos definidos por el usuario.
user_id = "user1"

requisitos_usuario, cv_usuario = cargar_usuarios(user_id)

IA_extractor_dades = Agent(
    role="Extractor de Datos",
    goal=(
        "Extraer y estructurar la información clave del CV del usuario, identificando habilidades, experiencia, formación y logros relevantes para el mercado laboral actual."
    ),
    backstory=(
        "Eres un experto en análisis de CVs y perfiles profesionales. Tienes una gran capacidad para identificar y extraer la información más relevante de un CV, incluyendo habilidades técnicas, experiencia laboral, formación académica y logros destacados. Eres capaz de interpretar correctamente la información, incluso cuando está presentada de forma no estructurada, y de organizarla de manera clara y accesible para su posterior análisis."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_local
)

IA_cercador_feina = Agent(
    role="Buscador de Empleo",
    goal=(
        "Buscar y recopilar ofertas de trabajo abiertas que cumplan de forma estricta los requisitos definidos por el usuario, incluyendo horario, disponibilidad, localización, tipo de jornada, modalidad de trabajo y cualquier otro criterio relevante."
    ),
    backstory=(
        "Eres un especialista en localizar oportunidades laborales de alta calidad en múltiples "
        "bolsas de trabajo y plataformas profesionales. Trabajas con precisión y criterio, evitando "
        "resultados irrelevantes. Analizas cada oferta en profundidad, interpretas correctamente "
        "los requisitos y condiciones, y seleccionas únicamente aquellas que encajan realmente con "
        "los parámetros del usuario. Para cada oferta válida, registras la información clave de forma "
        "estructurada: puesto, empresa, ubicación, condiciones, requisitos y justificación del encaje."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_local
)
#Agente per analitzar les ofertes recopilades i filtrar les que millor s'ajusten al perfil del usuari..
IA_analista_oportunidades = Agent(
    role="Analista de Oportunidades",
    goal=(
        "Analizar las ofertas recopiladas y filtrar aquellas que mejor se ajustan al perfil "
        "del usuario, priorizando las que tienen mayor probabilidad de éxito según su CV."
    ),
    backstory=(
        "Eres un analista experto en evaluar la compatibilidad entre perfiles profesionales y "
        "ofertas de empleo. Tienes una gran capacidad para detectar el encaje real entre los "
        "requisitos de una oferta y las habilidades, experiencia y formación del candidato. "
        "No te limitas a clasificar: justificas cada decisión, identificas fortalezas, carencias "
        "y nivel de ajuste, y priorizas las oportunidades más viables y estratégicas."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_local
)
#Agente para editar el CV y redactar cartas de presentación personalizadas para cada oferta seleccionada.
IA_editor_cv = Agent(
    role="Editor de CV",
    goal=(
        "Adaptar y optimizar el CV del usuario y redactar una carta de presentación específica "
        "para cada oferta seleccionada, maximizando la claridad, relevancia y alineación con la vacante."
    ),
    backstory=(
        "Eres un experto en recursos humanos y procesos de selección. Conoces cómo funcionan los "
        "sistemas de filtrado (ATS) y cómo evalúan los reclutadores las candidaturas. Eres capaz de "
        "reescribir CVs para destacar los aspectos más relevantes según cada oferta, optimizando "
        "palabras clave, estructura y contenido. Además, redactas cartas de presentación personalizadas, "
        "claras y persuasivas, alineadas con los requisitos del puesto y enfocadas a maximizar las "
        "probabilidades de ser seleccionado."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_local
)

#Inicialitzem les tasques que cada agent ha de realitzar.
#------------------------------------------------------------------------   

tarea_extraccion = Task(
    description=(
        "Analiza el siguiente texto extraído de un CV en formato LaTeX:\n\n"
        "{cv_content}\n\n" # <--- Pass the text directly here
        "Extraer y estructurar la información clave del CV, identificando habilidades, experiencia, formación y logros relevantes para el mercado laboral actual. "
    ),
    expected_output="CV del usuario estructurado en formato Markdown con su información de contacto, perfil profesional, experiencia y formación. Asegurarse de que esté correctamente formateado usando la estructura basada en este ejemplo {ejemplo_cv}.",
    agent=IA_extractor_dades,
    output_file="outputs/cv_estructurado.md"
)

tarea_busqueda=Task(
    description=(f"Buscar ofertas que cumplan estos requisitos:\n{requisitos_usuario}"),
    expected_output=("Una lista de ofertas de trabajo que cumplan estrictamente los requisitos definidos por el usuario, incluyendo detalles clave como puesto, empresa, ubicación, condiciones, requisitos y justificación del encaje."),
    agent=IA_cercador_feina
)

tarea_analisis = Task(
    description=(
        "Analizar las ofertas obtenidas y compararlas con el CV del usuario."
        "Filtrar y priorizar aquellas que mejor encajan con su perfil profesional."
    ),
    expected_output=(
        "Lista priorizada de ofertas filtradas. Para cada una indicar: "
        "nivel de encaje (alto, medio, bajo), fortalezas del candidato respecto a la oferta, "
        "posibles carencias y justificación del ranking."
    ),
    agent=IA_analista_oportunidades
)

tarea_cv=Task(
    description=(
        "Para las mejores ofertas seleccionadas, adaptar el CV del usuario y redactar "
        "una carta de presentación específica para cada oferta."
    ),
    expected_output=(
        "Para cada oferta seleccionada: "
        "1) Versión optimizada del CV adaptada a la oferta, "
        "2) Carta de presentación personalizada alineada con los requisitos del puesto."
    ),
    agent=IA_editor_cv,
    output_file="outputs/candidaturas.md"
)
#Inicialitzem la CREW
#-------------------------------------------------------------------
equipo= Crew(
    agents=[
        IA_extractor_dades,
        IA_cercador_feina,
        IA_analista_oportunidades,
        IA_editor_cv
    ],
    tasks=[
        tarea_extraccion,
        tarea_busqueda,
        tarea_analisis,
        tarea_cv
    ],
    verbose=True
)

#EXECUCIO
#-------------------------------------
# Ejecución
resultado = equipo.kickoff(inputs={
    "cv_content": latex_text,
    "ejemplo_cv": ejemplo_cv
    })









