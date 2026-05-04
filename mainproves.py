#CREWAI llibreria per instanciar i definir els agents
from crewai import Agent, Task, Crew, LLM
import os, json
#LLAMA 3 com a llenguatge LLM instal·lat en local
llm_local = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434" # Port per defecte de Llama3
)

path_example = "outputs/cv_estructurado_ejemplo.md" #Ruta al ejemplo de CV estructurado para el agente extractor de datos.

#Input del usuario
#-------------------------------------------------
print("Indica el path del CV en formato LaTeX que quieres analizar: (ejemplo: data/users/software_engineer_cv.tex)")

cv_path_raw = input()
cv_path = "data/users/"+cv_path_raw if not cv_path_raw.startswith("data/users/") else cv_path_raw
with open(cv_path, 'r') as f:
    latex_text = f.read()

print("Indica la ubicación de tu busqueda de trabajo: (ejemplo: Barcelona, Madrid, etc.)")
location = input()

print("Indica tu disponibilidad horaria: (ejemplo: jornada completa, media jornada, etc.)")
availability = input()

print("Indica el tipo de jornada que prefieres: (ejemplo: presencial, remoto, híbrido)")
work_type = input()

print("Indica tu disponibilidad para empezar a trabajar: (ejemplo: inmediata, 1 mes, etc.)")
start_date = input()

print("Indica el tipo de puesto que buscas: (ejemplo: desarrollador backend, analista de datos, etc.)")
job_type = input()

requisitos_usuario = f"Ubicación: {location}\nDisponibilidad: {availability}\nTipo de jornada: {work_type}\nDisponibilidad para empezar: {start_date}\nTipo de puesto: {job_type}"

#Funciones & Herramientas
#-------------------------------------------------
# Leer el ejemplo de CV estructurado en formato Markdown
with open(path_example, "r", encoding="utf-8") as file:
    ejemplo_cv = file.read()

#Inicialitzem els agents amb el model local.
#------------------------------------------------------------------------

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

IA_redactor_carta = Agent(
    role="Redactor de carta de presentación",
    goal=(
        "Redactar una única carta de presentación personalizada exclusivamente para la mejor oferta seleccionada, usando el CV del usuario y el análisis de compatibilidad realizado."
    ),
    backstory=(
        "Eres un especialista en redacción profesional orientada a procesos de selección. "
        "Tu trabajo consiste en crear una carta de presentación clara, específica y adaptada "
        "únicamente a la oferta con mayor encaje. No generas cartas genéricas ni múltiples versiones. "
        "Analizas la mejor oportunidad detectada, identificas los puntos fuertes del candidato y redactas "
        "una carta breve, profesional y alineada con los requisitos reales del puesto. Como mucho de 150 palabas."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_local
)

IA_resumidor_proceso = Agent(
    role="Resumidor del Proceso",
    goal=(
        "Resumir de manera clara y concisa todo el proceso realizado por los agentes, explicando qué se ha hecho, qué resultados se han obtenido y cuál ha sido la decisión final."
    ),
    backstory=(
        "Eres un experto en síntesis de procesos complejos. Tienes capacidad para resumir de forma "
        "ordenada el trabajo realizado por varios agentes, eliminando información redundante y destacando "
        "solo los puntos más importantes. Tu objetivo es que el usuario entienda rápidamente qué CV se ha "
        "analizado, qué ofertas se han buscado, cuál ha sido la mejor oportunidad y qué documentos finales "
        "se han generado."
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
        "{cv_content}\n\n" 
        "Extraer y estructurar la información clave del CV, identificando habilidades, experiencia, formación y logros "
        "relevantes para el mercado laboral actual. "
    ),
    expected_output="CV del usuario estructurado en formato Markdown con su información de contacto, perfil profesional, " \
    "experiencia y formación. Asegurarse de que esté correctamente formateado usando la estructura basada en este ejemplo {ejemplo_cv}.",
    agent=IA_extractor_dades,
    output_file="outputs/cv_estructurado.md"
)

tarea_busqueda=Task(
    description=(f"Buscar ofertas que cumplan estos requisitos:\n{requisitos_usuario}\n\n"),
    expected_output=("Una lista de ofertas de trabajo que cumplan estrictamente los requisitos definidos "
    "por el usuario, incluyendo detalles clave como puesto, empresa, ubicación, condiciones, requisitos y justificación del encaje."
    "Incluye en enlace a la oferta original."),
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
    agent=IA_analista_oportunidades,
    context=[
        tarea_extraccion,
        tarea_busqueda
    ]
)


tarea_carta_mejor_oferta = Task(
    description=(
        "A partir del CV estructurado, las ofertas encontradas y el análisis de oportunidades, "
        "identificar la oferta con mayor nivel de encaje y redactar una única carta de presentación "
        "personalizada exclusivamente para esa oferta. "
        "No redactar cartas para ofertas secundarias ni generar versiones alternativas."
    ),
    expected_output=(
        "Carta de presentación personalizada únicamente para la mejor oferta seleccionada. "
        "Debe incluir: saludo profesional, breve presentación del candidato, relación directa "
        "entre el perfil del usuario y los requisitos de la oferta, motivación por el puesto "
        "y cierre formal."
    ),
    agent=IA_redactor_carta,
    context=[
        tarea_extraccion,
        tarea_busqueda,
        tarea_analisis
    ],
    output_file="outputs/carta_mejor_oferta.md" #eloutput+nombre
)

tarea_resumen_proceso = Task(
    description=(
        "Resumir todo el proceso realizado por el sistema multiagente de manera clara y concisa. "
        "Explicar qué información se ha extraído del CV, qué criterios se han usado para buscar ofertas, "
        "qué ofertas se han analizado, cuál ha sido seleccionada como mejor opción y qué documentos finales "
        "se han generado."
    ),
    expected_output=(
        "Resumen final del proceso en formato Markdown. Debe incluir: "
        "1) CV analizado, "
        "2) requisitos del usuario, "
        "3) búsqueda de ofertas, "
        "4) análisis y filtrado, "
        "5) mejor oferta seleccionada, "
        "6) carta de presentación generada, "
        "7) conclusión final breve."
    ),
    agent=IA_resumidor_proceso,
    context=[
        tarea_extraccion,
        tarea_busqueda,
        tarea_analisis,
        tarea_carta_mejor_oferta
    ],
    output_file="outputs/resumen_proceso.md"
)
#Inicialitzem la CREW
#-------------------------------------------------------------------
equipo= Crew(
    agents=[
        IA_extractor_dades,
        IA_cercador_feina,
        IA_analista_oportunidades,
        IA_redactor_carta,
        IA_resumidor_proceso
    ],
    tasks=[
        tarea_extraccion,
        tarea_busqueda,
        tarea_analisis,
        tarea_carta_mejor_oferta,
        tarea_resumen_proceso
    ],
    verbose=True
)

#EXECUCIO
#-------------------------------------
# Ejecución
resultado = equipo.kickoff(inputs={
    "cv_content": latex_text,
    "ejemplo_cv": ejemplo_cv,
    "requisitos_usuario": requisitos_usuario
    })









