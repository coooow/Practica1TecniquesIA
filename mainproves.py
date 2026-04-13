#CREWAI llibreria per instanciar i definir els agents
from crewai import Agent, Task, Crew, LLM

#LLAMA 3 com a llenguatge LLM instal·lat en local
llm_local = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434" # Port per defecte de Llama3
)

#Inicialitzem els agents amb el model local.
#------------------------------------------------------------------------

#Agente para buscar ofertas de trabajo abiertas que cumplan estrictamente los requisitos definidos por el usuario.
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

tarea_busqueda=Task(
    description="Buscar ofertas de trabajo abiertas que cumplan estrictamente los requisitos definidos por el usuario.",
    expected_output="Una lista de ofertas de trabajo que cumplan estrictamente los requisitos definidos por el usuario, incluyendo detalles clave como puesto, empresa, ubicación, condiciones, requisitos y justificación del encaje."
    agent=IA_cercador_feina
)

tarea_analisis=Task(
    description="Analizar las ofertas recopiladas y filtrar aquellas que mejor se ajustan al perfil del usuario.",
    expected_output="Una lista priorizada de ofertas de trabajo que mejor se ajustan al perfil del usuario, con una justificación detallada del nivel de ajuste para cada una."
    agent=IA_analista_oportunidades
)   








