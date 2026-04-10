#CREWAI llibreria per instanciar i definir els agents
from crewai import Agent, Task, Crew, LLM

#LLAMA 3 com a llenguatge LLM instal·lat en local
llm_local = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434" # Port per defecte de Llama3
)
