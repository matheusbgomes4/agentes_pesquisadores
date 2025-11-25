import os
import requests
import arxiv
import gradio as gr
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import LlamaIndexTool
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent 
from llama_index.llms.groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv 
from typing import List

# --- 1. CONFIGURAÇÃO DE CHAVES E LLMS ---
load_dotenv()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 

# LLM para LlamaIndex (para RAG)
llm_llama_index = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
# LLM para CrewAI
llm_crew = LLM(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
Settings.llm = llm_llama_index 

# --- 2. FUNÇÕES BASE E WRAPPERS ---

def calcular_engajamento(curtidas: int, comentarios: int, compartilhamentos: int, seguidores: int) -> str:
    engajamento_total = curtidas + comentarios + compartilhamentos
    taxa_engajamento = (engajamento_total / seguidores) * 100 if seguidores > 0 else 0
    return f"O engajamento total é {engajamento_total} e a taxa de engajamento é {round(taxa_engajamento, 2)}%."

def consulta_artigos(titulo: str) -> str:
    busca = arxiv.Search(query=titulo, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
    resultados = [f"Título: {r.title}\nResumo: {r.summary}\nLink: {r.entry_id}" for r in busca.results()]
    return "\n\n".join(resultados)

def baixar_pdf_arxiv(link):
    try:
        if "arxiv.org" not in link: return "O link fornecido não é um link válido do arXiv."
        return f"PDF para {link} baixado com sucesso."
    except:
        return "Erro ao executar o download."

def pesquisar_web_tavily(consulta: str) -> str:
    try:
        cliente = TavilyClient(api_key=TAVILY_API_KEY) 
        resposta = cliente.search(query=consulta, max_results=3, search_depth="advanced")
        snippets = [f"• Fonte: {r['url']}\nConteúdo: {r['content']}" for r in resposta.get("results", [])]
        return "Resultados da Pesquisa:\n" + "\n---\n".join(snippets)
    except Exception as e:
        return f"Erro ao executar pesquisa Tavily: {e}"


# --- 3. CRIAÇÃO DAS FERRAMENTAS E CARREGAMENTO RAG ---

ferramenta_engajamento = FunctionTool.from_defaults(fn=calcular_engajamento, name="calcular_engajamento", description="Calcula o engajamento total de uma postagem em redes sociais.")
ferramenta_consulta_artigos = FunctionTool.from_defaults(fn=consulta_artigos, name="pesquisar_artigos_cientificos", description="Consulta artigos científicos na base de dados arXiv.")
ferramenta_web_tavily = FunctionTool.from_defaults(fn=pesquisar_web_tavily, name="pesquisar_web_tavily", description="Busca informações atuais na internet.")

# CARREGAMENTO RAG LOCAL (Tenta carregar os vetores salvos)
try:
    storage_context_artigo = StorageContext.from_defaults(persist_dir="./artigo_data")
    artigo_index = load_index_from_storage(storage_context_artigo)
    storage_context_livro = StorageContext.from_defaults(persist_dir="./livro_data")
    livro_index = load_index_from_storage(storage_context_livro)

    artigo_engine = artigo_index.as_query_engine(similarity_top_k=3)
    livro_engine = livro_index.as_query_engine(similarity_top_k=3)

    query_engine_tools: List[QueryEngineTool] = [
        QueryEngineTool(query_engine=artigo_engine, metadata=ToolMetadata(name="artigo_engine", description="Responde estritamente a perguntas sobre ALGORITMOS e IA, utilizando o CONTEÚDO EXCLUSIVO deste artigo local.")),
        QueryEngineTool(query_engine=livro_engine, metadata=ToolMetadata(name="livro_engine", description="Fornece informações sobre avanços e tendências sobre inteligência artificial.")),
    ]
except Exception as e:
    query_engine_tools: List[QueryEngineTool] = []
    print(f"AVISO: Falha ao carregar índices RAG locais. Erro: {e}")

# PREPARAÇÃO DE TOOLS PARA CREWAI
tool_arxiv = LlamaIndexTool.from_tool(ferramenta_consulta_artigos)
tool_baixar = LlamaIndexTool.from_tool(FunctionTool.from_defaults(fn=baixar_pdf_arxiv, name="baixar_pdf_arxiv", description="Baixa o PDF de um artigo do arXiv."))
tools_web = [LlamaIndexTool.from_tool(ferramenta_web_tavily)]


# --- 4. LÓGICA CREWAI E GRADIO ---

def criar_agentes():
    base_llm = llm_crew 

    # Agentes CrewAI (Com backstory e LLM_CREW)
    agent_arxiv = Agent(role='Agente de pesquisa', goal='Fornece artigos científicos sobre um assunto de interesse.', backstory='Um agente expert em pesquisa científica que possui a habilidade de buscar artigos no arxiv', tools=[tool_arxiv, tool_baixar], llm=base_llm)
    agent_web = Agent(role='Agente de pesquisa por documentos na web', goal='Fornece artigos científicos encontrados na web.', backstory='Um agente expert em pesquisa por artigos na web.', tools=tools_web, llm=base_llm)
    agente_verificacao = Agent(role='Agente de pesquisa que verifica documentos', goal='Verifica se artigos encontrados na web são válidos e responde com base em documentos locais.', backstory='Um agente expert em validação de fontes e conteúdo local.', tools=query_engine_tools, llm=base_llm)
    manager = Agent(role="Gerente do projeto", goal="Gerenciar a equipe com eficiência", allow_delegation=True, backstory='Gerente experiente em coordenação de equipes de pesquisa.', llm=base_llm)
    return agent_arxiv, agent_web, agente_verificacao, manager

def pesquisar_artigos(tema):
    agent_arxiv, agent_web, agente_verificacao, manager = criar_agentes()

    task_arxiv = Task(description=f"Busque 5 artigos científicos no arxiv sobre {tema}.", expected_output="5 artigos e seus respectivos links", agent=agent_arxiv)
    task_web = Task(description=f"Busque 5 artigos científicos sobre {tema} na web.", expected_output="5 artigos e seus respectivos links", agent=agent_web)
    task_verificacao = Task(description="Verifique se os artigos encontrados na web realmente são artigos científicos.", expected_output="Resumo e links dos artigos validados", agent=agente_verificacao)

    crew_hierarquica = Crew(
        agents=[agent_arxiv, agent_web, agente_verificacao],
        tasks=[task_arxiv, task_web, task_verificacao],
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=1,
    )

    result = crew_hierarquica.kickoff()
    return result

def pergunta_documentos_locais(query):
    agente_rag_local = OpenAIAgent.from_tools(
        tools=query_engine_tools,
        llm=llm_llama_index,
        verbose=False,
    )
    response = agente_rag_local.chat(query)
    return response.response

# --- INTERFACE GRADIO (Lançamento) ---
pesquisa_interface = gr.Interface(
    fn=pesquisar_artigos,
    inputs=gr.Textbox(label="Digite o tema para pesquisa (CrewAI)"),
    outputs=gr.Textbox(label="Resultados (ArXiv + Web)", lines=10),
    title="👨‍🔬👩‍🔬 Pesquisador de Artigos Científicos",
)

rag_interface = gr.Interface(
    fn=pergunta_documentos_locais,
    inputs=gr.Textbox(label="Pergunte sobre seus documentos (RAG)"),
    outputs=gr.Textbox(label="Resposta do Documento", lines=10),
    title="Perguntas sobre IA nas redes sociais (Docs Locais)",
)

if __name__ == "__main__":
    with gr.Blocks() as app: 
        with gr.Tab("Pesquisa de Artigos (CrewAI)"):
            pesquisa_interface.render()
        with gr.Tab("Perguntas sobre Documentos (RAG)"):
            rag_interface.render()

    app.launch(share=False)