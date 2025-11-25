# 🚀 Agentes Pesquisadores com CrewAI + LlamaIndex + RAG + Web Search
Sistema multiagente para pesquisa científica automatizada com arXiv, Tavily, RAG local e inferência LLM Groq (Llama 3.3-70B)
---

# 📌 Resumo do Projeto

Este projeto implementa um sistema completo de agentes inteligentes capaz de:

Pesquisar artigos no arXiv

Buscar informações científicas atualizadas na web

Validar documentos via RAG local com LlamaIndex

Orquestrar pesquisas com CrewAI (processo hierárquico)

Fornecer respostas estruturadas e contextualizadas

Expor tudo em uma interface Gradio multilayout
---
# 🧠 Arquitetura Geral
flowchart TD

    A[Usuário] --> B[Gradio UI<br>Aba 1: Pesquisa por tema (CrewAI)<br>Aba 2: Perguntas sobre documentos locais (RAG)]

    B --> C[CrewAI - Processo Hierárquico<br>Agente Gerente]

    C --> D1[Agente arXiv<br>Busca artigos científicos]
    C --> D2[Agente Web / Tavily<br>Busca na web]
    C --> D3[Agente Verificador<br>RAG local com LlamaIndex]

    D3 --> E[LlamaIndex + Vetores Locais<br>Embeddings • Similaridade • k-NN<br>Respostas ancoradas em dados]

    E --> F[LLM Groq<br>Llama 3.3-70B Versatile]

---
# ⭐ Principais Funcionalidades
## ✅ 1. Pesquisa Científica Multicanal (CrewAI)
O sistema coordena 3 agentes:
| Agente                     | Função                                 |
| -------------------------- | -------------------------------------- |
| **Agente arXiv**           | Busca artigos científicos relevantes   |
| **Agente Web (Tavily)**    | Busca artigos modernos na web          |
| **Agente Verificador RAG** | Valida fontes usando documentos locais |
---
# ✅ 2. RAG Local com LlamaIndex

Suporte a múltiplas bases de conhecimento:

./artigo_data

./livro_data

Usado para:

Validação documental

Respostas contextualizadas

Consulta direta via interface
---
# ✅ 3. Web Search Inteligente (Tavily API)

Integração para respostas sempre atualizadas:

Papers recentes

Notas técnicas

Relatórios científicos

Resumos de páginas confiáveis
---
# ✅ 4. Cálculo de Métricas e Ferramentas Auxiliares

Inclui ferramentas como:

## 📊 Cálculo de engajamento

## 🔍 Consulta e download de PDFs do arXiv

## 🌐 Pesquisa supervisionada com Tavily
---
# ✅ 5. Interface Gradio Profissional

Duas abas completamente independentes:

Pesquisa Acadêmica Multiagente (CrewAI)

Consulta de Documentos via RAG
---
# 📂 Estrutura do Projeto
agentes_pesquisadores/
│
├── app.py                 # Código principal
├── artigo_data/           # Vetores/índices RAG - artigos (persist)
├── livro_data/            # Vetores/índices RAG - livro (persist)
├── requirements.txt
└── README.md
---
# 🛠️ Tecnologias Utilizadas
## 🧠 LLMs

Groq (Llama-3.3-70B-versatile)

LlamaIndex QueryEngine

CrewAI LLM wrapper
---
## 🔧 Frameworks

CrewAI (Sistema multiagente profissional)

LlamaIndex (RAG, embeddings, query engines)

Gradio (UI)

Tavily API (Pesquisa Web)

arXiv API

dotenv
---
# 📦 Outras libs

requests

Python typing

FunctionTool, LlamaIndexTool
---
# 🧪 Como Executar Localmente

1. Clonar o repositório
 git clone https://github.com/matheusbgomes4/agentes_pesquisadores
cd agentes_pesquisadores

2. Criar o ambiente virtual
python -m venv venv
source venv/bin/activate

3. Instalar dependências
pip install -r requirements.txt

4. Configurar variáveis
GROQ_API_KEY=xxxx
TAVILY_API_KEY=xxxx

5. Rodar o sistema
python app.py

A interface abrirá em:
http://127.0.0.1:7860
---
# 📈 Demonstração Visual

<img width="1613" height="541" alt="image" src="https://github.com/user-attachments/assets/9c9b4458-684d-40f6-b2dd-1b9eec0a20ce" />
<img width="1717" height="524" alt="image" src="https://github.com/user-attachments/assets/67a64f09-4015-4807-9d3d-0d2f883e4afe" />

---
