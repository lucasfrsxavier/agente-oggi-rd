# 🤖 Agente de Integração — Oggi + RD

Este projeto implementa um **agente interno de perguntas e respostas (RAG)** para apoiar o time do **Estúdio Oggi** e da **RD Exclusive** durante o processo de fusão.

O agente responde dúvidas com base **exclusivamente** nos materiais internos disponibilizados (PDFs), como:

* manuais de processos da RD Exclusive
* materiais de treinamento da plataforma **Operand**

A aplicação foi construída em **Python**, utilizando **Streamlit** para a interface web e **LangChain 1.x (LCEL)** para orquestração do fluxo de RAG.

---

## 🎯 Objetivo do Projeto

* Centralizar o conhecimento operacional da RD Exclusive
* Reduzir dúvidas recorrentes do time
* Apoiar onboarding e consultas rápidas sobre processos e uso do Operand
* Garantir respostas confiáveis, baseadas apenas em documentação oficial

> ⚠️ O agente **não inventa respostas**. Se a informação não estiver nos documentos, ele informa claramente que não sabe.

---

## 🧠 Arquitetura (Visão Geral)

* **Interface:** Streamlit (chat web)
* **LLM:** Llama 3.1 8B (via Groq)
* **Embeddings:** sentence-transformers / all-MiniLM-L6-v2
* **Vector Store:** FAISS (in-memory)
* **Orquestração:** LangChain 1.x (LCEL)
* **Fonte de dados:** PDFs locais (pasta `docs/`)

Fluxo simplificado:

1. PDFs são lidos e transformados em chunks
2. Os chunks são vetorizados e armazenados no FAISS
3. A pergunta do usuário é usada para recuperar contexto relevante
4. O LLM responde com base **somente** nesse contexto

---

## 📁 Estrutura do Projeto

```
agente-oggi-rd/
│
├── app.py              # Aplicação Streamlit
├── requirements.txt    # Dependências do projeto
├── README.md           # Este arquivo
├── .gitignore
└── docs/               # PDFs usados como base de conhecimento
    ├── treinamento_operand.pdf
    └── manual_processos_rd.pdf
```

---

## ▶️ Como Rodar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/lucasfrsxavier/agente-oggi-rd.git
cd agente-oggi-rd
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GROQ_API_KEY=coloque_sua_chave_aqui
```

### 5. Adicionar os PDFs

Coloque os arquivos PDF na pasta `docs/`.

### 6. Executar a aplicação

```bash
streamlit run app.py
```

A aplicação ficará disponível em:

```
http://localhost:8501
```

---

## ☁️ Deploy (Streamlit Cloud)

O projeto está preparado para deploy via **Streamlit Community Cloud**:

* Repositório privado no GitHub
* Variável `GROQ_API_KEY` configurada em **Secrets** (formato TOML)
* Nenhuma dependência de arquivos locais fora do repositório

> Observação: o primeiro acesso pode demorar alguns segundos devido ao *cold start*.

---

## 🗣️ Tom de Voz do Agente

O agente foi configurado para atuar como:

* um colega de trabalho experiente
* prestativo e colaborativo
* claro e direto
* sem linguagem robótica ou formalidade excessiva

Sempre respeitando o escopo dos documentos.

---

## 🚧 Próximos Passos Planejados

* Persistência do FAISS (evitar reprocessar PDFs a cada deploy)
* Indicação de fonte/trecho do documento nas respostas
* Ajustes finos de recuperação de contexto

---

## 👤 Autor

Projeto desenvolvido por **Lucas Xavier**
IA Engineer — Estúdio Oggi

---

## 📌 Aviso Importante

Este agente é **exclusivamente para uso interno**.
As respostas refletem apenas os materiais fornecidos e **não substituem decisões formais, validações legais ou orientações de liderança**.