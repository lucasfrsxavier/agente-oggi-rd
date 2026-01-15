import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain 1.x (LCEL)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 1. Configuração da Página
# -----------------------------
st.set_page_config(page_title="Agente Oggi & RD", page_icon="🤖")
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("A chave da GROQ não foi encontrada. Verifique o arquivo .env")
    st.stop()

# -----------------------------
# 2. Base de Conhecimento (PDFs)
# -----------------------------
@st.cache_resource
def get_knowledge_base():
    pdf_folder = "docs"
    all_text = ""

    if not os.path.exists(pdf_folder):
        st.error(f"Pasta '{pdf_folder}' não encontrada.")
        return None

    files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not files:
        st.error("Nenhum PDF encontrado na pasta 'docs'.")
        return None

    for pdf_file in files:
        reader = PdfReader(os.path.join(pdf_folder, pdf_file))
        for page in reader.pages:
            all_text += page.extract_text() or ""

    if not all_text.strip():
        st.warning("Não foi possível extrair texto dos PDFs.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore

# -----------------------------
# 3. Interface
# -----------------------------
st.title("🤖 Agente de Integração: Oggi + RD")
st.markdown("Agente alimentado por **Llama 3 (via Groq)** usando LangChain 1.x (LCEL).")

with st.spinner("Processando base de conhecimento..."):
    vectorstore = get_knowledge_base()

if vectorstore:

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        Você é um assistente interno experiente que apoia o time da Oggi e da RD Exclusive durante o processo de fusão.

        Você atua como um colega de trabalho prestativo e confiável, ajudando outras pessoas do time a esclarecer dúvidas sobre:
        - processos internos da RD Exclusive
        - uso da plataforma Operand
        - fluxos operacionais descritos nos materiais de treinamento e manuais disponíveis

        Seu tom de voz deve ser:
        - profissional, mas próximo
        - claro e direto
        - colaborativo, como alguém que quer ajudar de verdade
        - sem formalidade excessiva ou linguagem robótica

        Diretrizes de resposta:

        1. Use prioritariamente as informações presentes no contexto fornecido.
        2. Nunca invente procedimentos, regras ou políticas que não estejam explícitas ou claramente implícitas nos documentos.
        3. Quando a resposta estiver bem definida no contexto:
            - explique de forma clara, prática e, quando fizer sentido, em passos.
        4. Quando a resposta não estiver totalmente coberta pelo contexto:
            - deixe isso claro de forma natural (ex: “o material que temos hoje não entra nesse nível de detalhe”).
            - ajude o usuário a entender como esse tipo de situação normalmente funciona, sem afirmar que seja o procedimento oficial.
            - se necessário, faça perguntas para entender melhor a situação antes de responder.
        5. Se a dúvida for genérica ou ambígua, peça esclarecimentos antes de assumir qualquer coisa.
        6. Quando a resposta depender de permissões, configurações específicas ou decisões de outra área, deixe isso explícito e oriente o usuário sobre quem normalmente cuida desse tipo de assunto.
        7. Se o tema estiver fora do escopo dos documentos e do sistema Operand, diga isso de forma clara e direta.

        Regras de linguagem:
        - Não use emojis.
        - Não seja excessivamente formal.
        - Não use respostas evasivas ou genéricas.
        - Fale como alguém do time falando com outro alguém do time.

        Contexto:
        {context}

        Pergunta:
        {question}

        """
    )

    # LCEL RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Pergunte sobre os manuais..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Pesquisando..."):
                try:
                    answer = rag_chain.invoke(user_input)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")
