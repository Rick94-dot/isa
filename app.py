"""
ISA - Backend com Gemini 3.5 Flash-Lite

Fluxo:
  Frontend manda o histórico da conversa (POST /api/chat)
    -> Backend chama o Gemini
    -> Se o Gemini pedir a ferramenta fetch_website, o backend busca o site
       e devolve o conteúdo pro Gemini
    -> Backend devolve a resposta final pro frontend
"""

import os
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Falta a variável GEMINI_API_KEY. Configure ela no arquivo .env "
        "(veja .env.example). Você pode gerar uma chave em "
        "https://aistudio.google.com/apikey"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("isa-backend")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "Você é a ISA, uma assistente de IA que fala português do Brasil. "
    "Quando precisar de informações de um site específico para responder, "
    "use a ferramenta fetch_website. Seja direta, clara e simpática."
)

MAX_CHARS = 8000  # limite de caracteres extraídos de cada página

# --------------------------------------------------------------------------
# Ferramenta: buscar conteúdo de um site
# --------------------------------------------------------------------------

def fetch_website(url: str) -> str:
    """Baixa uma página e retorna o texto limpo (sem tags, scripts, etc.)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ISABot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > MAX_CHARS:
            clean_text = clean_text[:MAX_CHARS] + "\n[...conteúdo truncado...]"

        return clean_text if clean_text else "A página não retornou conteúdo de texto."

    except requests.exceptions.RequestException as e:
        return f"Erro ao acessar o site: {e}"


fetch_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="fetch_website",
            description=(
                "Busca o conteúdo de texto de uma página web a partir de uma URL. "
                "Use quando precisar ler ou analisar o conteúdo de um site específico "
                "para responder à pergunta do usuário."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "url": types.Schema(
                        type="STRING",
                        description="A URL completa do site, incluindo https://",
                    )
                },
                required=["url"],
            ),
        )
    ]
)

generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[fetch_tool],
)

# --------------------------------------------------------------------------
# Conversa com o Gemini (com suporte a tool use)
# --------------------------------------------------------------------------

def ask_isa(history: list[dict]) -> str:
    """
    history: lista de {"role": "user"|"bot", "text": "..."} vinda do frontend.
    Retorna o texto final da resposta da ISA.
    """
    contents = []
    for msg in history:
        if not msg.get("text"):
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["text"])]))

    if not contents:
        return "Não recebi nenhuma mensagem."

    # Loop de tool use: repete enquanto o modelo continuar pedindo ferramentas
    for _ in range(5):  # limite de segurança contra loops infinitos
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generation_config,
        )

        candidate = response.candidates[0]
        function_call = None
        text_parts = []

        for part in candidate.content.parts:
            if part.function_call:
                function_call = part.function_call
            elif part.text:
                text_parts.append(part.text)

        if function_call is None:
            return "".join(text_parts) or "Não consegui gerar uma resposta."

        # O modelo pediu a ferramenta: registramos a fala dele e executamos
        contents.append(candidate.content)

        if function_call.name == "fetch_website":
            url = function_call.args.get("url", "")
            logger.info(f"Buscando site: {url}")
            result_text = fetch_website(url)
        else:
            result_text = f"Ferramenta desconhecida: {function_call.name}"

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result_text},
                    )
                ],
            )
        )

    return "Cheguei no limite de passos tentando responder. Tente reformular a pergunta."


# --------------------------------------------------------------------------
# Servidor Flask
# --------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    history = data.get("history", [])

    try:
        reply = ask_isa(history)
    except Exception as e:
        logger.exception("Erro ao falar com o Gemini")
        reply = f"Ocorreu um erro ao falar com o Gemini: {e}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
