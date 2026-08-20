"""
ISA - Backend com Gemini 3.5 Flash-Lite + memória real via MongoDB

Não tem tela de login (isso fica pra uma próxima etapa, com código de
convite). Por enquanto, cada navegador que acessa ganha um identificador
próprio (guardado num cookie de sessão assinado) e, a partir dele:

  - o perfil (nome, foto, persona) fica salvo e é recuperado depois
  - cada conversa é salva no MongoDB, mensagem por mensagem
  - a lista de conversas do usuário é devolvida pro frontend popular a
    sidebar, permitindo voltar de onde parou
"""

import os
import logging
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from pymongo import MongoClient, DESCENDING

from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MONGODB_URI = os.getenv("MONGODB_URI")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Falta a variável GEMINI_API_KEY. Configure ela no arquivo .env "
        "(veja .env.example). Você pode gerar uma chave em "
        "https://aistudio.google.com/apikey"
    )

if not MONGODB_URI:
    raise RuntimeError(
        "Falta a variável MONGODB_URI. Configure ela no arquivo .env "
        "com a connection string do seu cluster do MongoDB Atlas."
    )

if not FLASK_SECRET_KEY:
    raise RuntimeError(
        "Falta a variável FLASK_SECRET_KEY. Gere uma com:\n"
        "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "e coloque o resultado no .env."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("isa-backend")

client = genai.Client(api_key=GEMINI_API_KEY)

BASE_SYSTEM_PROMPT = (
    "Você é a ISA, uma assistente de IA que fala português do Brasil. "
    "Quando precisar de informações de um site específico para responder, "
    "use a ferramenta fetch_website. Seja direta, clara e simpática."
)

MAX_CHARS = 8000  # limite de caracteres extraídos de cada página

# --------------------------------------------------------------------------
# MongoDB
# --------------------------------------------------------------------------

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client.get_database("isa_db")

users_col = db["users"]
conversations_col = db["conversations"]

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

# --------------------------------------------------------------------------
# Conversa com o Gemini (com suporte a tool use e persona por usuário)
# --------------------------------------------------------------------------

def build_system_prompt(persona: str) -> str:
    if persona:
        return (
            BASE_SYSTEM_PROMPT
            + "\n\nInstruções extras definidas pelo usuário sobre como você deve agir: "
            + persona
        )
    return BASE_SYSTEM_PROMPT


def ask_isa(history: list[dict], persona: str = "") -> str:
    contents = []
    for msg in history:
        if not msg.get("text"):
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["text"])]))

    if not contents:
        return "Não recebi nenhuma mensagem."

    generation_config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(persona),
        tools=[fetch_tool],
    )

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
# Identidade do usuário (sem login, só um cookie de sessão persistente)
# --------------------------------------------------------------------------

def get_or_create_user():
    """
    Cada navegador ganha um usuário próprio na primeira visita. O id fica
    guardado num cookie de sessão assinado (não dá pra falsificar sem a
    FLASK_SECRET_KEY) e persiste por 1 ano.
    """
    user_id = session.get("user_id")
    if user_id:
        try:
            user = users_col.find_one({"_id": ObjectId(user_id)})
            if user:
                return user
        except InvalidId:
            pass

    doc = {
        "name": "",
        "persona": "",
        "photo": "",
        "created_at": datetime.utcnow(),
    }
    result = users_col.insert_one(doc)
    doc["_id"] = result.inserted_id

    session.permanent = True
    session["user_id"] = str(doc["_id"])
    return doc


def serialize_user(user: dict) -> dict:
    return {
        "name": user.get("name", ""),
        "persona": user.get("persona", ""),
        "photo": user.get("photo", ""),
    }


# --------------------------------------------------------------------------
# Servidor Flask
# --------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.permanent_session_lifetime = timedelta(days=365)


@app.route("/")
def index():
    get_or_create_user()  # garante o cookie já na primeira carga da página
    return render_template("index.html")


@app.route("/api/me")
def me():
    user = get_or_create_user()
    return jsonify({"user": serialize_user(user)})


@app.route("/api/profile", methods=["POST"])
def update_profile():
    user = get_or_create_user()
    data = request.get_json(force=True) or {}

    updates = {}
    if "name" in data:
        updates["name"] = (data["name"] or "").strip()
    if "persona" in data:
        updates["persona"] = (data["persona"] or "").strip()
    if "photo" in data:
        updates["photo"] = data["photo"] or ""

    if updates:
        users_col.update_one({"_id": user["_id"]}, {"$set": updates})

    updated = users_col.find_one({"_id": user["_id"]})
    return jsonify({"user": serialize_user(updated)})


@app.route("/api/conversations")
def list_conversations():
    user = get_or_create_user()
    docs = conversations_col.find({"user_id": user["_id"]}).sort("updated_at", DESCENDING)

    result = []
    for d in docs:
        result.append({
            "client_id": d.get("client_id"),
            "title": d.get("title") or "Nova conversa",
            "messages": [
                {"role": m.get("role"), "text": m.get("text", "")}
                for m in d.get("messages", [])
            ],
        })
    return jsonify({"conversations": result})


@app.route("/api/chat", methods=["POST"])
def chat():
    user = get_or_create_user()
    data = request.get_json(force=True) or {}
    history = data.get("history", [])
    conversation_id = data.get("conversation_id")

    try:
        reply = ask_isa(history, persona=user.get("persona", ""))
    except Exception as e:
        logger.exception("Erro ao falar com o Gemini")
        reply = f"Ocorreu um erro ao falar com o Gemini: {e}"

    # Memória: salva a troca de mensagens no MongoDB, vinculada ao usuário
    # e à conversa (client_id gerado no navegador).
    if conversation_id and history:
        now = datetime.utcnow()
        last_user_msg = history[-1] if history[-1].get("role") == "user" else None

        new_messages = []
        if last_user_msg:
            new_messages.append({
                "role": "user",
                "text": last_user_msg["text"],
                "created_at": now,
            })
        new_messages.append({"role": "bot", "text": reply, "created_at": now})

        title = (last_user_msg["text"][:42] if last_user_msg else "Nova conversa")

        conversations_col.update_one(
            {"client_id": conversation_id, "user_id": user["_id"]},
            {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "user_id": user["_id"],
                    "client_id": conversation_id,
                    "title": title,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
