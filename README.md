# ISA — Backend com Gemini 3.5 Flash-Lite

Agora a ISA conversa de verdade, usando o Gemini 3.5 Flash-Lite (Google), e continua
com acesso a sites quando precisa consultar algo.

## O que roda aqui

1. Você abre a interface no navegador (servida pelo próprio Flask).
2. Você manda uma mensagem.
3. O backend chama o Gemini com o histórico da conversa.
4. Se o Gemini decidir que precisa ler um site, o backend busca a página e
   devolve o conteúdo pro Gemini continuar.
5. A resposta final volta pro navegador.

## Passo a passo

### 1. Conseguir a API key do Gemini

1. Acesse https://aistudio.google.com/apikey
2. Faça login com sua conta Google e gere uma chave.
3. Guarde essa chave.

### 2. Instalar as dependências

```bash
cd isa-backend
pip install -r requirements.txt
```

Se preferir isolar num ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate   # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar a chave

```bash
cp .env.example .env
```

Abra o `.env` e cole sua chave real:

```
GEMINI_API_KEY=sua_chave_aqui
```

### 4. Rodar

```bash
python app.py
```

Vai aparecer algo como `Running on http://127.0.0.1:5000`. Abra esse endereço
no navegador — a interface da ISA carrega direto, já conectada ao Gemini.

### 5. Testar

Manda uma mensagem qualquer, tipo:

```
Oi! Quem é você?
```

E pra testar o acesso a sites:

```
Acesse https://www.python.org e me diga do que se trata o site
```

## Sobre o seletor "Cerberus Think / Fast" na interface

Por enquanto os dois usam o mesmo modelo (Gemini 3.5 Flash-Lite) — o seletor é
só visual ainda. Se no futuro você quiser modelos diferentes de verdade pra
cada opção (ex: um mais rápido, outro que "pensa" mais), dá pra mapear cada
opção pra um `model` diferente no backend (o campo já é enviado pro
`/api/chat`, só falta o backend usar ele).

## Próximos passos (fora do escopo deste MVP)

- RAG de documentos
- Memória de conversas e perfil do usuário persistente (hoje reseta ao
  recarregar a página)
- Transcrição de áudio e arquivos
- Multi-usuário com controle de acesso

Esses itens já estão mapeados no documento de especificação do projeto.
