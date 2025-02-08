from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = FastAPI()

templates = Jinja2Templates(directory="templates")

template = """
                You are a helpful AI assistant. Provide a clear, concise, and user-friendly response to the following question:
                Question: {question}
                Answer:
        """
model = OllamaLLM(model="deepseek-coder")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.get("/")
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(data: dict):
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    # Process the question with DeepSeek
    response = chain.invoke({"question": question})

    return JSONResponse({"answer": response})
