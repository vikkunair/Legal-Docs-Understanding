from typing import Union

from fastapi import FastAPI,Form
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from starlette.responses import RedirectResponse
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates



model = AutoModelForQuestionAnswering.from_pretrained('../../CUAD_Q&A/cuad-models/roberta-base/')
tokenizer = AutoTokenizer.from_pretrained('../../CUAD_Q&A/cuad-models/roberta-base/', use_fast=False)

#Input data model
class Item(BaseModel):
    input_string : str
    question:str


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# @app.get("/docs")
# async def main():
#     return RedirectResponse(url="/docs/")

@app.post("/submit")
async def submit(request: Request, question: str = Form(...), contracttext: str = Form(...)):
    
    encoding = tokenizer.encode_plus(text=question, text_pair=contracttext)
    inputs = encoding['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs)
    outputs = model(input_ids=torch.tensor([inputs]))

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
    answer.strip()
    answer_modified = "<mark><b>" + answer + "</b></mark>"

    print(answer_modified)

    contracttext = contracttext.replace(answer,answer_modified)
    print(contracttext)
    # return "text"
    return contracttext

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def create_item(question: str,input_string:str):

    encoding = tokenizer.encode_plus(text=question, text_pair=input_string)
    inputs = encoding['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs)
    outputs = model(input_ids=torch.tensor([inputs]))

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
    answer.strip()

    return {"start_index":int(start_index.numpy()),"end_index":int(end_index.numpy()),"answer":answer.strip(),"input_string":input_string}