from fastapi import FastAPI, File, UploadFile
from img2vec_pytorch import Img2Vec
from PIL import Image
from urllib.request import urlopen


app = FastAPI()


@app.post("/url")
async def root(url: str):
    img2vec = Img2Vec(cuda=False)
    img = Image.open(urlopen(url)).convert('RGB')
    vec = img2vec.get_vec(img)
    return {"success": vec.tolist()}


@app.post("/file")
async def say_hello(file: UploadFile):
    img2vec = Img2Vec(cuda=False)
    img = Image.open(file.file).convert('RGB')
    vec = img2vec.get_vec(img)
    return {"success": vec.tolist()}
