from fastapi import FastAPI, File
from fastapi import UploadFile

from fastapi.responses import HTMLResponse
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
import uuid
from retinaface import RetinaFace

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/multiple", StaticFiles(directory="multiple"), name="multiple")


@app.post("/face/")
async def create_files(file: UploadFile=File(...)):

    filename = f"static/{uuid.uuid4().hex}.jpeg"


    with open(filename, "wb") as f:
        f.write(file.file.read())

    print(file.filename, filename)

    obj = DeepFace.analyze(img_path=filename, actions = ['age', 'gender', 'race', 'emotion'])

    return {"result": obj}

@app.post("/multiple_faces")
async def create_multiple(file:UploadFile=File(...)):

    filename = f"multiple/{uuid.uuid4().hex}.jpeg"

    with open(filename,"wb") as f:
        f.write(file.file.read())
    
    print(file.filename, filename)

    faces = RetinaFace.extract_faces(filename)
    print(len(faces))

    # len_faces = len(faces)
    all_obj = []
    for face in faces:
    
        obj = DeepFace.analyze(face, detector_backend = 'skip',actions = ['age', 'gender', 'race', 'emotion'])

        all_obj.append(obj)
        # print(obj)
    return  {"result":all_obj}
