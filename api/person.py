# /api/person.py
import os
from bson import Binary, ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from mongo import MongoDatabase, serialize_mongo_document
import json
from sauce import FaceRecognitionSystem
import pickle
from pymongo import MongoClient
import asyncio
from openai import AsyncOpenAI

# includes recognize person, create person, validate answers, and update answer
router = APIRouter()

# get embeddings

load_dotenv()

ai_client = AsyncOpenAI()
uri = os.getenv("MONGODB_URI")

client = MongoClient(uri)
db = client["arihan"]

embeddings_dict = {}
people = db.people.find({})

for person in people:
    eb = pickle.loads(pickle.loads(bytes(person["embeddings"])))
    embeddings_dict[str(person["_id"])] = eb

face_system = FaceRecognitionSystem()

print("All embeddings:", embeddings_dict)

app = FastAPI()

@router.post("/recognize_person")
async def recognize_person(
    eventId: str = Form(...),
    image: UploadFile = File(...),
    # dependency for embeddings
    # face_system: FaceRecognitionSystem = Depends(FaceRecognitionSystem)
):
    # validate that eventId is a valid ObjectId
    try:
        ObjectId(eventId)
    except:
        raise HTTPException(status_code=400, detail="Invalid eventId")
    
    # removed all for performance
    # db = MongoDatabase()

    # check if event exists
    # event = await db.get_event(eventId)
    # if not event:
    #     raise HTTPException(status_code=404, detail="Event not found")
    
    # convert image to file to bytes
    # image_bytes = await image.read()

    # face_system = FaceRecognitionSystem()

    # face_system.embeddings = {}

    # people = await db.get_people_by_event(eventId)
    # for person in people:
    #     eb = pickle.loads(pickle.loads(bytes(person["embeddings"])))
    #     print("Person:", person["name"], "Embeddings:", eb)
    #     face_system.embeddings[str(person["_id"])] = eb
    #     # print("Person:", person["name"], "Embeddings:", face_system.embeddings[str(person["_id"])])

    results = await asyncio.to_thread(face_system.identify_person, image.file.read())
    # results = face_system.identify_person(image)

    print("Results:", results)

    return results
    

@router.post("/create_person")
async def create_person(
    name: str = Form(...),
    eventId: str = Form(...),
    answers: str = Form(...),  # JSON-encoded string
    video: UploadFile = File(...),
):
    # decode
    try:
        answers_list = json.loads(answers)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format for answers")
    
    # Validate that answers_list is a list of strings
    if not isinstance(answers_list, list):
        raise HTTPException(status_code=400, detail="Answers must be a list")
    
    if not all(isinstance(answer, str) for answer in answers_list):
        raise HTTPException(status_code=400, detail="All answers must be strings")

    # validate that eventId is a valid ObjectId
    try:
        ObjectId(eventId)
    except:
        raise HTTPException(status_code=400, detail="Invalid eventId")
    
    db = MongoDatabase()

    # check if event exists
    event = await db.get_event(eventId)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # make sure # of answers match # of questions
    # answers_list = answers.split(",")
    print(answers_list, len(answers_list))
    if len(answers_list) != len(event["questions"]):
        raise HTTPException(status_code=400, detail="Number of answers must match number of questions. The number: " + str(len(answers_list)) + " vs " + str(len(event["questions"])))

    # convert video to file to bytes
    video_bytes = await video.read()

    # use a thread pool executor

    # pickle_outputs = face_system.process_uploaded_video(video_bytes, name)
    pickle_outputs = await asyncio.to_thread(face_system.process_uploaded_video, video_bytes, name)

    print("PICKLE OUTPUTS:", pickle_outputs)

    person = await db.insert_person({
        "eventId": ObjectId(eventId),
        "name": name,
        "answers": answers_list,
        "won": 0,
        "embeddings": Binary(pickle_outputs)
    })

    face_system.embeddings[str(person.inserted_id)] = pickle.loads(pickle_outputs)
    return {"person_id": str(person.inserted_id)}

# get user by id

@router.get("/person/{person_id}")
async def get_person(person_id: str):
    try:
        ObjectId(person_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    db = MongoDatabase()
    person = await db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    # print("Serialized person:", serialize_mongo_document(person))
    del person["embeddings"]
    return serialize_mongo_document(person)

# get user by name
@router.get("/person/name/{name}")
async def get_person_by_name(name: str):
    db = MongoDatabase()
    person = await db.get_person_by_name(name)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    del person["embeddings"]
    return serialize_mongo_document(person)

@router.get("/person/{person_id}/win")
async def win_person(person_id: str):
    try:
        ObjectId(person_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    db = MongoDatabase()
    await db.increment_person_wins(person_id)
    return {"message": "Person wins incremented"}

class AnswerRequest(BaseModel):
    question_index: int
    answer: str

@router.post("/person/{person_id}/validate_answer")
async def validate_answer(
    person_id: str,
    answer_request: AnswerRequest
):
    question_index = answer_request.question_index
    answer = answer_request.answer
    try:
        ObjectId(person_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    db = MongoDatabase()
    person = await db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    if 0 <= question_index < len(person["answers"]):
        # use open ai to validate answer

        system_prompt = "You are a helpful assistant. Your task is to evaluate if the answer is correct, given the correct answer and user's answer.\nYou must answer 'true' or 'false'"
        user_prompt = f"Correct answer: {person["answers"][question_index]}\nUser's answer: {answer}\nIs the user's answer correct? Be leniant. For example, 'I am gluten-free' and 'gluten-free' are the same."

        chat_completion = await ai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="gpt-4o",
            max_tokens=10,
            temperature=0.0
        )

            # Extract the response and return a boolean based on its content
        answer = chat_completion.choices[0].message.content.lower().strip()

        return answer in ['true', 'yes']
    raise HTTPException(status_code=400, detail="Invalid question index.")

active_connections: list[WebSocket] = []

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    # Accept connection
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Receive the frame from Unity WebSocket (JPEG image bytes)
            frame_data = await websocket.receive_bytes()
            # You can process the frame here, such as performing classification or saving it.
            # For simplicity, we're just logging the frame size.
            print(f"Received frame of size: {len(frame_data)} bytes")

            results = await asyncio.to_thread(face_system.identify_person, frame_data)

            await websocket.send_text(json.dumps(results))
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected")

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up on shutdown
    for connection in active_connections:
        await connection.close()