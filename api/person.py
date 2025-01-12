# /api/person.py
from typing import List
from bson import Binary, ObjectId
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from mongo import MongoDatabase, serialize_mongo_document
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
import json
from sauce import FaceRecognitionSystem
import pickle
import torch

# includes recognize person, create person, validate answers, and update answer
router = APIRouter()


app = FastAPI()
pcs = set()  # Keep track of active peer connections

# Make a RTC route for video streaming and use FaceRecognitionSystem to recognize faces
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    face_system = FaceRecognitionSystem()
    recorder = MediaRecorder("recording.mp4")

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            pc.addTrack(track)
            recorder.addTrack(track)
            await recorder.start()
            
            while True:
                frame = await track.recv()

                face_result = face_system.process_frame(frame)
                if face_result:
                    await websocket.send_json({
                        "type": "face_detected",
                        "data": face_result
                    })

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "offer":
                offer = RTCSessionDescription(
                    sdp=message["sdp"],
                    type=message["type"]
                )
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pcs.discard(pc)
        await recorder.stop()
        await pc.close()

@router.post("/recognize_person")
async def recognize_person(
    eventId: str = Form(...),
    image: UploadFile = File(...),
):
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
    
    # convert image to file to bytes
    # image_bytes = await image.read()

    face_system = FaceRecognitionSystem()

    face_system.embeddings = {}

    people = await db.get_people_by_event(eventId)
    for person in people:
        eb = pickle.loads(pickle.loads(bytes(person["embeddings"])))
        print("Person:", person["name"], "Embeddings:", eb)
        face_system.embeddings[str(person["_id"])] = eb
        # print("Person:", person["name"], "Embeddings:", face_system.embeddings[str(person["_id"])])

    results = face_system.identify_person(image)

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

    face_system = FaceRecognitionSystem()

    pickle_outputs = face_system.process_uploaded_video(video_bytes, name)

    print("PICKLE OUTPUTS:", pickle_outputs)

    person = await db.insert_person({
        "eventId": ObjectId(eventId),
        "name": name,
        "answers": answers_list,
        "won": 0,
        "embeddings": Binary(pickle_outputs)
    })
    return {"person_id": str(person.inserted_id)}

# get user by id

@router.get("/person/{person_id}")
async def get_person(person_id: str):
    db = MongoDatabase()
    person = await db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return serialize_mongo_document(person)

# get user by name
@router.get("/person/name/{name}")
async def get_person_by_name(name: str):
    db = MongoDatabase()
    person = await db.get_person_by_name(name)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return serialize_mongo_document(person)