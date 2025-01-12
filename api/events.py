# /api/events.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mongo import MongoDatabase, serialize_mongo_document


# includes get all events, get event by ID, create event, and delete event
router = APIRouter()


@router.get("/events/all")
async def test():
    db = MongoDatabase()
    events = await db.get_all_events()
    serialized_events = [serialize_mongo_document(event) for event in events]
    return serialized_events


@router.get("/event/{event_id}")
async def get_event(event_id: str):
    db = MongoDatabase()
    event = await db.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return serialize_mongo_document(event)


class CreateEvent(BaseModel):
    name: str
    questions: list[str]


@router.post("/event")
async def create_event(event_data: CreateEvent):
    db = MongoDatabase()
    event = await db.insert_event(event_data.dict())
    return {"event_id": str(event.inserted_id)}


@router.delete("/event/{event_id}")
async def delete_event(event_id: str):
    db = MongoDatabase()
    result = await db.remove_event(event_id)
    if result:
        return {"message": "Event deleted successfully"}
    raise HTTPException(status_code=404, detail="Event not found")
