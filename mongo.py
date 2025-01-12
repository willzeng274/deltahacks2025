import os
import pickle
from bson import Binary, ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from typing import Optional


class MongoDatabase:
    def __init__(self):
        load_dotenv()

        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI environment variable is not set.")

        self.client = AsyncIOMotorClient(uri, server_api=ServerApi("1"))

        try:
            # Ping the MongoDB server asynchronously
            self.client.admin.command("ping")
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

        self.db = self.client["arihan"]

    async def validate_event_data(self, data):
        """Validate event data before insertion."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if "name" not in data or not isinstance(data["name"], str):
            raise ValueError("Event 'name' is required and must be a string.")
        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError("Event 'questions' is required and must be a list.")
        return True

    async def validate_people_data(self, data):
        """Validate people data before insertion."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if "eventId" not in data or not isinstance(data["eventId"], ObjectId):
            raise ValueError("Field 'eventId' must be an ObjectId.")
        if "name" not in data or not isinstance(data["name"], str):
            raise ValueError("Field 'name' is required and must be a string.")
        if "answers" not in data or not isinstance(data["answers"], list):
            raise ValueError("Field 'answers' is required and must be a list.")
        if "won" not in data or not isinstance(data["won"], int):
            raise ValueError("Field 'won' is required and must be a boolean.")
        # Validate embeddings if they are present
        if "embeddings" in data:
            if not isinstance(data["embeddings"], Binary):
                raise ValueError("Field 'embeddings' must be of type Binary.")
        return True

    async def insert_event(self, event_data):
        """Insert event into the 'events' collection."""
        if await self.validate_event_data(event_data):
            return await self.db.events.insert_one(event_data)

    async def insert_person(self, person_data):
        """Insert person into the 'people' collection with embeddings."""
        if await self.validate_people_data(person_data):
            embeddings = person_data.pop("embeddings", None)
            event = await self.db.events.find_one({"_id": person_data["eventId"]})
            if event:
                answers = [""] * len(event["questions"])
                for index, i in enumerate(person_data["answers"]):
                    answers[index] = i

                person_data["answers"] = answers

                if embeddings:
                    # Serialize and store the embeddings as Binary data
                    person_data["embeddings"] = Binary(pickle.dumps(embeddings))

                return await self.db.people.insert_one(person_data)
            else:
                raise ValueError("Event not found.")
        else:
            raise ValueError("Invalid data for people insertion.")

    async def update_person_answer(self, person_id, question_index, new_answer):
        """Update a specific answer for a person."""
        if not isinstance(question_index, int) or question_index < 0:
            raise ValueError("Question index must be a non-negative integer.")

        person = await self.db.people.find_one({"_id": ObjectId(person_id)})
        if person:
            if 0 <= question_index < len(person["answers"]):
                person["answers"][question_index] = new_answer
                await self.db.people.update_one(
                    {"_id": ObjectId(person_id)},
                    {"$set": {"answers": person["answers"]}},
                )
                return True
            else:
                raise ValueError("Invalid question index.")
        else:
            raise ValueError("Person not found.")

    async def remove_event(self, event_id):
        """Remove an event and its associated people."""
        event = await self.db.events.find_one({"_id": ObjectId(event_id)})
        if event:
            await self.db.people.delete_many({"eventId": ObjectId(event_id)})
            await self.db.events.delete_one({"_id": ObjectId(event_id)})
            return True
        else:
            raise ValueError("Event not found.")

    async def get_event(self, event_id):
        """Get an event by its ID."""
        return await self.db.events.find_one({"_id": ObjectId(event_id)})
    
    async def get_event_by_name(self, name):
        """Get an event by its name."""
        return await self.db.events.find_one({"name": name})

    async def get_people_by_event(self, event_id):
        """Get all people associated with a specific event."""
        return [
            person
            async for person in self.db.people.find({"eventId": ObjectId(event_id)})
        ]

    async def get_person_embeddings(self, person_id):
        """Retrieve the embeddings of a person."""
        person = await self.db.people.find_one({"_id": ObjectId(person_id)})
        if person and "embeddings" in person:
            embeddings = pickle.loads(person["embeddings"])
            return embeddings
        return None
    
    async def get_person(self, person_id):
        """Get a person by their ID."""
        return await self.db.people.find_one({"_id": ObjectId(person_id)})
    
    async def get_person_by_name(self, name):
        """Get a person by their name."""
        return await self.db.people.find_one({"name": name})

    async def get_all_events(self):
        """Get all events."""
        return [event async for event in self.db.events.find()]
    
    async def increment_person_wins(self, person_id):
        """Increment the 'won' field of a person by 1."""
        return await self.db.people.update_one(
            {"_id": ObjectId(person_id)},
            {"$inc": {"won": 1}},
        )


# utils


def serialize_mongo_document(doc):
    """Convert MongoDB documents with ObjectId to JSON-serializable format."""
    if not doc:
        return None
    # doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
    # Convert the rest of the ObjectIDs to string
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
    return doc


# db = MongoDatabase()
#
#
# async def get_mongo_db() -> MongoDatabase:
#     db = MongoDatabase()
#     try:
#         yield db
#     finally:
#         db.client.close()
