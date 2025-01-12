import pickle
from mongo import MongoDatabase
from bson import Binary


async def test_database():
    db = MongoDatabase()

    event_data = {
        "name": "Sample Event",
        "questions": ["What is your name?", "What is your age?"],
    }

    # Insert event
    event_inserted = await db.insert_event(event_data)
    print(f"Inserted event with ID: {event_inserted.inserted_id}")

    # Get event from DB
    event_from_db = await db.get_event(event_inserted.inserted_id)
    assert event_from_db["name"] == event_data["name"], "Event name mismatch"
    assert (
        event_from_db["questions"] == event_data["questions"]
    ), "Event questions mismatch"

    person_data = {
        "eventId": event_inserted.inserted_id,
        "name": "John Doe",
        "answers": [],
    }

    embeddings = [0.1, 0.2, 0.3]

    serialized_embeddings = pickle.dumps(embeddings)

    person_data["embeddings"] = Binary(serialized_embeddings)
    person_inserted = await db.insert_person(person_data)
    print(f"Inserted person with ID: {person_inserted.inserted_id}")

    person_get = await db.db.people.find_one({"_id": person_inserted.inserted_id})
    print(f"Fetched person: {person_get}")

    assert person_get["name"] == person_data["name"], "Person name mismatch"
    assert person_get["eventId"] == person_data["eventId"], "Event ID mismatch"
    assert person_get["answers"] == [
        "",
        "",
    ]
    assert "embeddings" in person_get, "Embeddings not found in the person document"

    retrieved_embeddings = pickle.loads(person_get["embeddings"])
    assert retrieved_embeddings == embeddings, "Embeddings mismatch"

    update_result = await db.update_person_answer(person_inserted.inserted_id, 1, "30")
    print(f"Updated person answer: {update_result}")

    updated_person = await db.db.people.find_one({"_id": person_inserted.inserted_id})
    assert updated_person["answers"][1] == "30", "Answer update failed"

    event = await db.get_event(event_inserted.inserted_id)
    print(f"Fetched event: {event}")
    assert event["name"] == event_data["name"], "Event name mismatch"
    assert event["questions"] == event_data["questions"], "Event questions mismatch"

    people = await db.get_people_by_event(event_inserted.inserted_id)
    print(f"People in event: {people}")
    assert len(people) == 1, "People count mismatch"
    assert people[0]["name"] == person_data["name"], "Person name mismatch"
    assert people[0]["answers"] == ["", "30"], "Answers mismatch"

    remove_result = await db.remove_event(event_inserted.inserted_id)
    print(f"Removed event and associated people: {remove_result}")

    removed_event = await db.get_event(event_inserted.inserted_id)
    assert removed_event is None, "Event was not removed correctly"

    removed_people = await db.get_people_by_event(event_inserted.inserted_id)
    assert len(removed_people) == 0, "People were not removed correctly"
