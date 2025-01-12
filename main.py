import os
import asyncio

from test import test_database
from fastapi import FastAPI
from api import events, person


# .env MONGODB_URI must be set
if os.getenv("TESTING") in ["1", "True", "true"]:
    asyncio.create_task(test_database())

app = FastAPI()
app.include_router(events.router)
app.include_router(person.router)
