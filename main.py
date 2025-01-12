import os
import asyncio

from test import test_database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import events, person


# .env MONGODB_URI must be set
if os.getenv("TESTING") in ["1", "True", "true"]:
    asyncio.create_task(test_database())

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (including WebSocket)
    allow_headers=["*"],  # Allow all headers
)
app.include_router(events.router)
app.include_router(person.router)
