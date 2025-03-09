from motor.motor_asyncio import AsyncIOMotorClient

class Database:
    client = None
    db = None

    @classmethod
    async def connect(cls, uri: str):
        uri=uri or "mongodb://localhost:27017"
        cls.client = AsyncIOMotorClient(uri)
        cls.db = cls.client["dog_compliments"]

    @classmethod
    async def close(cls):
        if cls.client:
            cls.client.close()

    @classmethod
    async def get_collection(cls, name: str):
        if not cls.db:
            await cls.connect()
        return cls.db[name] 