from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.get("/")
async def root():
    return {"message": "Hello World"}