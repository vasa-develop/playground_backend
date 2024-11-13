from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from games.routes import router as games_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include game routes
app.include_router(games_router, prefix="/games", tags=["games"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
