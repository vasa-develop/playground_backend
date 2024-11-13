from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .environments.breakout_env import BreakoutEnvironment

router = APIRouter()

# Global environment instance
env = None

class GameState(BaseModel):
    state: List[List[float]]
    reward: float
    done: bool
    info: Dict[str, Any]

class ActionRequest(BaseModel):
    action: int

@router.post("/breakout/init", response_model=GameState)
async def init_breakout():
    """Initialize Breakout environment."""
    global env
    try:
        if env is not None:
            env.close()
        env = BreakoutEnvironment()
        state = env.reset()
        return GameState(
            state=state.tolist(),
            reward=0.0,
            done=False,
            info={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/breakout/step", response_model=GameState)
async def step_breakout(action_request: ActionRequest):
    """Execute action in Breakout environment."""
    global env
    try:
        if env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")

        state, reward, done, info = env.step(action_request.action)
        return GameState(
            state=state.tolist(),
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/breakout/valid-actions")
async def get_valid_actions():
    """Get list of valid actions for Breakout."""
    global env
    try:
        if env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        return {"valid_actions": env.get_valid_actions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/breakout/reset", response_model=GameState)
async def reset_breakout():
    """Reset Breakout environment."""
    global env
    try:
        if env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        state = env.reset()
        return GameState(
            state=state.tolist(),
            reward=0.0,
            done=False,
            info={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global env
    if env is not None:
        env.close()
