from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .environments.breakout_env import BreakoutEnvironment
import uuid

router = APIRouter()

# Dictionary to store environment instances
environments = {}

class GameState(BaseModel):
    state: List[List[List[float]]]  # Shape: (4, 84, 84) for frame stack
    reward: float
    done: bool
    info: Dict[str, Any]
    session_id: str
    suggested_action: Optional[int] = None

class ActionRequest(BaseModel):
    action: int
    use_ai: bool = False
    session_id: str

@router.post("/breakout/init", response_model=GameState)
async def init_breakout():
    """Initialize Breakout environment."""
    try:
        session_id = str(uuid.uuid4())
        env = BreakoutEnvironment()
        environments[session_id] = env
        state, info = env.reset()  # Now returns tuple of (state, info)

        # Get AI suggestion if available
        suggested_action = env.get_ai_suggestion(state) if hasattr(env, 'get_ai_suggestion') else None

        return GameState(
            state=state,  # State is already a list
            reward=0.0,
            done=False,
            info=info,  # Use the info from reset
            session_id=session_id,
            suggested_action=suggested_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/breakout/step", response_model=GameState)
async def step_breakout(action_request: ActionRequest):
    """Execute action in Breakout environment."""
    try:
        env = environments.get(action_request.session_id)
        if env is None:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Use AI suggestion if requested
        action = env.get_ai_suggestion(env.get_state()) if action_request.use_ai and hasattr(env, 'get_ai_suggestion') else action_request.action

        state, reward, done, info = env.step(action)

        # Get next AI suggestion
        suggested_action = env.get_ai_suggestion(state) if hasattr(env, 'get_ai_suggestion') else None

        return GameState(
            state=state,  # State is already a list
            reward=float(reward),
            done=bool(done),
            info=info,
            session_id=action_request.session_id,
            suggested_action=suggested_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/breakout/valid-actions")
async def get_valid_actions():
    """Get list of valid actions for Breakout."""
    try:
        # Create temporary env just to get valid actions
        temp_env = BreakoutEnvironment()
        valid_actions = temp_env.get_valid_actions()
        temp_env.close()
        return {"valid_actions": valid_actions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/breakout/reset", response_model=GameState)
async def reset_breakout(session_id: str):
    """Reset Breakout environment."""
    try:
        env = environments.get(session_id)
        if env is None:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        state, info = env.reset()  # Now returns tuple of (state, info)

        # Get AI suggestion if available
        suggested_action = env.get_ai_suggestion(state) if hasattr(env, 'get_ai_suggestion') else None

        return GameState(
            state=state,  # State is already a list
            reward=0.0,
            done=False,
            info=info,  # Use the info from reset
            session_id=session_id,
            suggested_action=suggested_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    for env in environments.values():
        env.close()
    environments.clear()
