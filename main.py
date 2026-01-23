"""
Hamo-UME: Hamo Unified Mind Engine
Backend API Server for Hamo Pro and Hamo Client

Tech Stack: Python + FastAPI
Version: 0.1.0 (Prototype)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import random
import uuid

# ============================================================
# ENUMS
# ============================================================

class EmotionType(str, Enum):
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    JOY = "joy"
    NEUTRAL = "neutral"

class PersonalityTrait(str, Enum):
    INTROVERT = "introvert"
    EXTROVERT = "extrovert"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PERFECTIONIST = "perfectionist"
    PEOPLE_PLEASER = "people_pleaser"
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"

class RelationshipStyle(str, Enum):
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"

# ============================================================
# DATA MODELS - User AI Mind
# ============================================================

class PersonalityCharacteristics(BaseModel):
    """User's personality profile"""
    primary_traits: list[PersonalityTrait] = Field(default_factory=list)
    openness: float = Field(ge=0, le=1, description="0-1 scale")
    conscientiousness: float = Field(ge=0, le=1)
    extraversion: float = Field(ge=0, le=1)
    agreeableness: float = Field(ge=0, le=1)
    neuroticism: float = Field(ge=0, le=1)
    description: str = ""

class EmotionPattern(BaseModel):
    """User's emotional patterns and triggers"""
    dominant_emotions: list[EmotionType] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    coping_mechanisms: list[str] = Field(default_factory=list)
    emotional_stability: float = Field(ge=0, le=1)
    description: str = ""

class CognitionBeliefs(BaseModel):
    """User's cognitive patterns and core beliefs"""
    core_beliefs: list[str] = Field(default_factory=list)
    cognitive_distortions: list[str] = Field(default_factory=list)
    thinking_patterns: list[str] = Field(default_factory=list)
    self_perception: str = ""
    world_perception: str = ""
    future_perception: str = ""

class RelationshipManipulations(BaseModel):
    """User's relationship patterns and attachment style"""
    attachment_style: RelationshipStyle = RelationshipStyle.SECURE
    relationship_patterns: list[str] = Field(default_factory=list)
    communication_style: str = ""
    conflict_resolution: str = ""
    trust_level: float = Field(ge=0, le=1)
    intimacy_comfort: float = Field(ge=0, le=1)

class UserAIMind(BaseModel):
    """Complete User AI Mind Profile"""
    user_id: str
    avatar_id: str
    personality: PersonalityCharacteristics
    emotion_pattern: EmotionPattern
    cognition_beliefs: CognitionBeliefs
    relationship_manipulations: RelationshipManipulations
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0, le=1, description="Model confidence")

# ============================================================
# DATA MODELS - API Requests/Responses
# ============================================================

class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    sender: str  # "client" or "avatar"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class TrainingRequest(BaseModel):
    """Request to train AI Mind with conversation data"""
    user_id: str
    avatar_id: str
    session_id: str
    conversation: list[ConversationMessage]
    session_notes: Optional[str] = None

class TrainingResponse(BaseModel):
    """Response after training request"""
    success: bool
    message: str
    training_id: str
    estimated_completion: datetime

class SessionFeedback(BaseModel):
    """Client's session feedback - Being, Feeling, Knowing"""
    user_id: str
    session_id: str
    # Being - physical/somatic state
    being_energy_level: float = Field(ge=0, le=10)
    being_physical_comfort: float = Field(ge=0, le=10)
    being_description: Optional[str] = None
    # Feeling - emotional state
    feeling_primary_emotion: EmotionType
    feeling_intensity: float = Field(ge=0, le=10)
    feeling_description: Optional[str] = None
    # Knowing - cognitive/insight state
    knowing_clarity: float = Field(ge=0, le=10)
    knowing_insights: list[str] = Field(default_factory=list)
    knowing_description: Optional[str] = None
    # Overall
    overall_rating: float = Field(ge=0, le=10)
    timestamp: datetime = Field(default_factory=datetime.now)

class FeedbackResponse(BaseModel):
    """Response after feedback submission"""
    success: bool
    message: str
    feedback_id: str

# ============================================================
# MOCK DATA GENERATOR
# ============================================================

class MockDataGenerator:
    """Generates realistic mock data for prototyping"""
    
    @staticmethod
    def generate_user_ai_mind(user_id: str, avatar_id: str) -> UserAIMind:
        """Generate a simulated User AI Mind profile"""
        
        # Simulated personality
        personality = PersonalityCharacteristics(
            primary_traits=random.sample(list(PersonalityTrait), k=random.randint(2, 4)),
            openness=round(random.uniform(0.3, 0.9), 2),
            conscientiousness=round(random.uniform(0.3, 0.9), 2),
            extraversion=round(random.uniform(0.2, 0.8), 2),
            agreeableness=round(random.uniform(0.4, 0.9), 2),
            neuroticism=round(random.uniform(0.3, 0.7), 2),
            description="Client shows introverted tendencies with high conscientiousness. "
                       "Tends toward perfectionism in work-related tasks."
        )
        
        # Simulated emotion patterns
        emotion_pattern = EmotionPattern(
            dominant_emotions=random.sample([EmotionType.ANXIETY, EmotionType.SADNESS, 
                                            EmotionType.NEUTRAL], k=2),
            triggers=[
                "Work deadlines and performance pressure",
                "Social situations with unfamiliar people",
                "Conflict or perceived criticism"
            ],
            coping_mechanisms=[
                "Withdrawal and isolation",
                "Over-preparation and planning",
                "Seeking reassurance from trusted others"
            ],
            emotional_stability=round(random.uniform(0.4, 0.7), 2),
            description="Experiences heightened anxiety in performance situations. "
                       "Emotional responses tend toward internalization."
        )
        
        # Simulated cognition beliefs
        cognition_beliefs = CognitionBeliefs(
            core_beliefs=[
                "I must perform perfectly to be accepted",
                "Making mistakes means I am a failure",
                "Others' needs are more important than mine"
            ],
            cognitive_distortions=[
                "All-or-nothing thinking",
                "Catastrophizing",
                "Mind reading"
            ],
            thinking_patterns=[
                "Rumination on past events",
                "Anticipatory worry about future scenarios",
                "Self-critical internal dialogue"
            ],
            self_perception="Views self as capable but fundamentally flawed",
            world_perception="World is demanding and judgmental",
            future_perception="Future success depends on perfect performance"
        )
        
        # Simulated relationship patterns
        relationship_manipulations = RelationshipManipulations(
            attachment_style=random.choice([RelationshipStyle.ANXIOUS, 
                                           RelationshipStyle.AVOIDANT]),
            relationship_patterns=[
                "Difficulty expressing needs directly",
                "Fear of abandonment in close relationships",
                "Tendency to over-accommodate others"
            ],
            communication_style="Indirect, tends to hint rather than state needs explicitly",
            conflict_resolution="Avoidant - prefers to minimize or ignore conflicts",
            trust_level=round(random.uniform(0.4, 0.7), 2),
            intimacy_comfort=round(random.uniform(0.3, 0.6), 2)
        )
        
        return UserAIMind(
            user_id=user_id,
            avatar_id=avatar_id,
            personality=personality,
            emotion_pattern=emotion_pattern,
            cognition_beliefs=cognition_beliefs,
            relationship_manipulations=relationship_manipulations,
            last_updated=datetime.now(),
            confidence_score=round(random.uniform(0.7, 0.95), 2)
        )

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hamo-UME API",
    description="Hamo Unified Mind Engine - Backend API for AI Therapy Platform",
    version="0.1.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for prototype
mind_cache: dict[str, UserAIMind] = {}
feedback_storage: list[SessionFeedback] = []
training_queue: list[TrainingRequest] = []

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "service": "Hamo-UME",
        "version": "0.1.0",
        "status": "running",
        "description": "Hamo Unified Mind Engine API"
    }

@app.get("/api/v1/mind/{user_id}/{avatar_id}", response_model=UserAIMind)
async def get_user_ai_mind(user_id: str, avatar_id: str):
    """
    getUserAIMind() - Get User's AI Mind Profile
    
    Returns the up-to-date AI Mind for a specific User + Pro Avatar pair,
    including Personality, Emotion Pattern, Cognition Beliefs, 
    and Relationship Manipulations.
    
    Usage: Combine with Pro Avatar profile and conversation history
    to create context prompt for LLM therapy responses.
    """
    cache_key = f"{user_id}_{avatar_id}"
    
    # Check cache first
    if cache_key in mind_cache:
        return mind_cache[cache_key]
    
    # Generate new mock data for prototype
    user_mind = MockDataGenerator.generate_user_ai_mind(user_id, avatar_id)
    mind_cache[cache_key] = user_mind
    
    return user_mind

@app.post("/api/v1/mind/train", response_model=TrainingResponse)
async def submit_training_request(request: TrainingRequest):
    """
    submitTrainingRequest() - Submit conversation for AI Mind training
    
    Accepts conversation data between Pro Avatar and Client User
    to update and refine the User's AI Mind model.
    """
    if not request.conversation:
        raise HTTPException(status_code=400, detail="Conversation cannot be empty")
    
    # Store in training queue (prototype simulation)
    training_queue.append(request)
    training_id = str(uuid.uuid4())
    
    # Simulate training process
    return TrainingResponse(
        success=True,
        message=f"Training request queued successfully. "
                f"Processing {len(request.conversation)} messages.",
        training_id=training_id,
        estimated_completion=datetime.now()
    )

@app.post("/api/v1/feedback/session", response_model=FeedbackResponse)
async def submit_session_feedback(feedback: SessionFeedback):
    """
    submitSessionFeedback() - Submit client's session feedback
    
    Accepts feedback on Being (physical), Feeling (emotional), 
    and Knowing (cognitive/insight) states after a session.
    """
    # Store feedback (prototype simulation)
    feedback_storage.append(feedback)
    feedback_id = str(uuid.uuid4())
    
    return FeedbackResponse(
        success=True,
        message="Session feedback recorded successfully",
        feedback_id=feedback_id
    )

@app.get("/api/v1/feedback/{user_id}", response_model=list[SessionFeedback])
async def get_user_feedback_history(user_id: str):
    """Get all feedback history for a specific user"""
    user_feedback = [f for f in feedback_storage if f.user_id == user_id]
    return user_feedback

@app.get("/api/v1/training/status/{training_id}")
async def get_training_status(training_id: str):
    """Check status of a training request"""
    # Prototype: Always return completed
    return {
        "training_id": training_id,
        "status": "completed",
        "progress": 100,
        "message": "AI Mind model updated successfully"
    }

# ============================================================
# RUN SERVER
# ============================================================
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
