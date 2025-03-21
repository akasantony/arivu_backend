from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

class AgentType(str, Enum):
    """Available agent types in the system."""
    MATH = "math"
    SCIENCE = "science"
    LANGUAGE = "language"
    HISTORY = "history"

class LearningLevel(str, Enum):
    """Standard learning levels available across agents."""
    ELEMENTARY = "elementary"
    BEGINNER = "beginner"
    MIDDLE_SCHOOL = "middle_school"
    INTERMEDIATE = "intermediate"
    HIGH_SCHOOL = "high_school"
    ADVANCED = "advanced"
    UNDERGRADUATE = "undergraduate"
    FLUENT = "fluent"
    GRADUATE = "graduate"
    EXPERT = "expert"

class AgentBase(BaseModel):
    """Base model for agent data."""
    name: str
    description: str
    agent_type: AgentType
    default_learning_level: LearningLevel = Field(default=LearningLevel.INTERMEDIATE)
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None

class AgentCreate(AgentBase):
    """Model for creating a new agent."""
    pass

class AgentUpdate(BaseModel):
    """Model for updating an existing agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    default_learning_level: Optional[LearningLevel] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class Agent(AgentBase):
    """Full agent model with system fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True

class AgentLearningLevelInfo(BaseModel):
    """Information about a specific learning level for an agent."""
    id: str
    name: str
    description: str
    complexity: int = Field(ge=1, le=10)
    vocabulary_level: int = Field(ge=1, le=10)
    depth_of_explanation: int = Field(ge=1, le=10)
    example_frequency: int = Field(ge=1, le=10)
    assumed_prior_knowledge: int = Field(ge=1, le=10)

class AgentDetails(Agent):
    """Detailed agent model including available learning levels."""
    available_learning_levels: List[AgentLearningLevelInfo]
    tools: List[str]

class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"

class MessageBase(BaseModel):
    """Base model for conversation messages."""
    content: str
    role: MessageRole
    metadata: Optional[Dict[str, Any]] = None

class Message(MessageBase):
    """Full message model with system fields."""
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True

class ConversationBase(BaseModel):
    """Base model for conversations."""
    title: str
    user_id: UUID
    agent_id: UUID
    learning_level: LearningLevel = Field(default=LearningLevel.INTERMEDIATE)
    metadata: Optional[Dict[str, Any]] = None

class ConversationCreate(ConversationBase):
    """Model for creating a new conversation."""
    pass

class ConversationUpdate(BaseModel):
    """Model for updating an existing conversation."""
    title: Optional[str] = None
    learning_level: Optional[LearningLevel] = None
    metadata: Optional[Dict[str, Any]] = None

class Conversation(ConversationBase):
    """Full conversation model with system fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = []
    
    class Config:
        orm_mode = True

class ConversationDetails(Conversation):
    """Detailed conversation model including agent details."""
    agent: Agent

class UserMessageRequest(BaseModel):
    """Model for a user sending a message in a conversation."""
    content: str
    conversation_id: UUID
    metadata: Optional[Dict[str, Any]] = None

class AgentMessageResponse(BaseModel):
    """Model for agent's response to a user message."""
    message: Message
    sources: Optional[List[Dict[str, Any]]] = None
    tools_used: Optional[List[str]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    suggested_questions: Optional[List[str]] = None

class ConversationHistoryRequest(BaseModel):
    """Model for requesting conversation history."""
    conversation_id: UUID
    limit: Optional[int] = 50
    offset: Optional[int] = 0

class LearningProgressResponse(BaseModel):
    """Model for student's learning progress with an agent."""
    strengths: Dict[str, float]
    weaknesses: Dict[str, float]
    learning_pace: int = Field(ge=1, le=10)
    recommended_focus_areas: List[str]
    progress_over_time: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
class ExerciseBase(BaseModel):
    """Base model for exercises generated by agents."""
    instruction: str
    content: str
    solution: Optional[str] = None
    hints: Optional[List[str]] = None
    difficulty: int = Field(ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None

class MathProblemRequest(BaseModel):
    """Model for requesting math practice problems."""
    topic: str
    difficulty: Optional[int] = None
    count: Optional[int] = 3
    include_solutions: bool = True

class MathProblem(ExerciseBase):
    """Model for math practice problems."""
    problem_type: str
    

class ScienceExperimentRequest(BaseModel):
    """Model for requesting science experiments."""
    topic: str
    difficulty: Optional[int] = None
    safety_level: Optional[int] = Field(ge=1, le=5, default=3)

class ScienceExperiment(BaseModel):
    """Model for science experiments."""
    title: str
    objective: str
    materials: List[str]
    procedure: List[str]
    expected_results: str
    explanation: str
    safety_notes: str
    difficulty: int = Field(ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None

class LanguageExerciseRequest(BaseModel):
    """Model for requesting language exercises."""
    skill: str
    difficulty: Optional[int] = None
    count: Optional[int] = 3

class WritingPromptRequest(BaseModel):
    """Model for requesting writing prompts."""
    topic: Optional[str] = None
    difficulty: Optional[int] = None
    word_count: Optional[int] = None

class WritingPrompt(BaseModel):
    """Model for writing prompts."""
    prompt: str
    guidelines: List[str]
    word_count_recommendation: int
    difficulty: int = Field(ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None

class HistoricalTimelineRequest(BaseModel):
    """Model for requesting historical timelines."""
    era: Optional[str] = None
    region: Optional[str] = None
    focus: Optional[str] = None
    limit: Optional[int] = 10

class TimelineEvent(BaseModel):
    """Model for historical timeline events."""
    year: str
    event: str
    significance: str
    connections: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class HistoricalTimeline(BaseModel):
    """Model for historical timelines."""
    era: Optional[str] = None
    region: Optional[str] = None
    focus: Optional[str] = None
    events: List[TimelineEvent]
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PrimarySourceRequest(BaseModel):
    """Model for requesting historical primary sources."""
    topic: str
    difficulty: Optional[int] = None
    count: Optional[int] = 3

class PrimarySource(BaseModel):
    """Model for historical primary sources."""
    title: str
    type: str
    time_period: str
    description: str
    analysis_questions: List[str]
    difficulty: int = Field(ge=1, le=10)
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None