from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.orm import Session
from uuid import UUID

from arivu.dependencies import get_db, get_current_user, get_llm_service, get_tool_registry
from arivu.db.database import get_agent_instance
from arivu.agents.factory import AgentFactory
from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry
from arivu.users.models import User

from . import models
from . import service

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=models.Agent)
async def create_agent(
    agent_data: models.AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new agent (admin only).
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can create agents")
    
    return service.create_agent(db=db, agent_data=agent_data)

@router.get("/", response_model=List[models.Agent])
async def list_agents(
    enabled: Optional[bool] = Query(None),
    agent_type: Optional[models.AgentType] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all available agents, with optional filtering.
    """
    return service.get_agents(
        db=db, 
        enabled=enabled, 
        agent_type=agent_type
    )

@router.get("/{agent_id}", response_model=models.AgentDetails)
async def get_agent(
    agent_id: UUID = Path(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific agent.
    """
    agent_db = service.get_agent(db=db, agent_id=agent_id)
    if not agent_db:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get agent instance to retrieve available learning levels and tools
    agent_instance = get_agent_instance(
        db=db,
        agent_id=agent_id,
        llm_service=get_llm_service(),
        tool_registry=get_tool_registry()
    )
    
    # Convert learning levels to the expected model format
    learning_levels = [
        models.AgentLearningLevelInfo(
            id=level.id,
            name=level.name,
            description=level.description,
            complexity=level.complexity,
            vocabulary_level=level.vocabulary_level,
            depth_of_explanation=level.depth_of_explanation,
            example_frequency=level.example_frequency,
            assumed_prior_knowledge=level.assumed_prior_knowledge
        )
        for level_id, level in agent_instance.learning_levels.items()
    ]
    
    # Convert agent DB model to API model and add learning levels
    agent = models.Agent.from_orm(agent_db)
    agent_details = models.AgentDetails(
        **agent.dict(),
        available_learning_levels=learning_levels,
        tools=agent_instance._get_subject_tools()
    )
    
    return agent_details

@router.put("/{agent_id}", response_model=models.Agent)
async def update_agent(
    agent_id: UUID,
    agent_data: models.AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing agent (admin only).
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can update agents")
    
    agent = service.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return service.update_agent(db=db, agent=agent, agent_data=agent_data)

@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an agent (admin only).
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can delete agents")
    
    agent = service.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    service.delete_agent(db=db, agent=agent)
    return None

@router.post("/conversations", response_model=models.Conversation)
async def create_conversation(
    conversation_data: models.ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new conversation with an agent.
    """
    # Verify the user is either creating a conversation for themselves
    # or is an admin/teacher creating it for another user
    if conversation_data.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(
            status_code=403, 
            detail="You can only create conversations for yourself unless you're an admin or teacher"
        )
    
    # Check if the agent exists
    agent = service.get_agent(db=db, agent_id=conversation_data.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check if the learning level is valid for this agent
    agent_instance = get_agent_instance(
        db=db,
        agent_id=conversation_data.agent_id,
        llm_service=get_llm_service(),
        tool_registry=get_tool_registry()
    )
    valid_levels = list(agent_instance.learning_levels.keys())
    requested_level = conversation_data.learning_level.value
    
    if requested_level not in valid_levels:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid learning level for this agent. Valid levels are: {', '.join(valid_levels)}"
        )
    
    return service.create_conversation(db=db, conversation_data=conversation_data)

@router.get("/conversations/{conversation_id}", response_model=models.ConversationDetails)
async def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get details of a specific conversation.
    """
    conversation = service.get_conversation(db=db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization - user can access their own conversations,
    # admins and teachers can access any conversation
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    
    return service.get_conversation_details(db=db, conversation=conversation)

@router.get("/conversations", response_model=List[models.Conversation])
async def list_conversations(
    user_id: Optional[UUID] = Query(None),
    agent_id: Optional[UUID] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List conversations, with optional filtering by user or agent.
    """
    # If user_id is specified, check authorization
    if user_id is not None and user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(
            status_code=403, 
            detail="You can only list your own conversations unless you're an admin or teacher"
        )
    
    # If no user_id is specified, default to current user unless admin/teacher
    if user_id is None and not (current_user.is_admin or current_user.is_teacher):
        user_id = current_user.id
    
    return service.list_conversations(
        db=db,
        user_id=user_id,
        agent_id=agent_id,
        limit=limit,
        offset=offset
    )

@router.put("/conversations/{conversation_id}", response_model=models.Conversation)
async def update_conversation(
    conversation_id: UUID,
    conversation_data: models.ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a conversation (title, learning level, metadata).
    """
    conversation = service.get_conversation(db=db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to update this conversation")
    
    # If learning level is being updated, verify it's valid for the agent
    if conversation_data.learning_level:
        agent_instance = get_agent_instance(
            db=db,
            agent_id=conversation.agent_id,
            llm_service=get_llm_service(),
            tool_registry=get_tool_registry()
        )
        valid_levels = list(agent_instance.learning_levels.keys())
        requested_level = conversation_data.learning_level.value
        
        if requested_level not in valid_levels:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid learning level for this agent. Valid levels are: {', '.join(valid_levels)}"
            )
    
    return service.update_conversation(
        db=db,
        conversation=conversation,
        conversation_data=conversation_data
    )

@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a conversation.
    """
    conversation = service.get_conversation(db=db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to delete this conversation")
    
    service.delete_conversation(db=db, conversation=conversation)
    return None

@router.post("/message", response_model=models.AgentMessageResponse)
async def send_message(
    message_request: models.UserMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Send a message to an agent and get a response.
    """
    conversation = service.get_conversation(db=db, conversation_id=message_request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to message in this conversation")
    
    # Add user message to the conversation
    user_message = service.add_message(
        db=db,
        conversation_id=conversation.id,
        content=message_request.content,
        role=models.MessageRole.USER,
        metadata=message_request.metadata
    )
    
    # Get agent instance
    agent_instance = get_agent_instance(
        db=db,
        agent_id=conversation.agent_id,
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Set the learning level for this conversation
    agent_instance.set_learning_level(conversation.learning_level.value)
    
    # Load conversation history into agent's memory
    history = service.get_conversation_messages(
        db=db,
        conversation_id=conversation.id,
        limit=20  # Limit to recent messages for context window efficiency
    )
    
    for msg in history:
        if msg.id != user_message.id:  # Skip the message we just added
            agent_instance.memory.add_interaction(
                user_message=msg.content if msg.role == models.MessageRole.USER else None,
                assistant_message=msg.content if msg.role == models.MessageRole.AGENT else None
            )
    
    # Generate agent response
    response = agent_instance.generate_response(message_request.content)
    
    # Save agent response to database
    agent_message = service.add_message(
        db=db,
        conversation_id=conversation.id,
        content=response.message,
        role=models.MessageRole.AGENT,
        metadata={
            "tools_used": response.tools_used,
            "sources": response.sources,
            "confidence": response.confidence,
            "learning_level": agent_instance.current_level.id
        }
    )
    
    # In the background, update the conversation metadata with latest learning metrics
    background_tasks.add_task(
        service.update_conversation_learning_metrics,
        db=db,
        conversation_id=conversation.id,
        agent_instance=agent_instance
    )
    
    # Generate suggested follow-up questions (could be enhanced with LLM call)
    suggested_questions = service.generate_suggested_questions(
        agent_instance=agent_instance,
        conversation_history=history + [user_message],
        agent_response=response
    )
    
    return models.AgentMessageResponse(
        message=agent_message,
        sources=response.sources,
        tools_used=response.tools_used,
        confidence=response.confidence,
        metadata=response.metadata,
        suggested_questions=suggested_questions
    )

@router.get("/conversations/{conversation_id}/messages", response_model=List[models.Message])
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get messages for a specific conversation.
    """
    conversation = service.get_conversation(db=db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to access messages in this conversation")
    
    return service.get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        limit=limit,
        offset=offset
    )

@router.get("/conversations/{conversation_id}/progress", response_model=models.LearningProgressResponse)
async def get_learning_progress(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Get learning progress metrics for a specific conversation.
    """
    conversation = service.get_conversation(db=db, conversation_id=conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check authorization
    if conversation.user_id != current_user.id and not (current_user.is_admin or current_user.is_teacher):
        raise HTTPException(status_code=403, detail="Not authorized to access progress for this conversation")
    
    # Get agent instance
    agent_instance = get_agent_instance(
        db=db,
        agent_id=conversation.agent_id,
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Load conversation history into agent's memory to ensure metrics are up-to-date
    history = service.get_conversation_messages(
        db=db,
        conversation_id=conversation.id,
        limit=50  # Use more messages for accurate analysis
    )
    
    # Process conversation history to update agent's metrics
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            user_msg = history[i]
            agent_msg = history[i + 1]
            if user_msg.role == models.MessageRole.USER and agent_msg.role == models.MessageRole.AGENT:
                # Update agent's memory and metrics
                agent_instance.memory.add_interaction(user_msg.content, agent_msg.content)
                
                # For math agent, update topic proficiency
                if conversation.agent.agent_type == models.AgentType.MATH:
                    agent_instance.update_topic_proficiency(user_msg.content, {"message": agent_msg.content})
                
                # For science agent, update topic proficiency
                elif conversation.agent.agent_type == models.AgentType.SCIENCE:
                    agent_instance.update_topic_proficiency(user_msg.content, {"message": agent_msg.content})
                
                # For language agent, update language proficiency
                elif conversation.agent.agent_type == models.AgentType.LANGUAGE:
                    agent_instance.update_language_proficiency(user_msg.content, {"message": agent_msg.content})
                
                # For history agent, update historical knowledge
                elif conversation.agent.agent_type == models.AgentType.HISTORY:
                    agent_instance.update_historical_knowledge(user_msg.content, {"message": agent_msg.content})
    
    # Get learning progress analysis
    progress = agent_instance.analyze_strengths_weaknesses()
    
    # Add progress over time data (could be enhanced with actual time-series data)
    progress_over_time = {
        "learning_pace": [
            {"timestamp": msg.timestamp.isoformat(), "value": 5}  # Placeholder data
            for msg in history if msg.role == models.MessageRole.AGENT
        ]
    }
    
    return models.LearningProgressResponse(
        strengths=progress["strengths"],
        weaknesses=progress["weaknesses"],
        learning_pace=progress["learning_pace"],
        recommended_focus_areas=progress["recommended_focus_areas"],
        progress_over_time=progress_over_time
    )

# SUBJECT-SPECIFIC ENDPOINTS

# Math Agent Endpoints
@router.post("/math/problems", response_model=List[models.MathProblem])
async def generate_math_problems(
    problem_request: models.MathProblemRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Generate math practice problems for a specific topic.
    """
    # Create a temporary math agent
    math_agent = AgentFactory.create_agent(
        agent_type="math",
        agent_id="temp-math-agent",
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Generate problems
    problems = math_agent.generate_practice_problems(
        topic=problem_request.topic,
        difficulty=problem_request.difficulty
    )
    
    # Convert to API model
    result = []
    for problem in problems[:problem_request.count]:
        result.append(models.MathProblem(
            instruction=f"Solve the following {problem_request.topic} problem:",
            content=problem["problem"],
            solution=problem["solution"] if problem_request.include_solutions else None,
            hints=problem.get("hints", []),
            difficulty=problem.get("difficulty", 5),
            problem_type=problem_request.topic,
            metadata={"topic": problem_request.topic}
        ))
    
    return result

# Science Agent Endpoints
@router.post("/science/experiments", response_model=models.ScienceExperiment)
async def generate_science_experiment(
    experiment_request: models.ScienceExperimentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Generate a science experiment for a specific topic.
    """
    # Create a temporary science agent
    science_agent = AgentFactory.create_agent(
        agent_type="science",
        agent_id="temp-science-agent",
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Generate experiment
    experiment = science_agent.generate_experiment(
        topic=experiment_request.topic,
        difficulty=experiment_request.difficulty
    )
    
    # Safety check based on requested safety level
    safety_level = experiment_request.safety_level
    if safety_level < 5 and "safety_notes" in experiment:
        # Add additional safety warnings for lower safety level requests
        experiment["safety_notes"] += f"\n\nNOTE: This experiment is recommended for safety level {safety_level}/5. " \
                                      f"Adult supervision is required."
    
    # Convert to API model
    return models.ScienceExperiment(
        title=experiment.get("title", f"Experiment about {experiment_request.topic}"),
        objective=experiment.get("objective", ""),
        materials=experiment.get("materials", []),
        procedure=experiment.get("procedure", []),
        expected_results=experiment.get("expected_results", ""),
        explanation=experiment.get("explanation", ""),
        safety_notes=experiment.get("safety_notes", ""),
        difficulty=experiment.get("difficulty", 5),
        metadata={"topic": experiment_request.topic}
    )

# Language Agent Endpoints
@router.post("/language/exercises", response_model=List[models.ExerciseBase])
async def generate_language_exercises(
    exercise_request: models.LanguageExerciseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Generate language exercises for a specific skill.
    """
    # Create a temporary language agent
    language_agent = AgentFactory.create_agent(
        agent_type="language",
        agent_id="temp-language-agent",
        llm_service=llm_service,
        tool_registry=tool_registry,
        config={"target_language": "english"}  # Default to English
    )
    
    # Generate exercises
    exercises = language_agent.generate_exercises(
        skill=exercise_request.skill,
        difficulty=exercise_request.difficulty
    )
    
    # Convert to API model
    result = []
    for exercise in exercises[:exercise_request.count]:
        result.append(models.ExerciseBase(
            instruction=exercise.get("instruction", ""),
            content=exercise.get("exercise", ""),
            solution=exercise.get("solution", ""),
            hints=exercise.get("hints", []),
            difficulty=exercise.get("difficulty", 5),
            metadata={"skill": exercise_request.skill}
        ))
    
    return result

@router.post("/language/writing-prompts", response_model=models.WritingPrompt)
async def generate_writing_prompt(
    prompt_request: models.WritingPromptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Generate a writing prompt, optionally for a specific topic.
    """
    # Create a temporary language agent
    language_agent = AgentFactory.create_agent(
        agent_type="language",
        agent_id="temp-language-agent",
        llm_service=llm_service,
        tool_registry=tool_registry,
        config={"target_language": "english"}  # Default to English
    )
    
    # Generate prompt
    prompt = language_agent.generate_writing_prompt(
        topic=prompt_request.topic,
        difficulty=prompt_request.difficulty
    )
    
    # Adjust word count if requested
    word_count = prompt.get("word_count_recommendation", 300)
    if prompt_request.word_count:
        word_count = prompt_request.word_count
    
    # Convert to API model
    return models.WritingPrompt(
        prompt=prompt.get("prompt", ""),
        guidelines=prompt.get("guidelines", []),
        word_count_recommendation=word_count,
        difficulty=prompt.get("difficulty", 5),
        metadata={"topic": prompt_request.topic}
    )

# History Agent Endpoints
@router.post("/history/timelines", response_model=models.HistoricalTimeline)
async def generate_historical_timeline(
    timeline_request: models.HistoricalTimelineRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Generate a historical timeline for a specific era and/or region.
    """
    # Create a temporary history agent
    history_agent = AgentFactory.create_agent(
        agent_type="history",
        agent_id="temp-history-agent",
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Generate timeline
    timeline_events = history_agent.generate_timeline(
        era=timeline_request.era,
        region=timeline_request.region
    )
    
    # Limit the number of events if requested
    if timeline_request.limit and len(timeline_events) > timeline_request.limit:
        timeline_events = timeline_events[:timeline_request.limit]
    
    # Convert to API model
    events = []
    for event in timeline_events:
        events.append(models.TimelineEvent(
            year=event.get("year", ""),
            event=event.get("event", ""),
            significance=event.get("significance", ""),
            connections=event.get("connections", []),
            metadata={}
        ))
    
    return models.HistoricalTimeline(
        era=timeline_request.era,
        region=timeline_request.region,
        focus=timeline_request.focus,
        events=events,
        context=f"Timeline of {timeline_request.focus or 'historical events'} "
                f"for {timeline_request.era or 'all eras'} "
                f"in {timeline_request.region or 'all regions'}",
        metadata={}
    )

@router.post("/history/primary-sources", response_model=List[models.PrimarySource])
async def suggest_primary_sources(
    source_request: models.PrimarySourceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Suggest historical primary sources for a specific topic.
    """
    # Create a temporary history agent
    history_agent = AgentFactory.create_agent(
        agent_type="history",
        agent_id="temp-history-agent",
        llm_service=llm_service,
        tool_registry=tool_registry
    )
    
    # Get source suggestions
    sources = history_agent.suggest_primary_sources(
        topic=source_request.topic,
        difficulty=source_request.difficulty
    )
    
    # Convert to API model
    result = []
    for source in sources[:source_request.count]:
        result.append(models.PrimarySource(
            title=source.get("title", ""),
            type=source.get("type", "Document"),
            time_period=source.get("time_period", ""),
            description=source.get("description", ""),
            analysis_questions=source.get("analysis_questions", []),
            difficulty=source.get("difficulty", 5),
            url=source.get("url", None),
            metadata={"topic": source_request.topic}
        ))
    
    return result