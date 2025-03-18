"""
Script to initialize the database with default agents.
Run this after setting up the database tables.
"""
import uuid
from sqlalchemy.orm import Session

from arivu.db.database import get_db
from arivu.db.models import Agent
from arivu.agents.models import AgentType, LearningLevel

def create_default_agents(db: Session):
    """Create default agents in the database if they don't exist."""
    # Check if we already have agents
    existing_count = db.query(Agent).count()
    if existing_count > 0:
        print(f"Database already has {existing_count} agents. Skipping initialization.")
        return
    
    # Math Agent
    math_agent = Agent(
        id=uuid.uuid4(),
        name="Math Mentor",
        description="mathematics education",
        agent_type=AgentType.MATH.value,
        default_learning_level=LearningLevel.INTERMEDIATE.value,
        enabled=True,
        config={
            "tools": ["calculator", "graph_generator", "equation_solver"]
        }
    )
    
    # Science Agent
    science_agent = Agent(
        id=uuid.uuid4(),
        name="Science Guide",
        description="science education across physics, chemistry, biology, and earth sciences",
        agent_type=AgentType.SCIENCE.value,
        default_learning_level=LearningLevel.HIGH_SCHOOL.value,
        enabled=True,
        config={
            "tools": ["calculator", "graph_generator", "chemical_equation_balancer", "periodic_table"]
        }
    )
    
    # English Language Agent
    english_agent = Agent(
        id=uuid.uuid4(),
        name="English Tutor",
        description="English language education and literacy skills",
        agent_type=AgentType.LANGUAGE.value,
        default_learning_level=LearningLevel.INTERMEDIATE.value,
        enabled=True,
        config={
            "target_language": "english",
            "tools": ["dictionary", "thesaurus", "grammar_checker"]
        }
    )
    
    # French Language Agent
    french_agent = Agent(
        id=uuid.uuid4(),
        name="French Tutor",
        description="French language education and literacy skills",
        agent_type=AgentType.LANGUAGE.value,
        default_learning_level=LearningLevel.BEGINNER.value,
        enabled=True,
        config={
            "target_language": "french",
            "tools": ["dictionary", "thesaurus", "grammar_checker", "translation_tool"]
        }
    )
    
    # History Agent
    history_agent = Agent(
        id=uuid.uuid4(),
        name="History Guide",
        description="history education across different periods and regions",
        agent_type=AgentType.HISTORY.value,
        default_learning_level=LearningLevel.HIGH_SCHOOL.value,
        enabled=True,
        config={
            "tools": ["timeline_generator", "map_visualizer", "primary_source_database"]
        }
    )
    
    # Add all agents to the database
    db.add_all([math_agent, science_agent, english_agent, french_agent, history_agent])
    db.commit()
    
    print("Successfully created default agents:")
    print(f"- {math_agent.name} (ID: {math_agent.id})")
    print(f"- {science_agent.name} (ID: {science_agent.id})")
    print(f"- {english_agent.name} (ID: {english_agent.id})")
    print(f"- {french_agent.name} (ID: {french_agent.id})")
    print(f"- {history_agent.name} (ID: {history_agent.id})")

if __name__ == "__main__":
    # Get a database session
    db_session = next(get_db())
    try:
        create_default_agents(db_session)
    finally:
        db_session.close()