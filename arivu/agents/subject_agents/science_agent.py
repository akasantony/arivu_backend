from typing import List, Dict, Any
from arivu.agents.base_agent import BaseAgent
from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class ScienceAgent(BaseAgent):
    """
    Agent specializing in science education across multiple disciplines.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_service: LLMService,
        memory: Memory,
        tool_registry: ToolRegistry,
    ):
        # Define science-specific learning levels
        from arivu.agents.base_agent import LearningLevel
        
        science_learning_levels = [
            LearningLevel(
                id="elementary",
                name="Elementary",
                description="Basic science concepts for elementary education",
                complexity=2,
                vocabulary_level=2,
                depth_of_explanation=3,
                example_frequency=8,
                assumed_prior_knowledge=1
            ),
            LearningLevel(
                id="middle_school",
                name="Middle School",
                description="Science concepts for middle school education",
                complexity=4,
                vocabulary_level=4,
                depth_of_explanation=5,
                example_frequency=7,
                assumed_prior_knowledge=3
            ),
            LearningLevel(
                id="high_school",
                name="High School",
                description="Science concepts for high school education",
                complexity=6,
                vocabulary_level=6,
                depth_of_explanation=7,
                example_frequency=6,
                assumed_prior_knowledge=5
            ),
            LearningLevel(
                id="undergraduate",
                name="Undergraduate",
                description="Science concepts for undergraduate education",
                complexity=8,
                vocabulary_level=8,
                depth_of_explanation=8,
                example_frequency=5,
                assumed_prior_knowledge=7
            ),
            LearningLevel(
                id="graduate",
                name="Graduate",
                description="Advanced science concepts for graduate education",
                complexity=10,
                vocabulary_level=10,
                depth_of_explanation=10,
                example_frequency=4,
                assumed_prior_knowledge=9
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name="Science Guide",
            description="science education across physics, chemistry, biology, and earth sciences",
            llm_service=llm_service,
            memory=memory,
            tool_registry=tool_registry,
            learning_levels=science_learning_levels,
            default_level="high_school"
        )
        
        # Science-specific discipline tracking
        self.science_disciplines = {
            "physics": {"score": 0, "interactions": 0},
            "chemistry": {"score": 0, "interactions": 0},
            "biology": {"score": 0, "interactions": 0},
            "earth_science": {"score": 0, "interactions": 0},
            "astronomy": {"score": 0, "interactions": 0},
            "environmental_science": {"score": 0, "interactions": 0}
        }
        
        # Topic tracking within disciplines
        self.discipline_topics = {
            "physics": ["mechanics", "electricity", "magnetism", "thermodynamics", "quantum", "relativity"],
            "chemistry": ["atomic structure", "chemical bonds", "reactions", "organic", "biochemistry", "thermodynamics"],
            "biology": ["cells", "genetics", "evolution", "ecology", "physiology", "microbiology"],
            "earth_science": ["geology", "meteorology", "oceanography", "climate", "natural resources"],
            "astronomy": ["solar system", "stars", "galaxies", "cosmology", "space exploration"],
            "environmental_science": ["ecosystems", "pollution", "conservation", "sustainability", "climate change"]
        }
        
        # Initialize topic tracking
        self.topic_proficiency = {}
        for discipline, topics in self.discipline_topics.items():
            for topic in topics:
                self.topic_proficiency[topic] = {"score": 0, "interactions": 0}
    
    def _get_subject_specific_instructions(self) -> str:
        """Get science-specific instructions for the agent."""
        return """
        As a science education specialist:
        
        1. Emphasize the scientific method and evidence-based reasoning.
        2. Connect scientific principles to observable phenomena and real-world applications.
        3. Use analogies and models to explain complex concepts.
        4. Acknowledge the tentative nature of scientific knowledge.
        5. Incorporate historical context of scientific discoveries when relevant.
        6. Explain both the "what" and the "why" of scientific phenomena.
        7. Encourage curiosity and questioning.
        8. Address common misconceptions in science education.
        
        Major disciplines include:
        - Physics: mechanics, electricity and magnetism, thermodynamics, waves, modern physics
        - Chemistry: atomic structure, bonding, reactions, organic chemistry, biochemistry
        - Biology: cells, genetics, evolution, ecology, physiology, biodiversity
        - Earth Science: geology, meteorology, oceanography, climate science
        - Astronomy: solar system, stars, galaxies, cosmology
        - Environmental Science: ecosystems, pollution, conservation, sustainability
        
        For elementary level: Focus on observable phenomena, basic classifications, simple 
        cause-effect relationships, and developing scientific curiosity.
        
        For middle school level: Introduce more structured scientific concepts, basic models,
        and simple mathematical relationships in science.
        
        For high school level: Develop deeper understanding of scientific principles, 
        introduce more advanced models, and strengthen connections between disciplines.
        
        For undergraduate level: Focus on detailed mechanisms, theoretical frameworks,
        research methodologies, and interdisciplinary connections.
        
        For graduate level: Emphasize current research, complex systems, specialized 
        techniques, and the frontiers of scientific knowledge.
        """
    
    def _get_subject_tools(self) -> List[str]:
        """Get a list of tool IDs relevant to science education."""
        return [
            "calculator",
            "graph_generator",
            "chemical_equation_balancer",
            "periodic_table",
            "molecule_visualizer",
            "physics_simulator",
            "web_search"
        ]
    
    def categorize_science_topic(self, user_input: str) -> Dict[str, Dict[str, float]]:
        """
        Categorize the user input into relevant science disciplines and topics.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: Nested mapping of discipline -> topic -> relevance score (0-1)
        """
        # This is a simplified version. In a real implementation, 
        # this might use a classifier model or more sophisticated logic
        
        discipline_keywords = {
            "physics": ["force", "motion", "energy", "wave", "light", "electricity", "magnetism", 
                        "quantum", "relativity", "mechanics", "thermodynamics"],
            "chemistry": ["element", "compound", "reaction", "molecule", "acid", "base", "atom", 
                         "bond", "organic", "inorganic", "solution", "concentration"],
            "biology": ["cell", "gene", "protein", "evolution", "species", "organism", "ecosystem", 
                       "anatomy", "physiology", "dna", "heredity", "photosynthesis"],
            "earth_science": ["rock", "mineral", "plate", "earthquake", "volcano", "weather", "climate", 
                             "ocean", "atmosphere", "erosion", "glacier", "soil"],
            "astronomy": ["planet", "star", "galaxy", "universe", "solar system", "comet", "asteroid", 
                         "telescope", "orbit", "moon", "sun", "black hole"],
            "environmental_science": ["ecosystem", "pollution", "conservation", "sustainability", 
                                     "biodiversity", "climate change", "resource", "habitat"]
        }
        
        # Topic-specific keywords within each discipline
        topic_keywords = {
            "mechanics": ["motion", "force", "velocity", "acceleration", "newton", "momentum", "inertia"],
            "electricity": ["current", "voltage", "circuit", "resistance", "conductor", "insulator"],
            "magnetism": ["magnet", "field", "compass", "attract", "repel", "pole", "electromagnetic"],
            "thermodynamics": ["heat", "temperature", "entropy", "energy", "thermal", "insulation"],
            "quantum": ["quantum", "particle", "wave", "uncertainty", "superposition", "entanglement"],
            "relativity": ["einstein", "relativity", "spacetime", "gravity", "mass", "energy"],
            
            "atomic structure": ["atom", "electron", "proton", "neutron", "orbital", "energy level"],
            "chemical bonds": ["bond", "ionic", "covalent", "metallic", "electronegativity"],
            "reactions": ["reaction", "reactant", "product", "catalyst", "equilibrium", "rate"],
            "organic": ["carbon", "hydrocarbon", "functional group", "polymer", "organic"],
            "biochemistry": ["protein", "enzyme", "carbohydrate", "lipid", "nucleic acid", "metabolism"],
            
            "cells": ["cell", "membrane", "organelle", "nucleus", "mitochondria", "cytoplasm"],
            "genetics": ["gene", "dna", "chromosome", "heredity", "mutation", "allele"],
            "evolution": ["evolution", "natural selection", "adaptation", "species", "darwin"],
            "ecology": ["ecosystem", "food web", "habitat", "population", "community", "niche"],
            "physiology": ["organ", "system", "tissue", "homeostasis", "regulation", "function"],
            
            # Additional topics could be added for other disciplines
        }
        
        results = {discipline: {} for discipline in discipline_keywords}
        lower_input = user_input.lower()
        
        # First, score each discipline
        discipline_scores = {}
        for discipline, keywords in discipline_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            discipline_scores[discipline] = min(1.0, score / 3)  # Normalize score
        
        # Then, for each discipline with a relevant score, check topic relevance
        for discipline, score in discipline_scores.items():
            if score > 0.2:  # Only check topics for relevant disciplines
                for topic in self.discipline_topics.get(discipline, []):
                    if topic in topic_keywords:
                        topic_score = sum(1 for keyword in topic_keywords[topic] if keyword in lower_input)
                        results[discipline][topic] = min(1.0, topic_score / 2)  # Normalize score
        
        return results
    
    def update_topic_proficiency(self, user_input: str, response_data: Dict[str, Any]) -> None:
        """
        Update the student's proficiency in specific science topics based on interactions.
        
        Args:
            user_input: The user's message
            response_data: Data about the response and interaction
        """
        # Categorize the current interaction
        topic_relevance = self.categorize_science_topic(user_input)
        
        # Look for indicators of understanding or confusion
        understanding_indicators = ["i understand", "that makes sense", "got it", "thanks", "clear now"]
        confusion_indicators = ["still confused", "i don't get it", "unclear", "what do you mean"]
        
        understanding_score = sum(1 for indicator in understanding_indicators if indicator in user_input.lower())
        confusion_score = sum(1 for indicator in confusion_indicators if indicator in user_input.lower())
        
        # Default slight positive score if no clear indicators
        proficiency_change = 0.1 if (understanding_score == 0 and confusion_score == 0) else understanding_score - confusion_score
        
        # Update discipline and topic scores based on relevance and proficiency change
        for discipline, topics in topic_relevance.items():
            if any(topics.values()):  # If any topics are relevant in this discipline
                discipline_relevance = sum(topics.values()) / len(topics) if topics else 0
                
                # Update discipline score
                self.science_disciplines[discipline]["interactions"] += 1
                self.science_disciplines[discipline]["score"] += proficiency_change * discipline_relevance
                
                # Normalize score
                self.science_disciplines[discipline]["score"] = max(-10, min(10, self.science_disciplines[discipline]["score"]))
                
                # Update individual topic scores
                for topic, relevance in topics.items():
                    if relevance > 0.2:  # Only update topics with sufficient relevance
                        if topic in self.topic_proficiency:
                            self.topic_proficiency[topic]["interactions"] += 1
                            self.topic_proficiency[topic]["score"] += proficiency_change * relevance
                            
                            # Normalize score
                            self.topic_proficiency[topic]["score"] = max(-10, min(10, self.topic_proficiency[topic]["score"]))
        
        # Update user metrics for strengths and weaknesses
        # Topics with positive scores are strengths, negative are weaknesses
        self.user_metrics["topic_strengths"] = {
            topic: data["score"] 
            for topic, data in self.topic_proficiency.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        self.user_metrics["topic_weaknesses"] = {
            topic: abs(data["score"]) 
            for topic, data in self.topic_proficiency.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Override to add science-specific processing.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: The agent's response
        """
        response = super().generate_response(user_input)
        
        # Update topic proficiency based on this interaction
        self.update_topic_proficiency(user_input, response.dict())
        
        return response
    
    def generate_experiment(self, topic: str, difficulty: int = None) -> Dict[str, Any]:
        """
        Generate a science experiment or demonstration related to a specific topic.
        
        Args:
            topic: The science topic for the experiment
            difficulty: Difficulty level (1-10, defaults to current learning level)
            
        Returns:
            Dict: Details of the experiment
        """
        if difficulty is None:
            difficulty = self.current_level.complexity
        
        # This would typically call the LLM to generate a custom experiment
        # For now, we'll just return a placeholder
        
        return {
            "title": f"Experiment about {topic}",
            "objective": f"To demonstrate {topic} principles at difficulty level {difficulty}",
            "materials": ["Material 1", "Material 2"],
            "procedure": ["Step 1", "Step 2"],
            "expected_results": "Expected results description",
            "explanation": "Scientific explanation of results",
            "safety_notes": "Safety considerations",
            "difficulty": difficulty
        }
    
    def get_relevant_diagrams(self, topic: str) -> List[str]:
        """
        Get descriptions of relevant diagrams for a science topic.
        
        Args:
            topic: The science topic
            
        Returns:
            List: Descriptions of relevant diagrams
        """
        # In a real implementation, this might retrieve actual diagram references
        # or generate custom diagrams
        
        return [f"Diagram illustrating {topic} concept"]