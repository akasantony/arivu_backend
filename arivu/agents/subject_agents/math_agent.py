from typing import List, Dict, Any
from arivu.agents.base_agent import BaseAgent
from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class MathAgent(BaseAgent):
    """
    Agent specializing in mathematics education.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_service: LLMService,
        memory: Memory,
        tool_registry: ToolRegistry,
    ):
        # Define math-specific learning levels
        from arivu.agents.base_agent import LearningLevel
        
        math_learning_levels = [
            LearningLevel(
                id="elementary",
                name="Elementary",
                description="Basic math concepts suitable for elementary education",
                complexity=2,
                vocabulary_level=2,
                depth_of_explanation=3,
                example_frequency=8,
                assumed_prior_knowledge=1
            ),
            LearningLevel(
                id="intermediate",
                name="Intermediate",
                description="Middle school to early high school math concepts",
                complexity=5,
                vocabulary_level=5,
                depth_of_explanation=5,
                example_frequency=6,
                assumed_prior_knowledge=4
            ),
            LearningLevel(
                id="advanced",
                name="Advanced",
                description="High school to early undergraduate math concepts",
                complexity=8,
                vocabulary_level=7,
                depth_of_explanation=8,
                example_frequency=5,
                assumed_prior_knowledge=7
            ),
            LearningLevel(
                id="expert",
                name="Expert",
                description="Advanced undergraduate to graduate level math concepts",
                complexity=10,
                vocabulary_level=9,
                depth_of_explanation=10,
                example_frequency=4,
                assumed_prior_knowledge=9
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name="Math Mentor",
            description="mathematics education",
            llm_service=llm_service,
            memory=memory,
            tool_registry=tool_registry,
            learning_levels=math_learning_levels,
            default_level="intermediate"
        )
        
        # Math-specific topic tracking
        self.math_topics = {
            "arithmetic": {"score": 0, "interactions": 0},
            "algebra": {"score": 0, "interactions": 0},
            "geometry": {"score": 0, "interactions": 0},
            "trigonometry": {"score": 0, "interactions": 0},
            "calculus": {"score": 0, "interactions": 0},
            "statistics": {"score": 0, "interactions": 0},
            "number_theory": {"score": 0, "interactions": 0},
            "discrete_math": {"score": 0, "interactions": 0}
        }
    
    def _get_subject_specific_instructions(self) -> str:
        """Get math-specific instructions for the agent."""
        return """
        As a math education specialist:
        
        1. Encourage problem-solving approaches rather than just providing answers.
        2. Use step-by-step explanations for mathematical processes.
        3. Provide visual representations when helpful (describe them in detail).
        4. Connect mathematical concepts to real-world applications.
        5. Use appropriate mathematical notation (while ensuring it's understandable).
        6. Provide multiple approaches to solving problems when relevant.
        7. Explain the "why" behind mathematical rules, not just the "how".
        8. Adjust complexity based on the student's current learning level.
        
        Subject areas include:
        - Arithmetic and number sense
        - Algebra and equations
        - Geometry and spatial reasoning
        - Trigonometry
        - Calculus and analysis
        - Statistics and probability
        - Number theory
        - Discrete mathematics
        
        For elementary level: Focus on building number sense, basic operations, 
        simple fractions, and introducing basic geometric concepts.
        
        For intermediate level: Develop algebraic thinking, work with equations, 
        expand geometric understanding, and introduce basic probability.
        
        For advanced level: Cover functions, trigonometry, introductory calculus,
        statistical analysis, and more complex problem solving.
        
        For expert level: Explore proof-based mathematics, advanced calculus, 
        abstract algebra, complex analysis, and mathematical modeling.
        """
    
    def _get_subject_tools(self) -> List[str]:
        """Get a list of tool IDs relevant to mathematics."""
        return [
            "calculator",
            "graph_generator",
            "equation_solver",
            "geometry_visualizer",
            "statistics_analyzer"
        ]
    
    def categorize_math_topic(self, user_input: str) -> Dict[str, float]:
        """
        Categorize the user input into relevant math topics.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: Mapping of topic to relevance score (0-1)
        """
        # This is a simplified version. In a real implementation, 
        # this might use a classifier model or more sophisticated logic
        
        topic_keywords = {
            "arithmetic": ["add", "subtract", "multiply", "divide", "fraction", "decimal", "percent"],
            "algebra": ["equation", "variable", "solve for", "expression", "factor", "polynomial"],
            "geometry": ["shape", "angle", "triangle", "circle", "area", "volume", "perimeter"],
            "trigonometry": ["sine", "cosine", "tangent", "angle", "radian", "degree", "triangle"],
            "calculus": ["derivative", "integral", "limit", "function", "rate of change", "optimize"],
            "statistics": ["average", "mean", "median", "mode", "probability", "distribution", "data"],
            "number_theory": ["prime", "factor", "divisor", "remainder", "modulo", "congruence"],
            "discrete_math": ["graph", "set", "logic", "proof", "combinatorics", "recursion"]
        }
        
        results = {}
        lower_input = user_input.lower()
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            results[topic] = min(1.0, score / 3)  # Normalize score, max out at 1.0
        
        return results
    
    def update_topic_proficiency(self, user_input: str, response_data: Dict[str, Any]) -> None:
        """
        Update the student's proficiency in specific math topics based on interactions.
        
        Args:
            user_input: The user's message
            response_data: Data about the response and interaction
        """
        # Categorize the current interaction
        topic_relevance = self.categorize_math_topic(user_input)
        
        # Look for indicators of understanding or confusion
        understanding_indicators = ["i understand", "that makes sense", "got it", "thanks", "clear now"]
        confusion_indicators = ["still confused", "i don't get it", "unclear", "what do you mean"]
        
        understanding_score = sum(1 for indicator in understanding_indicators if indicator in user_input.lower())
        confusion_score = sum(1 for indicator in confusion_indicators if indicator in user_input.lower())
        
        # Default slight positive score if no clear indicators
        proficiency_change = 0.1 if (understanding_score == 0 and confusion_score == 0) else understanding_score - confusion_score
        
        # Update topic scores based on relevance and proficiency change
        for topic, relevance in topic_relevance.items():
            if relevance > 0.2:  # Only update topics with sufficient relevance
                self.math_topics[topic]["interactions"] += 1
                
                # Weighted update based on relevance
                self.math_topics[topic]["score"] += proficiency_change * relevance
                
                # Normalize score to stay within reasonable bounds
                self.math_topics[topic]["score"] = max(-10, min(10, self.math_topics[topic]["score"]))
        
        # Update user metrics for strengths and weaknesses
        # Topics with positive scores are strengths, negative are weaknesses
        self.user_metrics["topic_strengths"] = {
            topic: data["score"] 
            for topic, data in self.math_topics.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        self.user_metrics["topic_weaknesses"] = {
            topic: abs(data["score"]) 
            for topic, data in self.math_topics.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Override to add math-specific processing.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: The agent's response
        """
        response = super().generate_response(user_input)
        
        # Update topic proficiency based on this interaction
        self.update_topic_proficiency(user_input, response.dict())
        
        return response
    
    def generate_practice_problems(self, topic: str, difficulty: int = None) -> List[Dict[str, Any]]:
        """
        Generate practice problems for a specific math topic.
        
        Args:
            topic: The math topic to generate problems for
            difficulty: Difficulty level (1-10, defaults to current learning level)
            
        Returns:
            List: A list of practice problems with solutions
        """
        if difficulty is None:
            difficulty = self.current_level.complexity
        
        # This would typically call the LLM to generate custom problems
        # For now, we'll just return a placeholder
        
        return [
            {
                "problem": f"Sample {topic} problem at difficulty level {difficulty}",
                "solution": "Sample solution would go here",
                "hints": ["Hint 1", "Hint 2"],
                "difficulty": difficulty
            }
        ]