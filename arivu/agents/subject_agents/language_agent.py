from typing import List, Dict, Any
from arivu.agents.base_agent import BaseAgent
from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class LanguageAgent(BaseAgent):
    """
    Agent specializing in language arts education, including reading, writing, 
    grammar, literature, and communication skills.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_service: LLMService,
        memory: Memory,
        tool_registry: ToolRegistry,
        target_language: str = "english"
    ):
        # Define language-specific learning levels
        from arivu.agents.base_agent import LearningLevel
        
        language_learning_levels = [
            LearningLevel(
                id="beginner",
                name="Beginner",
                description="Basic language skills for beginners",
                complexity=2,
                vocabulary_level=2,
                depth_of_explanation=3,
                example_frequency=9,
                assumed_prior_knowledge=1
            ),
            LearningLevel(
                id="intermediate",
                name="Intermediate",
                description="Language skills for intermediate learners",
                complexity=5,
                vocabulary_level=5,
                depth_of_explanation=5,
                example_frequency=7,
                assumed_prior_knowledge=4
            ),
            LearningLevel(
                id="advanced",
                name="Advanced",
                description="Advanced language skills for proficient learners",
                complexity=8,
                vocabulary_level=8,
                depth_of_explanation=7,
                example_frequency=5,
                assumed_prior_knowledge=7
            ),
            LearningLevel(
                id="fluent",
                name="Fluent",
                description="Language skills for nearly fluent learners",
                complexity=10,
                vocabulary_level=10,
                depth_of_explanation=9,
                example_frequency=4,
                assumed_prior_knowledge=9
            )
        ]
        
        self.target_language = target_language
        language_name = target_language.capitalize()
        
        super().__init__(
            agent_id=agent_id,
            name=f"{language_name} Language Tutor",
            description=f"{language_name} language education and literacy skills",
            llm_service=llm_service,
            memory=memory,
            tool_registry=tool_registry,
            learning_levels=language_learning_levels,
            default_level="intermediate"
        )
        
        # Language-specific skill tracking
        self.language_skills = {
            "vocabulary": {"score": 0, "interactions": 0},
            "grammar": {"score": 0, "interactions": 0},
            "reading": {"score": 0, "interactions": 0},
            "writing": {"score": 0, "interactions": 0},
            "listening": {"score": 0, "interactions": 0},
            "speaking": {"score": 0, "interactions": 0},
            "comprehension": {"score": 0, "interactions": 0},
            "literary_analysis": {"score": 0, "interactions": 0}
        }
        
        # Error tracking for language learning
        self.common_errors = {
            "grammar": [],
            "vocabulary": [],
            "pronunciation": [],
            "sentence_structure": []
        }
        
        # Vocabulary tracking
        self.vocabulary_bank = {
            "mastered": [],  # Words the student consistently uses correctly
            "learning": [],  # Words the student is currently learning
            "challenging": []  # Words the student often struggles with
        }
    
    def _get_subject_specific_instructions(self) -> str:
        """Get language-specific instructions for the agent."""
        return f"""
        As a {self.target_language} language education specialist:
        
        1. Adjust vocabulary and complexity to match the student's proficiency level.
        2. Provide clear examples and contextual uses of language concepts.
        3. Give constructive feedback on language usage, highlighting strengths first.
        4. Explain grammar rules in simple terms with practical examples.
        5. Encourage active language production appropriate to the student's level.
        6. Introduce new vocabulary with contextual examples and connections.
        7. Incorporate cultural context when relevant to language learning.
        8. For advanced learners, focus on nuance, idioms, and fluency.
        
        Key language skill areas include:
        - Vocabulary development and usage
        - Grammar and syntax
        - Reading comprehension and analysis
        - Writing skills and composition
        - Listening comprehension
        - Speaking fluency and pronunciation
        - Literary appreciation and analysis
        - Effective communication strategies
        
        For beginner level: Focus on basic vocabulary, simple sentence structures, 
        common expressions, and foundational grammar concepts.
        
        For intermediate level: Develop broader vocabulary, more complex grammar, 
        paragraph-level writing, and deeper reading comprehension.
        
        For advanced level: Refine stylistic elements, address subtle grammar points, 
        analyze literature, and develop complex arguments in writing.
        
        For fluent level: Focus on mastery of idioms, cultural nuances, advanced 
        literary analysis, persuasive communication, and professional/academic language.
        """
    
    def _get_subject_tools(self) -> List[str]:
        """Get a list of tool IDs relevant to language education."""
        return [
            "dictionary",
            "thesaurus",
            "grammar_checker",
            "text_analyzer",
            "translation_tool"
        ]
    
    def categorize_language_skill(self, user_input: str) -> Dict[str, float]:
        """
        Categorize the user input into relevant language skills.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: Mapping of skill to relevance score (0-1)
        """
        # This is a simplified version. In a real implementation, 
        # this might use a classifier model or more sophisticated NLP
        
        skill_keywords = {
            "vocabulary": ["word", "meaning", "definition", "synonym", "antonym", "term", "phrase"],
            "grammar": ["tense", "verb", "noun", "adjective", "adverb", "sentence", "structure", "rule"],
            "reading": ["read", "book", "article", "passage", "comprehension", "understand"],
            "writing": ["write", "essay", "paragraph", "composition", "story", "author"],
            "listening": ["listen", "hear", "audio", "podcast", "speech", "pronunciation"],
            "speaking": ["speak", "say", "pronounce", "conversation", "dialogue", "accent"],
            "comprehension": ["understand", "meaning", "interpretation", "context", "explain"],
            "literary_analysis": ["analyze", "theme", "character", "plot", "setting", "symbolism", "literature"]
        }
        
        results = {}
        lower_input = user_input.lower()
        
        for skill, keywords in skill_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            results[skill] = min(1.0, score / 3)  # Normalize score
        
        return results
    
    def analyze_language_errors(self, user_input: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze the user's input for language errors.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: Categorized language errors with suggestions
        """
        # This would typically use more sophisticated NLP tools
        # For demonstration, we'll implement a very simple analysis
        
        errors = {
            "grammar": [],
            "vocabulary": [],
            "sentence_structure": []
        }
        
        # Extremely simplified grammar checking
        # In a real implementation, this would use proper NLP tools
        if " i " in user_input.lower():
            errors["grammar"].append({
                "error": "lowercase personal pronoun",
                "context": "personal pronoun 'i' should be capitalized",
                "suggestion": "Replace 'i' with 'I'"
            })
        
        # Check for some common sentence structure issues
        if user_input.endswith(" and"):
            errors["sentence_structure"].append({
                "error": "dangling conjunction",
                "context": "sentence ends with conjunction 'and'",
                "suggestion": "Complete the sentence after 'and'"
            })
        
        return errors
    
    def update_language_proficiency(self, user_input: str, response_data: Dict[str, Any]) -> None:
        """
        Update the student's proficiency in language skills based on interactions.
        
        Args:
            user_input: The user's message
            response_data: Data about the response and interaction
        """
        # Categorize the current interaction
        skill_relevance = self.categorize_language_skill(user_input)
        
        # Look for indicators of understanding or confusion
        understanding_indicators = ["i understand", "that makes sense", "got it", "thanks", "clear now"]
        confusion_indicators = ["still confused", "i don't get it", "unclear", "what do you mean"]
        
        understanding_score = sum(1 for indicator in understanding_indicators if indicator in user_input.lower())
        confusion_score = sum(1 for indicator in confusion_indicators if indicator in user_input.lower())
        
        # Default slight positive score if no clear indicators
        proficiency_change = 0.1 if (understanding_score == 0 and confusion_score == 0) else understanding_score - confusion_score
        
        # Analyze for errors
        errors = self.analyze_language_errors(user_input)
        
        # Adjust proficiency based on errors
        for error_type, error_list in errors.items():
            if error_list and error_type in self.language_skills:
                # More errors means less proficiency
                proficiency_change -= 0.2 * len(error_list)
        
        # Update skill scores based on relevance and proficiency change
        for skill, relevance in skill_relevance.items():
            if relevance > 0.2:  # Only update skills with sufficient relevance
                self.language_skills[skill]["interactions"] += 1
                
                # Weighted update based on relevance
                self.language_skills[skill]["score"] += proficiency_change * relevance
                
                # Normalize score
                self.language_skills[skill]["score"] = max(-10, min(10, self.language_skills[skill]["score"]))
        
        # Update user metrics for strengths and weaknesses
        self.user_metrics["topic_strengths"] = {
            skill: data["score"] 
            for skill, data in self.language_skills.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        self.user_metrics["topic_weaknesses"] = {
            skill: abs(data["score"]) 
            for skill, data in self.language_skills.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
        
        # Update common errors tracking
        for error_type, error_list in errors.items():
            for error in error_list:
                if error not in self.common_errors[error_type]:
                    self.common_errors[error_type].append(error)
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Override to add language-specific processing.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: The agent's response
        """
        response = super().generate_response(user_input)
        
        # Update language proficiency based on this interaction
        self.update_language_proficiency(user_input, response.dict())
        
        return response
    
    def generate_exercises(self, skill: str, difficulty: int = None) -> List[Dict[str, Any]]:
        """
        Generate language exercises for a specific skill.
        
        Args:
            skill: The language skill to generate exercises for
            difficulty: Difficulty level (1-10, defaults to current learning level)
            
        Returns:
            List: A list of language exercises with solutions
        """
        if difficulty is None:
            difficulty = self.current_level.complexity
        
        # This would typically call the LLM to generate custom exercises
        # For now, we'll just return a placeholder
        
        return [
            {
                "instruction": f"Sample {skill} exercise at difficulty level {difficulty}",
                "exercise": "Exercise content would go here",
                "solution": "Sample solution would go here",
                "hints": ["Hint 1", "Hint 2"],
                "difficulty": difficulty
            }
        ]
    
    def get_vocabulary_recommendations(self) -> Dict[str, List[str]]:
        """
        Get vocabulary recommendations based on the student's proficiency.
        
        Returns:
            Dict: A dictionary with recommended vocabulary words for review and learning
        """
        # This would typically involve more sophisticated analysis of the student's vocabulary use
        
        return {
            "review": self.vocabulary_bank["challenging"][:5],
            "new_words": []  # Would be populated with appropriate new words for the student's level
        }
    
    def generate_writing_prompt(self, topic: str = None, difficulty: int = None) -> Dict[str, Any]:
        """
        Generate a writing prompt for the student.
        
        Args:
            topic: Optional topic for the writing prompt
            difficulty: Difficulty level (1-10, defaults to current learning level)
            
        Returns:
            Dict: A writing prompt with guidelines
        """
        if difficulty is None:
            difficulty = self.current_level.complexity
        
        # This would typically call the LLM to generate a custom writing prompt
        
        return {
            "prompt": f"Writing prompt about {topic or 'a general topic'} at level {difficulty}",
            "guidelines": [
                "Guideline 1",
                "Guideline 2"
            ],
            "word_count_recommendation": 100 * difficulty,
            "difficulty": difficulty
        }