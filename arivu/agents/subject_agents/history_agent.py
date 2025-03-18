from typing import List, Dict, Any
from arivu.agents.base_agent import BaseAgent
from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class HistoryAgent(BaseAgent):
    """
    Agent specializing in history education across different time periods and regions.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_service: LLMService,
        memory: Memory,
        tool_registry: ToolRegistry,
    ):
        # Define history-specific learning levels
        from arivu.agents.base_agent import LearningLevel
        
        history_learning_levels = [
            LearningLevel(
                id="elementary",
                name="Elementary",
                description="Basic history concepts for elementary education",
                complexity=2,
                vocabulary_level=2,
                depth_of_explanation=3,
                example_frequency=7,
                assumed_prior_knowledge=1
            ),
            LearningLevel(
                id="middle_school",
                name="Middle School",
                description="History concepts for middle school education",
                complexity=4,
                vocabulary_level=4,
                depth_of_explanation=5,
                example_frequency=6,
                assumed_prior_knowledge=3
            ),
            LearningLevel(
                id="high_school",
                name="High School",
                description="History concepts for high school education",
                complexity=6,
                vocabulary_level=6,
                depth_of_explanation=7,
                example_frequency=5,
                assumed_prior_knowledge=5
            ),
            LearningLevel(
                id="undergraduate",
                name="Undergraduate",
                description="History concepts for undergraduate education",
                complexity=8,
                vocabulary_level=8,
                depth_of_explanation=8,
                example_frequency=4,
                assumed_prior_knowledge=7
            ),
            LearningLevel(
                id="graduate",
                name="Graduate",
                description="Advanced history concepts for graduate education",
                complexity=10,
                vocabulary_level=10,
                depth_of_explanation=10,
                example_frequency=3,
                assumed_prior_knowledge=9
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name="History Guide",
            description="history education across different periods and regions",
            llm_service=llm_service,
            memory=memory,
            tool_registry=tool_registry,
            learning_levels=history_learning_levels,
            default_level="high_school"
        )
        
        # History-specific era and region tracking
        self.historical_eras = {
            "ancient": {"score": 0, "interactions": 0},  # Before 500 CE
            "medieval": {"score": 0, "interactions": 0},  # 500-1500 CE
            "early_modern": {"score": 0, "interactions": 0},  # 1500-1800 CE
            "modern": {"score": 0, "interactions": 0},  # 1800-1945 CE
            "contemporary": {"score": 0, "interactions": 0}  # Post-1945
        }
        
        self.geographical_regions = {
            "americas": {"score": 0, "interactions": 0},
            "europe": {"score": 0, "interactions": 0},
            "africa": {"score": 0, "interactions": 0},
            "asia": {"score": 0, "interactions": 0},
            "middle_east": {"score": 0, "interactions": 0},
            "oceania": {"score": 0, "interactions": 0}
        }
        
        # Historical concepts and skills tracking
        self.history_concepts = {
            "chronology": {"score": 0, "interactions": 0},
            "cause_effect": {"score": 0, "interactions": 0},
            "historical_context": {"score": 0, "interactions": 0},
            "primary_sources": {"score": 0, "interactions": 0},
            "historiography": {"score": 0, "interactions": 0},
            "historical_perspectives": {"score": 0, "interactions": 0}
        }
    
    def _get_subject_specific_instructions(self) -> str:
        """Get history-specific instructions for the agent."""
        return """
        As a history education specialist:
        
        1. Present history with appropriate context, complexity, and multiple perspectives.
        2. Connect historical events to broader themes and patterns.
        3. Emphasize the importance of evidence and primary sources in historical understanding.
        4. Discuss cause and effect relationships in historical developments.
        5. Acknowledge areas of historical debate and different interpretations.
        6. Help students understand the relevance of historical events to the present.
        7. Present history from diverse cultural and geographical perspectives.
        8. Encourage critical thinking about historical narratives and sources.
        
        Historical eras include:
        - Ancient History (before 500 CE): Early civilizations, classical antiquity
        - Medieval History (500-1500 CE): Middle Ages, feudalism, early empires
        - Early Modern History (1500-1800 CE): Renaissance, exploration, colonization
        - Modern History (1800-1945 CE): Industrialization, nationalism, world wars
        - Contemporary History (post-1945): Cold War, decolonization, globalization
        
        Geographical regions include:
        - Americas: North, Central, and South American history
        - Europe: Western, Eastern, and Southern European history
        - Africa: North, West, East, Central, and Southern African history
        - Asia: East, South, Southeast, and Central Asian history
        - Middle East: Southwest Asian and North African history
        - Oceania: Australian, New Zealand, and Pacific Islander history
        
        For elementary level: Focus on basic chronology, significant people and events, 
        simple cause-effect relationships, and concrete historical concepts.
        
        For middle school level: Introduce more detailed narratives, begin discussing 
        multiple perspectives, and develop connections between historical periods.
        
        For high school level: Develop understanding of historical context, analyze 
        primary sources, explore multiple interpretations, and examine historical themes.
        
        For undergraduate level: Emphasize historiography, complex causation, analytical 
        skills, and thematic approaches to historical understanding.
        
        For graduate level: Focus on specialized historical knowledge, historiographical 
        debates, advanced source analysis, and original historical interpretation.
        """
    
    def _get_subject_tools(self) -> List[str]:
        """Get a list of tool IDs relevant to history education."""
        return [
            "timeline_generator",
            "map_visualizer",
            "primary_source_database",
            "historical_comparison_tool",
            "web_search"
        ]
    
    def categorize_history_topic(self, user_input: str) -> Dict[str, Dict[str, float]]:
        """
        Categorize the user input into relevant historical eras, regions, and concepts.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: Nested mapping of category -> topic -> relevance score (0-1)
        """
        # This is a simplified version. In a real implementation, 
        # this might use a classifier model or more sophisticated logic
        
        era_keywords = {
            "ancient": ["ancient", "classical", "antiquity", "bronze age", "iron age", "egypt", "rome", "greece", "mesopotamia"],
            "medieval": ["medieval", "middle ages", "feudal", "castle", "knight", "crusade", "byzantine"],
            "early_modern": ["renaissance", "reformation", "enlightenment", "colonial", "empire", "exploration"],
            "modern": ["industrial", "revolution", "world war", "nationalism", "imperialism", "19th century"],
            "contemporary": ["cold war", "digital", "globalization", "post-war", "20th century", "21st century"]
        }
        
        region_keywords = {
            "americas": ["america", "united states", "canada", "mexico", "brazil", "argentina", "andes", "caribbean"],
            "europe": ["europe", "britain", "france", "germany", "italy", "spain", "russia", "greece"],
            "africa": ["africa", "egypt", "ethiopia", "ghana", "mali", "kongo", "south africa", "sahara"],
            "asia": ["asia", "china", "japan", "india", "korea", "vietnam", "mongolia", "indonesia"],
            "middle_east": ["middle east", "ottoman", "persia", "arabia", "mesopotamia", "israel", "turkey", "iran"],
            "oceania": ["australia", "new zealand", "pacific", "polynesia", "melanesia", "micronesia", "aboriginal"]
        }
        
        concept_keywords = {
            "chronology": ["timeline", "chronology", "period", "era", "century", "decade", "date"],
            "cause_effect": ["cause", "effect", "result", "impact", "influence", "consequence", "led to"],
            "historical_context": ["context", "background", "setting", "circumstances", "environment", "conditions"],
            "primary_sources": ["source", "document", "artifact", "record", "evidence", "account", "testimony"],
            "historiography": ["interpretation", "historian", "scholarship", "analysis", "perspective", "theory", "debate"],
            "historical_perspectives": ["perspective", "viewpoint", "experience", "memory", "identity", "narrative"]
        }
        
        results = {
            "eras": {},
            "regions": {},
            "concepts": {}
        }
        
        lower_input = user_input.lower()
        
        # Score each era
        for era, keywords in era_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            if score > 0:
                results["eras"][era] = min(1.0, score / 3)  # Normalize score
        
        # Score each region
        for region, keywords in region_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            if score > 0:
                results["regions"][region] = min(1.0, score / 3)  # Normalize score
        
        # Score each concept
        for concept, keywords in concept_keywords.items():
            score = sum(1 for keyword in keywords if keyword in lower_input)
            if score > 0:
                results["concepts"][concept] = min(1.0, score / 3)  # Normalize score
        
        return results
    
    def update_historical_knowledge(self, user_input: str, response_data: Dict[str, Any]) -> None:
        """
        Update the student's knowledge of historical topics based on interactions.
        
        Args:
            user_input: The user's message
            response_data: Data about the response and interaction
        """
        # Categorize the current interaction
        topic_relevance = self.categorize_history_topic(user_input)
        
        # Look for indicators of understanding or confusion
        understanding_indicators = ["i understand", "that makes sense", "got it", "thanks", "clear now"]
        confusion_indicators = ["still confused", "i don't get it", "unclear", "what do you mean"]
        
        understanding_score = sum(1 for indicator in understanding_indicators if indicator in user_input.lower())
        confusion_score = sum(1 for indicator in confusion_indicators if indicator in user_input.lower())
        
        # Default slight positive score if no clear indicators
        proficiency_change = 0.1 if (understanding_score == 0 and confusion_score == 0) else understanding_score - confusion_score
        
        # Update era knowledge
        for era, relevance in topic_relevance["eras"].items():
            if relevance > 0.2:  # Only update eras with sufficient relevance
                self.historical_eras[era]["interactions"] += 1
                self.historical_eras[era]["score"] += proficiency_change * relevance
                
                # Normalize score
                self.historical_eras[era]["score"] = max(-10, min(10, self.historical_eras[era]["score"]))
        
        # Update region knowledge
        for region, relevance in topic_relevance["regions"].items():
            if relevance > 0.2:  # Only update regions with sufficient relevance
                self.geographical_regions[region]["interactions"] += 1
                self.geographical_regions[region]["score"] += proficiency_change * relevance
                
                # Normalize score
                self.geographical_regions[region]["score"] = max(-10, min(10, self.geographical_regions[region]["score"]))
        
        # Update concept knowledge
        for concept, relevance in topic_relevance["concepts"].items():
            if relevance > 0.2:  # Only update concepts with sufficient relevance
                self.history_concepts[concept]["interactions"] += 1
                self.history_concepts[concept]["score"] += proficiency_change * relevance
                
                # Normalize score
                self.history_concepts[concept]["score"] = max(-10, min(10, self.history_concepts[concept]["score"]))
        
        # Update user metrics for strengths and weaknesses
        # Combine era and region strengths into a single map of topics
        era_strengths = {
            f"era:{era}": data["score"] 
            for era, data in self.historical_eras.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        region_strengths = {
            f"region:{region}": data["score"] 
            for region, data in self.geographical_regions.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        concept_strengths = {
            f"concept:{concept}": data["score"] 
            for concept, data in self.history_concepts.items() 
            if data["score"] > 0 and data["interactions"] > 2
        }
        
        # Combine all strengths
        self.user_metrics["topic_strengths"] = {**era_strengths, **region_strengths, **concept_strengths}
        
        # Similarly for weaknesses
        era_weaknesses = {
            f"era:{era}": abs(data["score"]) 
            for era, data in self.historical_eras.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
        
        region_weaknesses = {
            f"region:{region}": abs(data["score"]) 
            for region, data in self.geographical_regions.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
        
        concept_weaknesses = {
            f"concept:{concept}": abs(data["score"]) 
            for concept, data in self.history_concepts.items() 
            if data["score"] < 0 and data["interactions"] > 2
        }
        
        # Combine all weaknesses
        self.user_metrics["topic_weaknesses"] = {**era_weaknesses, **region_weaknesses, **concept_weaknesses}
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Override to add history-specific processing.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict: The agent's response
        """
        response = super().generate_response(user_input)
        
        # Update historical knowledge based on this interaction
        self.update_historical_knowledge(user_input, response.dict())
        
        return response
    
    def generate_timeline(self, era: str = None, region: str = None) -> List[Dict[str, Any]]:
        """
        Generate a historical timeline for a specific era and/or region.
        
        Args:
            era: Optional historical era to focus on
            region: Optional geographical region to focus on
            
        Returns:
            List: Timeline events
        """
        # This would typically call the LLM to generate a customized timeline
        # For now, we'll just return a placeholder
        
        focus = ""
        if era and region:
            focus = f"{era} era in {region}"
        elif era:
            focus = f"{era} era"
        elif region:
            focus = f"history of {region}"
        else:
            focus = "world history"
        
        return [
            {
                "year": "Sample Year",
                "event": f"Sample event from {focus}",
                "significance": "Significance explanation would go here",
                "connections": ["Connection to other events"]
            }
        ]
    
    def suggest_primary_sources(self, topic: str, difficulty: int = None) -> List[Dict[str, Any]]:
        """
        Suggest primary sources related to a historical topic.
        
        Args:
            topic: The historical topic
            difficulty: Difficulty level (1-10, defaults to current learning level)
            
        Returns:
            List: Suggested primary sources
        """
        if difficulty is None:
            difficulty = self.current_level.complexity
        
        # This would typically access a database of primary sources or generate suggestions
        
        return [
            {
                "title": f"Sample primary source about {topic}",
                "type": "Document/Artifact/Image/etc.",
                "time_period": "Historical period",
                "description": "Description of the source",
                "analysis_questions": ["Question 1", "Question 2"],
                "difficulty": difficulty
            }
        ]
    
    def compare_historical_periods(self, period1: str, period2: str) -> Dict[str, Any]:
        """
        Compare two historical periods.
        
        Args:
            period1: First historical period
            period2: Second historical period
            
        Returns:
            Dict: Comparison information
        """
        # This would typically call the LLM to generate a detailed comparison
        
        return {
            "similarities": [f"Similarity between {period1} and {period2}"],
            "differences": [f"Difference between {period1} and {period2}"],
            "key_themes": ["Theme 1", "Theme 2"],
            "historical_connections": ["Connection 1", "Connection 2"]
        }