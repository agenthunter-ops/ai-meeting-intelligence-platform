"""
AI Meeting Intelligence Platform - LLM Prompt Templates
======================================================
This module contains carefully crafted prompt templates for extracting
structured insights from meeting transcripts using Large Language Models.

Key Features:
- Structured output prompts with JSON schema validation
- Different prompt types for various insight extraction tasks
- Chain-of-thought prompting for better reasoning
- Few-shot examples for improved accuracy
- Configurable prompt parameters for different meeting types
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class PromptTemplate:
    """
    Base class for LLM prompt templates.
    Provides common functionality for prompt formatting and validation.
    """
    
    def __init__(self, template: str, required_vars: List[str] = None):
        """
        Initialize prompt template.
        
        Args:
            template (str): Template string with placeholder variables
            required_vars (List[str]): List of required template variables
        """
        self.template = template
        self.required_vars = required_vars or []
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            str: Formatted prompt string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Check for required variables
        missing_vars = [var for var in self.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")
    
    def validate_output(self, output: str) -> Dict[str, Any]:
        """
        Validate and parse LLM output.
        Should be overridden by subclasses for specific validation.
        
        Args:
            output (str): Raw LLM output
            
        Returns:
            Dict[str, Any]: Parsed and validated output
        """
        try:
            # Try to parse as JSON
            return json.loads(output)
        except json.JSONDecodeError:
            # Return as plain text if not valid JSON
            return {"text": output, "parsed": False}

class ActionItemExtractionPrompt(PromptTemplate):
    """
    Prompt template for extracting action items from meeting transcripts.
    Identifies tasks, assignments, deadlines, and priorities.
    """
    
    def __init__(self):
        # Detailed prompt template with examples and instructions
        template = """You are an expert meeting analyst specializing in action item extraction. Your task is to analyze the meeting transcript and extract all action items (tasks, assignments, deliverables) with their details.

INSTRUCTIONS:
1. Read the transcript carefully and identify all action items
2. For each action item, determine:
   - What needs to be done (task description)
   - Who is responsible (assignee)
   - When it needs to be completed (due date/timeframe)
   - Priority level (urgent, high, medium, low)
   - Current status (if mentioned)

3. Output your response as valid JSON with the following structure:

{{
  "action_items": [
    {{
      "task": "Clear, specific description of what needs to be done",
      "assignee": "Name of person responsible (or 'unassigned' if not specified)",
      "due_date": "Due date in YYYY-MM-DD format or 'not_specified'",
      "priority": "urgent|high|medium|low",
      "status": "open|in_progress|completed",
      "context": "Brief context or background for the task",
      "confidence": 0.95
    }}
  ],
  "summary": {{
    "total_action_items": 0,
    "urgent_items": 0,
    "high_priority_items": 0,
    "unassigned_items": 0
  }}
}}

EXAMPLES:

Example 1:
Transcript: "John, can you send the quarterly report to the client by Friday? It's really important we get this done."
Output:
{{
  "action_items": [
    {{
      "task": "Send quarterly report to client",
      "assignee": "John",
      "due_date": "not_specified",
      "priority": "high", 
      "status": "open",
      "context": "Client deliverable mentioned as important",
      "confidence": 0.9
    }}
  ]
}}

Example 2:
Transcript: "We need to update the website before the product launch next month. Sarah will handle the design updates and Mike will work on the backend. This is critical for our launch success."
Output:
{{
  "action_items": [
    {{
      "task": "Update website design for product launch",
      "assignee": "Sarah",
      "due_date": "not_specified",
      "priority": "urgent",
      "status": "open", 
      "context": "Critical for product launch success",
      "confidence": 0.95
    }},
    {{
      "task": "Update website backend for product launch", 
      "assignee": "Mike",
      "due_date": "not_specified",
      "priority": "urgent",
      "status": "open",
      "context": "Critical for product launch success",
      "confidence": 0.95
    }}
  ]
}}

Now analyze this meeting transcript:

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}

Please extract all action items following the format above. Be thorough but only include genuine action items (tasks that need to be completed by someone). Do not include general discussion points or decisions unless they involve specific actions."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees'])
    
    def validate_output(self, output: str) -> Dict[str, Any]:
        """
        Validate action item extraction output.
        Ensures proper JSON structure and required fields.
        """
        try:
            parsed = json.loads(output)
            
            # Validate structure
            if 'action_items' not in parsed:
                raise ValueError("Missing 'action_items' field")
            
            # Validate each action item
            validated_items = []
            for item in parsed['action_items']:
                validated_item = {
                    'task': str(item.get('task', '')),
                    'assignee': str(item.get('assignee', 'unassigned')),
                    'due_date': str(item.get('due_date', 'not_specified')),
                    'priority': str(item.get('priority', 'medium')),
                    'status': str(item.get('status', 'open')),
                    'context': str(item.get('context', '')),
                    'confidence': float(item.get('confidence', 0.8))
                }
                
                # Validate priority values
                if validated_item['priority'] not in ['urgent', 'high', 'medium', 'low']:
                    validated_item['priority'] = 'medium'
                
                # Validate status values
                if validated_item['status'] not in ['open', 'in_progress', 'completed']:
                    validated_item['status'] = 'open'
                
                # Validate confidence range
                if not (0.0 <= validated_item['confidence'] <= 1.0):
                    validated_item['confidence'] = 0.8
                
                validated_items.append(validated_item)
            
            # Generate summary
            summary = {
                'total_action_items': len(validated_items),
                'urgent_items': len([i for i in validated_items if i['priority'] == 'urgent']),
                'high_priority_items': len([i for i in validated_items if i['priority'] == 'high']),
                'unassigned_items': len([i for i in validated_items if i['assignee'] == 'unassigned'])
            }
            
            return {
                'action_items': validated_items,
                'summary': summary,
                'parsed': True
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {
                'error': f'Failed to parse action items: {str(e)}',
                'raw_output': output,
                'parsed': False
            }

class DecisionExtractionPrompt(PromptTemplate):
    """
    Prompt template for extracting decisions from meeting transcripts.
    Identifies choices made, their rationale, and impact.
    """
    
    def __init__(self):
        template = """You are an expert meeting analyst specializing in decision extraction. Your task is to analyze the meeting transcript and extract all significant decisions made during the meeting.

INSTRUCTIONS:
1. Identify all decisions made during the meeting
2. For each decision, determine:
   - What was decided (clear, specific decision)
   - The rationale or reasoning behind the decision
   - Who made or approved the decision
   - Impact level (critical, high, medium, low)
   - Any alternatives that were considered
   - Implementation details if mentioned

3. Output your response as valid JSON:

{{
  "decisions": [
    {{
      "decision": "Clear statement of what was decided",
      "rationale": "Reasoning behind the decision",
      "decision_maker": "Person or group who made the decision",
      "impact": "critical|high|medium|low",
      "alternatives_considered": ["alternative 1", "alternative 2"],
      "implementation_notes": "Any mentioned implementation details",
      "context": "Background context for the decision",
      "confidence": 0.9
    }}
  ],
  "summary": {{
    "total_decisions": 0,
    "critical_decisions": 0,
    "high_impact_decisions": 0
  }}
}}

EXAMPLES:

Example 1:
Transcript: "After discussing the budget options, we've decided to go with Option B - the mid-tier package. It gives us the features we need while staying within our budget constraints. Sarah approved this decision."
Output:
{{
  "decisions": [
    {{
      "decision": "Selected mid-tier package (Option B) for the project",
      "rationale": "Provides needed features while staying within budget constraints",
      "decision_maker": "Sarah",
      "impact": "high",
      "alternatives_considered": ["Other budget options mentioned"],
      "implementation_notes": "",
      "context": "Budget planning discussion",
      "confidence": 0.95
    }}
  ]
}}

Now analyze this meeting transcript:

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}

Please extract all decisions following the format above. Focus on concrete decisions that were made, not just topics that were discussed."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees'])
    
    def validate_output(self, output: str) -> Dict[str, Any]:
        """Validate decision extraction output."""
        try:
            parsed = json.loads(output)
            
            if 'decisions' not in parsed:
                raise ValueError("Missing 'decisions' field")
            
            validated_decisions = []
            for decision in parsed['decisions']:
                validated_decision = {
                    'decision': str(decision.get('decision', '')),
                    'rationale': str(decision.get('rationale', '')),
                    'decision_maker': str(decision.get('decision_maker', 'not_specified')),
                    'impact': str(decision.get('impact', 'medium')),
                    'alternatives_considered': decision.get('alternatives_considered', []),
                    'implementation_notes': str(decision.get('implementation_notes', '')),
                    'context': str(decision.get('context', '')),
                    'confidence': float(decision.get('confidence', 0.8))
                }
                
                # Validate impact level
                if validated_decision['impact'] not in ['critical', 'high', 'medium', 'low']:
                    validated_decision['impact'] = 'medium'
                
                validated_decisions.append(validated_decision)
            
            summary = {
                'total_decisions': len(validated_decisions),
                'critical_decisions': len([d for d in validated_decisions if d['impact'] == 'critical']),
                'high_impact_decisions': len([d for d in validated_decisions if d['impact'] == 'high'])
            }
            
            return {
                'decisions': validated_decisions,
                'summary': summary,
                'parsed': True
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {
                'error': f'Failed to parse decisions: {str(e)}',
                'raw_output': output,
                'parsed': False
            }

class SentimentAnalysisPrompt(PromptTemplate):
    """
    Prompt template for analyzing meeting sentiment and tone.
    Identifies overall mood, participant engagement, and emotional context.
    """
    
    def __init__(self):
        template = """You are an expert in sentiment analysis and meeting dynamics. Your task is to analyze the emotional tone and sentiment of the meeting transcript.

INSTRUCTIONS:
1. Analyze the overall sentiment of the meeting
2. Identify sentiment by speaker if possible
3. Note any significant emotional moments or tone shifts
4. Assess the level of engagement and collaboration

Output your response as valid JSON:

{{
  "overall_sentiment": {{
    "polarity": "positive|neutral|negative",
    "score": 0.5,
    "confidence": 0.9,
    "description": "Brief description of overall meeting tone"
  }},
  "speaker_sentiment": {{
    "Speaker Name": {{
      "polarity": "positive|neutral|negative", 
      "score": 0.3,
      "dominant_emotions": ["engaged", "frustrated", "enthusiastic"]
    }}
  }},
  "engagement_analysis": {{
    "overall_engagement": "high|medium|low",
    "collaboration_level": "high|medium|low",
    "energy_level": "high|medium|low",
    "notable_moments": [
      {{
        "timestamp": "approximate time or segment",
        "description": "What happened",
        "sentiment_impact": "positive|negative|neutral"
      }}
    ]
  }},
  "summary": {{
    "meeting_mood": "Brief description of meeting atmosphere",
    "key_sentiment_drivers": ["factor 1", "factor 2"],
    "recommendations": ["suggestion 1", "suggestion 2"]
  }}
}}

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}

Analyze the sentiment and emotional dynamics of this meeting."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees'])

class MeetingSummaryPrompt(PromptTemplate):
    """
    Prompt template for generating comprehensive meeting summaries.
    Creates executive summaries with key points and outcomes.
    """
    
    def __init__(self):
        template = """You are an expert meeting summarizer. Create a comprehensive but concise summary of the meeting that captures all key information.

INSTRUCTIONS:
1. Create an executive summary (2-3 sentences)
2. List key topics discussed
3. Highlight important outcomes
4. Note next steps and follow-ups
5. Include relevant metrics or data mentioned

Output as JSON:

{{
  "executive_summary": "2-3 sentence high-level summary",
  "key_topics": [
    {{
      "topic": "Topic name",
      "description": "What was discussed about this topic",
      "time_spent": "estimate of time spent discussing"
    }}
  ],
  "outcomes": [
    {{
      "type": "decision|action|agreement|announcement",
      "description": "What was the outcome",
      "importance": "high|medium|low"
    }}
  ],
  "next_steps": [
    "Next step 1",
    "Next step 2"
  ],
  "metrics_mentioned": [
    {{
      "metric": "Name of metric",
      "value": "Value mentioned",
      "context": "Context around the metric"
    }}
  ],
  "attendee_participation": {{
    "active_participants": ["name1", "name2"],
    "participation_notes": "Notes about who contributed what"
  }}
}}

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}
- Duration: {duration}

Generate a comprehensive summary following the format above."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees', 'duration'])

class RiskIdentificationPrompt(PromptTemplate):
    """
    Prompt template for identifying risks, concerns, and blockers mentioned in meetings.
    """
    
    def __init__(self):
        template = """You are an expert risk analyst. Your task is to identify potential risks, concerns, blockers, and issues mentioned in the meeting transcript.

INSTRUCTIONS:
1. Identify explicit risks that were mentioned
2. Identify implicit concerns or potential problems
3. Note any blockers or obstacles discussed
4. Assess risk severity and likelihood
5. Note any mitigation strategies mentioned

Output as JSON:

{{
  "risks": [
    {{
      "risk": "Description of the risk",
      "category": "technical|business|resource|timeline|quality|external",
      "severity": "critical|high|medium|low",
      "likelihood": "very_likely|likely|possible|unlikely",
      "impact": "Description of potential impact",
      "mitigation_mentioned": "Any mitigation strategies discussed",
      "owner": "Person responsible for addressing (if mentioned)",
      "confidence": 0.8
    }}
  ],
  "blockers": [
    {{
      "blocker": "Description of blocker",
      "affected_area": "What area/project is affected",
      "resolution_needed": "What needs to happen to resolve",
      "urgency": "urgent|high|medium|low"
    }}
  ],
  "concerns": [
    {{
      "concern": "Description of concern",
      "raised_by": "Who raised the concern",
      "discussion_outcome": "How it was addressed in the meeting"
    }}
  ]
}}

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}

Identify all risks, blockers, and concerns from this meeting."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees'])

class ComprehensiveInsightPrompt(PromptTemplate):
    """
    Master prompt that extracts all types of insights in one pass.
    More efficient for shorter transcripts but may be less accurate for complex meetings.
    """
    
    def __init__(self):
        template = """You are an expert meeting intelligence analyst. Analyze the meeting transcript and extract comprehensive insights including action items, decisions, sentiment, risks, and key topics.

MEETING TRANSCRIPT:
{transcript}

MEETING CONTEXT:
- Meeting Type: {meeting_type}
- Meeting Date: {meeting_date}
- Attendees: {attendees}
- Duration: {duration}

Please provide a comprehensive analysis in the following JSON format:

{{
  "action_items": [
    {{
      "task": "What needs to be done",
      "assignee": "Who is responsible",
      "due_date": "When it's due (YYYY-MM-DD or 'not_specified')",
      "priority": "urgent|high|medium|low",
      "confidence": 0.9
    }}
  ],
  "decisions": [
    {{
      "decision": "What was decided",
      "rationale": "Why this decision was made",
      "impact": "critical|high|medium|low",
      "decision_maker": "Who made the decision"
    }}
  ],
  "key_topics": [
    {{
      "topic": "Topic name",
      "summary": "Brief summary of discussion",
      "time_allocation": "How much time was spent",
      "participants": ["Who participated in this discussion"]
    }}
  ],
  "sentiment": {{
    "overall": "positive|neutral|negative",
    "score": 0.5,
    "engagement": "high|medium|low",
    "notable_moments": ["Any significant emotional moments"]
  }},
  "risks_and_concerns": [
    {{
      "item": "Description of risk or concern",
      "type": "risk|concern|blocker",
      "severity": "critical|high|medium|low"
    }}
  ],
  "summary": {{
    "executive_summary": "2-3 sentence high-level summary",
    "next_meeting_prep": ["Items to prepare for next meeting"],
    "follow_up_needed": ["Areas requiring follow-up"]
  }}
}}

Be thorough but concise. Only include items that are clearly present in the transcript."""

        super().__init__(template, required_vars=['transcript', 'meeting_type', 'meeting_date', 'attendees', 'duration'])

# Prompt factory for easy selection
class PromptFactory:
    """
    Factory class for creating and managing different prompt templates.
    Provides easy access to specialized prompts for different analysis types.
    """
    
    _prompts = {
        'action_items': ActionItemExtractionPrompt,
        'decisions': DecisionExtractionPrompt,
        'sentiment': SentimentAnalysisPrompt,
        'summary': MeetingSummaryPrompt,
        'risks': RiskIdentificationPrompt,
        'comprehensive': ComprehensiveInsightPrompt
    }
    
    @classmethod
    def get_prompt(cls, prompt_type: str) -> PromptTemplate:
        """
        Get a prompt template by type.
        
        Args:
            prompt_type (str): Type of prompt to retrieve
            
        Returns:
            PromptTemplate: The requested prompt template
            
        Raises:
            ValueError: If prompt type is not found
        """
        if prompt_type not in cls._prompts:
            available = ', '.join(cls._prompts.keys())
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {available}")
        
        return cls._prompts[prompt_type]()
    
    @classmethod
    def get_available_prompts(cls) -> List[str]:
        """Get list of available prompt types."""
        return list(cls._prompts.keys())

# Utility functions for prompt management
def format_meeting_context(
    meeting_type: str = "general",
    meeting_date: Optional[str] = None, 
    attendees: Optional[List[str]] = None,
    duration: Optional[str] = None
) -> Dict[str, str]:
    """
    Format meeting context for prompt templates.
    
    Args:
        meeting_type (str): Type of meeting
        meeting_date (str): Meeting date
        attendees (List[str]): List of attendees
        duration (str): Meeting duration
        
    Returns:
        Dict[str, str]: Formatted context dictionary
    """
    if meeting_date is None:
        meeting_date = datetime.now().strftime("%Y-%m-%d")
    
    if attendees is None:
        attendees = []
    
    attendee_str = ", ".join(attendees) if attendees else "Not specified"
    
    return {
        'meeting_type': meeting_type,
        'meeting_date': meeting_date,
        'attendees': attendee_str,
        'duration': duration or "Not specified"
    }

def extract_insights_with_prompts(
    transcript: str,
    insight_types: List[str],
    meeting_context: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Extract multiple types of insights using different prompt templates.
    
    Args:
        transcript (str): Meeting transcript text
        insight_types (List[str]): List of insight types to extract
        meeting_context (Dict[str, str]): Meeting context information
        
    Returns:
        Dict[str, Any]: Dictionary with insights for each requested type
    """
    if meeting_context is None:
        meeting_context = format_meeting_context()
    
    results = {}
    
    for insight_type in insight_types:
        try:
            prompt = PromptFactory.get_prompt(insight_type)
            formatted_prompt = prompt.format(transcript=transcript, **meeting_context)
            
            # This would be called with your LLM client
            # llm_output = your_llm_client.generate(formatted_prompt)
            # validated_output = prompt.validate_output(llm_output)
            
            results[insight_type] = {
                'prompt': formatted_prompt,
                'status': 'ready'
                # 'output': validated_output  # Would contain actual results
            }
            
        except Exception as e:
            results[insight_type] = {
                'error': str(e),
                'status': 'error'
            }
    
    return results

# Export all prompt classes and utilities
__all__ = [
    'PromptTemplate', 'ActionItemExtractionPrompt', 'DecisionExtractionPrompt',
    'SentimentAnalysisPrompt', 'MeetingSummaryPrompt', 'RiskIdentificationPrompt',
    'ComprehensiveInsightPrompt', 'PromptFactory',
    'format_meeting_context', 'extract_insights_with_prompts'
]

# Example usage and testing
if __name__ == "__main__":
    # Example of using the prompt factory
    print("🤖 AI Meeting Intelligence - Prompt Templates")
    print("=" * 50)
    
    # List available prompts
    available_prompts = PromptFactory
