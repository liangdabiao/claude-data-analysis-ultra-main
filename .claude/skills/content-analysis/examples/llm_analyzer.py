"""
LLM-Enhanced Content Analysis

This module provides LLM-powered content analysis capabilities for
advanced text understanding, sentiment analysis, and topic extraction
using models like OpenAI GPT and Qwen (通义千问).
"""

import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Optional, Union, Tuple
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class LLMAnalyzer:
    """
    LLM-powered content analysis toolkit.

    Provides advanced text analysis capabilities using Large Language Models
    for nuanced understanding, emotion detection, and semantic analysis.
    """

    def __init__(self, provider: str = 'openai', api_key: str = None, model: str = None):
        """
        Initialize the LLMAnalyzer.

        Args:
            provider: LLM provider ('openai', 'qwen', 'local')
            api_key: API key for the LLM service
            model: Model name to use
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.rate_limit_delay = 1.0  # Delay between API calls
        self.max_retries = 3

        # Configure default models
        if provider == 'openai':
            self.model = model or 'gpt-3.5-turbo'
            self.api_base = 'https://api.openai.com/v1'
            self.headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
        elif provider == 'qwen':
            self.model = model or 'qwen-turbo'
            self.api_base = 'https://dashscope.aliyuncs.com/api/v1'
            self.headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
        else:
            raise ValueError("Provider must be 'openai' or 'qwen'")

    def _make_api_request(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Make API request to the LLM service.

        Args:
            messages: List of messages for the conversation
            **kwargs: Additional parameters

        Returns:
            API response dictionary
        """
        for attempt in range(self.max_retries):
            try:
                payload = {
                    'model': self.model,
                    'messages': messages,
                    **kwargs
                }

                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                    continue
                else:
                    response.raise_for_status()

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.rate_limit_delay * (2 ** attempt))

        return {}

    def _extract_content_from_response(self, response: Dict) -> str:
        """Extract content from LLM response."""
        try:
            if self.provider == 'openai':
                return response['choices'][0]['message']['content']
            elif self.provider == 'qwen':
                return response['output']['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return ""

    def analyze_sentiment_llm(self, text: str,
                             context: str = None,
                             detailed: bool = False) -> Dict:
        """
        Analyze sentiment using LLM for nuanced understanding.

        Args:
            text: Text to analyze
            context: Additional context for analysis
            detailed: Whether to return detailed analysis

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'emotions': [],
                'explanation': 'No text provided'
            }

        # Build prompt
        if detailed:
            prompt = f"""
Analyze the sentiment of the following text in detail. Consider:
1. Overall sentiment (positive, negative, neutral)
2. Specific emotions present (joy, anger, fear, sadness, surprise, etc.)
3. Confidence level in your analysis
4. Brief explanation for your assessment

Text: {text}

Context: {context or 'No additional context provided'}

Please respond in JSON format:
{{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "emotions": ["emotion1", "emotion2", ...],
    "explanation": "Brief explanation"
}}
"""
        else:
            prompt = f"""
Analyze the sentiment of this text and respond with only "positive", "negative", or "neutral":

Text: {text}

Sentiment:"""

        messages = [
            {"role": "system", "content": "You are a sentiment analysis expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.1)
            content = self._extract_content_from_response(response)

            if detailed:
                try:
                    # Parse JSON response
                    result = json.loads(content.strip())
                    return result
                except json.JSONDecodeError:
                    # Fallback parsing
                    return {
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'emotions': [],
                        'explanation': content
                    }
            else:
                # Simple sentiment classification
                sentiment = content.strip().lower()
                if sentiment in ['positive', 'negative', 'neutral']:
                    return {
                        'sentiment': sentiment,
                        'confidence': 0.8,  # Default confidence
                        'text': text
                    }
                else:
                    return {
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'text': text
                    }

        except Exception as e:
            print(f"Error in LLM sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e),
                'text': text
            }

    def analyze_batch_sentiment_llm(self, texts: List[str],
                                  context: str = None,
                                  detailed: bool = False,
                                  max_workers: int = 3) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts using LLM.

        Args:
            texts: List of texts to analyze
            context: Additional context for analysis
            detailed: Whether to return detailed analysis
            max_workers: Maximum concurrent API calls

        Returns:
            List of sentiment analysis results
        """
        results = []

        # Use thread pool for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(self.analyze_sentiment_llm, text, context, detailed): text
                for text in texts
            }

            # Collect results as they complete
            for future in as_completed(future_to_text):
                text = future_to_text[future]
                try:
                    result = future.result()
                    result['text'] = text
                    results.append(result)

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    results.append({
                        'text': text,
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'error': str(e)
                    })

        return results

    def extract_topics_llm(self, texts: List[str],
                          topics_per_text: int = 3,
                          context: str = None) -> Dict:
        """
        Extract topics using LLM for semantic understanding.

        Args:
            texts: List of texts to analyze
            topics_per_text: Number of topics to extract per text
            context: Additional context for analysis

        Returns:
            Dictionary with topic extraction results
        """
        if not texts:
            return {'topics': [], 'text_topics': {}}

        # Build prompt
        prompt = f"""
Analyze the following texts and extract the main topics and themes.
For each text, identify the top {topics_per_text} most relevant topics.
Consider the semantic meaning and context.

Context: {context or 'No additional context provided'}

Texts to analyze:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Please respond in JSON format:
{{
    "overall_topics": ["topic1", "topic2", ...],
    "text_topics": {{
        "text_1": ["topic1", "topic2", ...],
        "text_2": ["topic1", "topic2", ...],
        ...
    }},
    "topic_descriptions": {{
        "topic1": "description of topic1",
        ...
    }}
}}
"""

        messages = [
            {"role": "system", "content": "You are a topic analysis expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.3)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())

                # Map text indices to original texts
                if 'text_topics' in result:
                    mapped_text_topics = {}
                    for i, text in enumerate(texts):
                        text_key = f"text_{i+1}"
                        if text_key in result['text_topics']:
                            mapped_text_topics[i] = result['text_topics'][text_key]

                    result['text_topics'] = mapped_text_topics

                return result

            except json.JSONDecodeError:
                # Fallback - try simple parsing
                return {
                    'overall_topics': [],
                    'text_topics': {},
                    'raw_response': content
                }

        except Exception as e:
            print(f"Error in LLM topic extraction: {e}")
            return {'error': str(e), 'overall_topics': [], 'text_topics': {}}

    def classify_content_llm(self, texts: List[str],
                           categories: List[str],
                           context: str = None) -> Dict:
        """
        Classify content into predefined categories using LLM.

        Args:
            texts: List of texts to classify
            categories: List of possible categories
            context: Additional context for classification

        Returns:
            Dictionary with classification results
        """
        if not texts or not categories:
            return {'classifications': [], 'uncategorized': []}

        # Build prompt
        categories_str = ", ".join(categories)
        prompt = f"""
Classify the following texts into one of these categories: {categories_str}

Context: {context or 'No additional context provided'}

Texts to classify:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Please respond in JSON format:
{{
    "classifications": [
        {{"text_index": 1, "category": "category_name", "confidence": 0.95, "reasoning": "brief explanation"}},
        ...
    ]
}}
"""

        messages = [
            {"role": "system", "content": "You are a content classification expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.1)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return result

            except json.JSONDecodeError:
                # Fallback - return empty classifications
                return {'classifications': [], 'raw_response': content}

        except Exception as e:
            print(f"Error in LLM content classification: {e}")
            return {'error': str(e), 'classifications': []}

    def summarize_text_llm(self, text: str,
                         max_length: int = 150,
                         style: str = 'neutral') -> str:
        """
        Generate a summary of text using LLM.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Summary style ('neutral', 'formal', 'casual')

        Returns:
            Generated summary text
        """
        if not text or not text.strip():
            return ""

        style_instructions = {
            'neutral': "Provide a balanced, objective summary.",
            'formal': "Provide a professional, formal summary.",
            'casual': "Provide a friendly, conversational summary."
        }

        prompt = f"""
Summarize the following text in no more than {max_length} words.
{style_instructions.get(style, style_instructions['neutral'])}

Text: {text}

Summary:"""

        messages = [
            {"role": "system", "content": "You are a text summarization expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.3)
            return self._extract_content_from_response(response).strip()

        except Exception as e:
            print(f"Error in LLM text summarization: {e}")
            return ""

    def analyze_content_quality_llm(self, text: str,
                                  criteria: List[str] = None) -> Dict:
        """
        Analyze content quality using LLM.

        Args:
            text: Text to analyze
            criteria: Quality criteria to evaluate

        Returns:
            Dictionary with quality analysis results
        """
        if not text or not text.strip():
            return {'overall_score': 0.0, 'criteria_scores': {}, 'suggestions': []}

        if not criteria:
            criteria = ['clarity', 'engagement', 'accuracy', 'relevance', 'structure']

        criteria_str = ", ".join(criteria)
        prompt = f"""
Analyze the quality of the following text based on these criteria: {criteria_str}

For each criterion, provide a score from 0.0 to 1.0 and brief feedback.
Also provide an overall quality score and suggestions for improvement.

Text: {text}

Please respond in JSON format:
{{
    "overall_score": 0.0-1.0,
    "criteria_scores": {{
        "criterion1": {{"score": 0.0-1.0, "feedback": "brief feedback"}},
        ...
    }},
    "suggestions": ["suggestion1", "suggestion2", ...]
}}
"""

        messages = [
            {"role": "system", "content": "You are a content quality analysis expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.2)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return result

            except json.JSONDecodeError:
                # Fallback - return basic structure
                return {
                    'overall_score': 0.5,
                    'criteria_scores': {},
                    'raw_response': content
                }

        except Exception as e:
            print(f"Error in LLM quality analysis: {e}")
            return {'error': str(e), 'overall_score': 0.0}

    def generate_content_insights_llm(self, texts: List[str],
                                    analysis_type: str = 'general') -> Dict:
        """
        Generate comprehensive content insights using LLM.

        Args:
            texts: List of texts to analyze
            analysis_type: Type of analysis ('general', 'marketing', 'sentiment', 'engagement')

        Returns:
            Dictionary with insights and recommendations
        """
        if not texts:
            return {'insights': [], 'recommendations': [], 'patterns': []}

        # Build analysis-specific prompt
        analysis_prompts = {
            'general': "Analyze these texts and provide general insights about content themes, patterns, and characteristics.",
            'marketing': "Analyze these marketing texts and provide insights about messaging effectiveness, audience appeal, and optimization opportunities.",
            'sentiment': "Analyze these texts focusing on emotional patterns, sentiment trends, and psychological insights.",
            'engagement': "Analyze these texts for engagement potential, viral characteristics, and audience interaction factors."
        }

        prompt = f"""
{analysis_prompts.get(analysis_type, analysis_prompts['general'])}

Provide:
1. Key insights about the content
2. Notable patterns and trends
3. Specific recommendations for improvement
4. Target audience characteristics

Texts to analyze:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Please respond in JSON format:
{{
    "insights": ["insight1", "insight2", ...],
    "patterns": ["pattern1", "pattern2", ...],
    "recommendations": ["recommendation1", "recommendation2", ...],
    "target_audience": ["characteristic1", "characteristic2", ...]
}}
"""

        messages = [
            {"role": "system", "content": "You are a content analysis and strategy expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.3)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return result

            except json.JSONDecodeError:
                # Fallback - try simple parsing
                return {
                    'insights': [],
                    'patterns': [],
                    'recommendations': [],
                    'raw_response': content
                }

        except Exception as e:
            print(f"Error in LLM insights generation: {e}")
            return {'error': str(e), 'insights': [], 'recommendations': []}

    def detect_viral_potential_llm(self, text: str) -> Dict:
        """
        Detect viral potential of content using LLM.

        Args:
            text: Text to analyze for viral potential

        Returns:
            Dictionary with viral potential analysis
        """
        if not text or not text.strip():
            return {'viral_score': 0.0, 'factors': [], 'recommendations': []}

        prompt = f"""
Analyze this text for viral potential. Consider factors like:
- Emotional appeal
- Shareability
- Novelty
- Controversy
- Practical value
- Entertainment value

Provide a viral potential score (0.0-1.0) and specific factors.

Text: {text}

Please respond in JSON format:
{{
    "viral_score": 0.0-1.0,
    "factors": [
        {{"factor": "emotional_appeal", "score": 0.0-1.0, "explanation": "brief explanation"}},
        ...
    ],
    "recommendations": ["recommendation1", "recommendation2", ...]
}}
"""

        messages = [
            {"role": "system", "content": "You are a viral content analysis expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.3)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return result

            except json.JSONDecodeError:
                # Fallback - return basic structure
                return {
                    'viral_score': 0.5,
                    'factors': [],
                    'raw_response': content
                }

        except Exception as e:
            print(f"Error in viral potential detection: {e}")
            return {'error': str(e), 'viral_score': 0.0}

    def translate_content_llm(self, text: str,
                            target_language: str,
                            preserve_style: bool = True) -> Dict:
        """
        Translate content using LLM.

        Args:
            text: Text to translate
            target_language: Target language for translation
            preserve_style: Whether to preserve original style

        Returns:
            Dictionary with translation results
        """
        if not text or not text.strip():
            return {'translated_text': '', 'confidence': 0.0}

        style_instruction = " and preserve the original tone and style" if preserve_style else ""
        prompt = f"""
Translate the following text to {target_language}{style_instruction}.

Text: {text}

Translation:"""

        messages = [
            {"role": "system", "content": f"You are a professional translator specializing in translation to {target_language}."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.1)
            translated_text = self._extract_content_from_response(response).strip()

            return {
                'translated_text': translated_text,
                'original_text': text,
                'target_language': target_language,
                'confidence': 0.9  # Default confidence
            }

        except Exception as e:
            print(f"Error in LLM translation: {e}")
            return {
                'translated_text': '',
                'error': str(e),
                'original_text': text
            }

    def analyze_content_llm_comprehensive(self, text: str,
                                        context: str = None) -> Dict:
        """
        Perform comprehensive content analysis using LLM.

        Args:
            text: Text to analyze
            context: Additional context for analysis

        Returns:
            Dictionary with comprehensive analysis results
        """
        if not text or not text.strip():
            return {}

        prompt = f"""
Perform a comprehensive analysis of this text. Include:
1. Sentiment analysis with emotion detection
2. Key topics and themes
3. Content quality assessment
4. Audience identification
5. Engagement potential
6. Improvement recommendations

Context: {context or 'No additional context provided'}

Text: {text}

Please respond in JSON format:
{{
    "sentiment": {{
        "overall": "positive/negative/neutral",
        "confidence": 0.0-1.0,
        "emotions": ["emotion1", "emotion2", ...]
    }},
    "topics": ["topic1", "topic2", ...],
    "quality": {{
        "score": 0.0-1.0,
        "clarity": 0.0-1.0,
        "engagement": 0.0-1.0
    }},
    "audience": ["characteristic1", "characteristic2", ...],
    "engagement_potential": {{
        "score": 0.0-1.0,
        "factors": ["factor1", "factor2", ...]
    }},
    "recommendations": ["recommendation1", "recommendation2", ...]
}}
"""

        messages = [
            {"role": "system", "content": "You are a comprehensive content analysis expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_request(messages, temperature=0.3)
            content = self._extract_content_from_response(response)

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return result

            except json.JSONDecodeError:
                # Fallback - perform individual analyses
                return {
                    'sentiment': self.analyze_sentiment_llm(text, context, detailed=True),
                    'topics': self.extract_topics_llm([text], context=context).get('overall_topics', []),
                    'raw_response': content
                }

        except Exception as e:
            print(f"Error in comprehensive LLM analysis: {e}")
            return {'error': str(e)}