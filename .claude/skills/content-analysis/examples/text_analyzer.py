"""
Text Analysis Core Functions

This module provides comprehensive text preprocessing and analysis utilities
for content analysis tasks including cleaning, tokenization, and basic NLP operations.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Download required NLTK data (only if not already downloaded)
def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

class TextAnalyzer:
    """
    Comprehensive text analysis toolkit.

    Provides text preprocessing, cleaning, tokenization, and basic NLP analysis
    for content analysis tasks.
    """

    def __init__(self, language='english'):
        """
        Initialize the TextAnalyzer.

        Args:
            language: Language for stopwords and processing (default: 'english')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()

        # Load stopwords for specified language
        try:
            self.stop_words = set(stopwords.words(language))
        except OSError:
            # Fallback to English if language not found
            self.stop_words = set(stopwords.words('english'))

        # Add custom stopwords
        custom_stopwords = {'http', 'https', 'www', 'com', 'org', 'net'}
        self.stop_words.update(custom_stopwords)

    def clean_text(self, text: str,
                   remove_urls: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = False,
                   remove_numbers: bool = False,
                   remove_punctuation: bool = True,
                   lowercase: bool = True) -> str:
        """
        Clean text by removing unwanted elements.

        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags (keep text without #)
            remove_numbers: Whether to remove numbers
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove mentions (@username)
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)

        # Remove hashtags (keep text, remove #)
        if remove_hashtags:
            text = re.sub(r'#(\w+)', r'\1', text)

        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Convert to lowercase
        if lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text: str,
                     remove_stopwords: bool = True,
                     lemmatize: bool = True) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text to tokenize
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens

        Returns:
            List of tokens
        """
        # Clean text first
        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Remove single character tokens
        tokens = [token for token in tokens if len(token) > 1]

        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_keywords(self, text: str,
                        top_k: int = 10,
                        method: str = 'frequency') -> List[Tuple[str, float]]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            top_k: Number of top keywords to return
            method: Keyword extraction method ('frequency' or 'tfidf')

        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []

        # Tokenize text
        tokens = self.tokenize_text(text)

        if not tokens:
            return []

        if method == 'frequency':
            # Simple frequency counting
            word_freq = Counter(tokens)
            return word_freq.most_common(top_k)

        elif method == 'tfidf':
            # TF-IDF scoring
            try:
                tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = tfidf.fit_transform([text])

                # Get feature names and scores
                feature_names = tfidf.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]

                # Create keyword-score pairs
                keywords = [(feature_names[i], tfidf_scores[i])
                           for i in range(len(feature_names))
                           if tfidf_scores[i] > 0]

                # Sort by score and return top k
                keywords.sort(key=lambda x: x[1], reverse=True)
                return keywords[:top_k]

            except Exception:
                # Fallback to frequency method
                word_freq = Counter(tokens)
                return word_freq.most_common(top_k)

        else:
            raise ValueError("Method must be 'frequency' or 'tfidf'")

    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate basic readability metrics.

        Args:
            text: Input text

        Returns:
            Dictionary with readability metrics
        """
        if not text:
            return {}

        # Clean text for analysis
        cleaned_text = self.clean_text(text, remove_punctuation=False)

        # Split into sentences and words
        sentences = sent_tokenize(cleaned_text)
        words = word_tokenize(cleaned_text)

        # Filter out punctuation
        words = [word for word in words if word.isalpha()]

        if not sentences or not words:
            return {}

        # Calculate metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word) for word in words)

        # Average values
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words

        # Flesch Reading Ease (simplified version)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        return {
            'num_sentences': num_sentences,
            'num_words': num_words,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word,
            'flesch_reading_ease': max(0, min(100, flesch_score)),  # Clamp to 0-100
            'reading_level': self._get_reading_level(flesch_score)
        }

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting (approximate)."""
        vowels = "aeiouy"
        word = word.lower()
        syllable_count = 0
        prev_char_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel

        # Handle silent 'e' at the end
        if word.endswith('e'):
            syllable_count -= 1

        return max(1, syllable_count)

    def _get_reading_level(self, flesch_score: float) -> str:
        """Get reading level from Flesch score."""
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

    def cluster_texts(self, texts: List[str],
                     num_clusters: int = 5,
                     method: str = 'kmeans') -> Dict:
        """
        Cluster texts by similarity.

        Args:
            texts: List of texts to cluster
            num_clusters: Number of clusters to create
            method: Clustering method ('kmeans')

        Returns:
            Dictionary with clustering results
        """
        if not texts or len(texts) < num_clusters:
            return {}

        # Preprocess and vectorize texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text]  # Remove empty texts

        if len(cleaned_texts) < num_clusters:
            return {}

        try:
            # TF-IDF vectorization
            tfidf = TfidfVectorizer(max_features=1000,
                                   stop_words='english',
                                   ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(cleaned_texts)

            if method == 'kmeans':
                # K-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)

                # Get cluster centers and feature names
                feature_names = tfidf.get_feature_names_out()
                cluster_centers = kmeans.cluster_centers_

                # Extract top terms for each cluster
                cluster_terms = {}
                for i in range(num_clusters):
                    top_indices = cluster_centers[i].argsort()[-10:][::-1]
                    cluster_terms[i] = [feature_names[idx] for idx in top_indices]

                return {
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_terms': cluster_terms,
                    'num_clusters': num_clusters,
                    'feature_names': feature_names.tolist()
                }

        except Exception as e:
            print(f"Error in clustering: {e}")
            return {}

    def find_similar_texts(self, query_text: str,
                         text_corpus: List[str],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar texts to a query text.

        Args:
            query_text: Query text
            text_corpus: List of texts to search through
            top_k: Number of most similar texts to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if not query_text or not text_corpus:
            return []

        # Prepare texts
        all_texts = [query_text] + text_corpus
        cleaned_texts = [self.clean_text(text) for text in all_texts]
        cleaned_texts = [text for text in cleaned_texts if text]

        if len(cleaned_texts) < 2:
            return []

        try:
            # TF-IDF vectorization
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(cleaned_texts)

            # Calculate cosine similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Get top k most similar texts
            top_indices = similarities.argsort()[-top_k:][::-1]

            return [(int(idx), float(similarities[idx])) for idx in top_indices]

        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return []

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract basic entities from text (simplified version).

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        if not text:
            return {}

        # Clean and tokenize
        cleaned_text = self.clean_text(text, remove_numbers=False)
        tokens = word_tokenize(cleaned_text)

        # Simple pattern-based entity extraction
        entities = {
            'emails': [],
            'phone_numbers': [],
            'hashtags': [],
            'mentions': [],
            'urls': [],
            'numbers': []
        }

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)

        # Phone number pattern (simplified)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities['phone_numbers'] = re.findall(phone_pattern, text)

        # Hashtag pattern
        hashtag_pattern = r'#\w+'
        entities['hashtags'] = re.findall(hashtag_pattern, text)

        # Mention pattern
        mention_pattern = r'@\w+'
        entities['mentions'] = re.findall(mention_pattern, text)

        # URL pattern
        url_pattern = r'https?://\S+|www\.\S+'
        entities['urls'] = re.findall(url_pattern, text)

        # Number pattern
        number_pattern = r'\b\d+\.?\d*\b'
        entities['numbers'] = re.findall(number_pattern, text)

        return entities

    def get_text_statistics(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive text statistics.

        Args:
            text: Input text

        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {}

        # Basic counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))

        # Token statistics
        tokens = self.tokenize_text(text)
        unique_words = len(set(tokens))

        # Average lengths
        avg_word_length = np.mean([len(word) for word in word_tokenize(text)]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'unique_word_count': unique_words,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0
        }

class SocialMediaAnalyzer(TextAnalyzer):
    """
    Specialized analyzer for social media content.
    """

    def __init__(self):
        super().__init__()
        # Add social media specific stopwords
        social_stopwords = {'rt', 'retweet', 'like', 'follow', 'follower', 'share'}
        self.stop_words.update(social_stopwords)

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from social media text."""
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text)
        return [tag[1:] for tag in hashtags]  # Remove # symbol

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from social media text."""
        mention_pattern = r'@\w+'
        return re.findall(mention_pattern, text)

    def count_engagement_indicators(self, text: str) -> Dict[str, int]:
        """Count engagement indicators in social media text."""
        indicators = {
            'emojis': len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)),
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'uppercase_words': len([word for word in text.split() if word.isupper()]),
            'mentions': len(self.extract_mentions(text)),
            'hashtags': len(self.extract_hashtags(text))
        }
        return indicators

    def detect_social_media_platform(self, text: str) -> str:
        """Detect which social media platform the text is from."""
        text_lower = text.lower()

        if 'rt @' in text_lower or 'retweet' in text_lower:
            return 'Twitter'
        elif 'instagram.com' in text_lower or 'insta' in text_lower:
            return 'Instagram'
        elif 'facebook.com' in text_lower or 'fb.com' in text_lower:
            return 'Facebook'
        elif 'linkedin.com' in text_lower:
            return 'LinkedIn'
        elif 'tiktok.com' in text_lower or 'tiktok' in text_lower:
            return 'TikTok'
        else:
            return 'Unknown'