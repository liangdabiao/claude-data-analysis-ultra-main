# Content Analysis Skill

This skill provides comprehensive content analysis capabilities combining traditional NLP techniques with modern LLM-powered insights for deep text understanding and actionable intelligence.

## Overview

The Content Analysis Skill enables sophisticated text analysis across multiple domains including social media, marketing content, product reviews, and video content. It offers both traditional statistical NLP methods and advanced LLM-enhanced analysis for maximum accuracy and insight generation.

## Features

### Core Capabilities
- **Dual-Mode Analysis**: Traditional NLP + LLM-enhanced methods
- **Multi-Language Support**: English, Chinese, and other languages
- **Sentiment Analysis**: From basic polarity to nuanced emotion detection
- **Topic Extraction**: Statistical and semantic topic identification
- **Content Classification**: Automated categorization and clustering
- **Viral Content Detection**: Pattern recognition for trending content

### Analysis Types
1. **Sentiment Intelligence**: Emotion detection, tone analysis, mood tracking
2. **Topic Modeling**: Keyword extraction, semantic themes, trend identification
3. **Content Quality**: Engagement prediction, readability assessment
4. **Competitive Intelligence**: Content comparison, market positioning
5. **Audience Insights**: Demographic analysis, preference modeling

## File Structure

```
content-analysis/
├── SKILL.md                      # Main skill definition
├── README.md                     # This file
├── scripts/                      # Core analysis modules
│   ├── text_analyzer.py          # Text preprocessing utilities
│   ├── sentiment_analyzer.py     # Sentiment analysis tools
│   ├── topic_analyzer.py         # Topic extraction and modeling
│   ├── llm_analyzer.py           # LLM-powered analysis
│   └── content_visualizer.py     # Visualization tools
└── examples/                     # Usage examples
    ├── basic_content_analysis.py # Traditional NLP analysis
    ├── llm_enhanced_analysis.py  # LLM-enhanced analysis
    ├── social_media_analysis.py  # Social media specific analysis
    └── sample_data/              # Example datasets
```

## Getting Started

### Prerequisites

**Basic Requirements:**
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
```

**LLM Enhancement (Optional):**
```bash
pip install openai dashscope requests
```

**NLTK Setup:**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Basic Usage

1. **Prepare your content data** with text fields and optional metadata
2. **Choose analysis mode**: Traditional NLP, LLM-enhanced, or hybrid
3. **Configure analysis parameters**: Language, sentiment model, topic extraction
4. **Run analysis** and generate insights
5. **Visualize results** with interactive charts and reports

### Data Format Requirements

Your data should include:
- **Text Content**: Primary text fields (titles, descriptions, comments, etc.)
- **Optional Metadata**: Author, date, category, engagement metrics
- **Language Information**: Language codes for multi-language analysis

## Examples

### Basic Sentiment Analysis
```python
from scripts.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
results = analyzer.analyze_batch(text_data)
analyzer.visualize_sentiment_distribution(results)
```

### LLM-Enhanced Topic Extraction
```python
from scripts.llm_analyzer import LLMAnalyzer

llm = LLMAnalyzer(provider='openai')
topics = llm.extract_topics(content_list, topics_per_item=5)
summaries = llm.generate_summaries(content_by_category)
```

### Social Media Analysis
```python
from scripts.text_analyzer import SocialMediaAnalyzer

sma = SocialMediaAnalyzer()
trends = sma.analyze_hashtags(social_data)
viral_patterns = sma.identify_viral_content(content_data)
```

## Common Use Cases

### Social Media Management
- **Brand Monitoring**: Track brand sentiment across platforms
- **Campaign Analysis**: Measure campaign effectiveness and engagement
- **Influencer Identification**: Find trending content and creators
- **Crisis Detection**: Early warning for negative sentiment spikes

### Content Marketing
- **SEO Optimization**: Keyword and topic analysis for content strategy
- **Performance Prediction**: Predict content engagement and virality
- **Competitive Analysis**: Compare content performance against competitors
- **Audience Understanding**: Analyze audience preferences and behaviors

### Product and Service Analysis
- **Customer Feedback**: Analyze reviews and support tickets
- **Feature Requests**: Extract product improvement suggestions
- **Market Research**: Identify market trends and opportunities
- **Quality Assurance**: Monitor product mentions and sentiment

### Video Content Analysis
- **YouTube Optimization**: Title and description analysis for discoverability
- **Comment Analysis**: Understand audience engagement and feedback
- **Trend Identification**: Spot emerging video trends and topics
- **Content Strategy**: Data-driven video content planning

## Advanced Analytics

### Sentiment Intelligence
- **Fine-grained Emotions**: Joy, anger, fear, sadness, surprise detection
- **Temporal Sentiment**: Track sentiment changes over time
- **Comparative Analysis**: Compare sentiment across brands or products
- **Context Understanding**: Detect sarcasm, irony, and complex emotions

### Topic Intelligence
- **Emerging Topics**: Identify new and trending topics
- **Topic Evolution**: Track how topics change over time
- **Cross-Domain Topics**: Find topics spanning multiple categories
- **Semantic Clustering**: Group content by meaning rather than keywords

### Content Performance
- **Engagement Prediction**: Forecast likes, shares, comments
- **Virality Factors**: Identify characteristics of viral content
- **Optimization Recommendations**: Suggest content improvements
- **A/B Testing Analysis**: Compare different content variations

## Best Practices

### Data Preparation
- **Text Cleaning**: Remove noise, normalize format, handle special characters
- **Language Detection**: Automatically identify and process different languages
- **Quality Filtering**: Remove low-quality or spam content
- **Metadata Enrichment**: Add context information for better analysis

### Analysis Strategy
- **Multi-Method Approach**: Combine traditional NLP with LLM analysis
- **Sampling Techniques**: Use representative samples for large datasets
- **Validation**: Cross-validate results with human annotation
- **Iterative Refinement**: Continuously improve models and parameters

### Cost and Performance
- **Smart Sampling**: Use LLM analysis for high-value content only
- **Batch Processing**: Optimize API calls and processing efficiency
- **Caching**: Store results for repeated analysis
- **Resource Monitoring**: Track usage and optimize resource allocation

## Integration Options

### API Integration
- **REST APIs**: Connect with content management systems
- **Webhooks**: Real-time content analysis triggers
- **Scheduled Jobs**: Automated periodic analysis
- **Stream Processing**: Real-time content stream analysis

### Platform Integration
- **Social Media APIs**: Twitter, Facebook, Instagram integration
- **Content Management**: WordPress, Drupal, CMS integration
- **Analytics Platforms**: Google Analytics, Adobe Analytics
- **CRM Systems**: Salesforce, HubSpot integration

## Troubleshooting

### Common Issues

1. **Low Analysis Accuracy**
   - Check text preprocessing and cleaning
   - Verify language detection and model selection
   - Ensure sufficient training data for custom models

2. **High API Costs**
   - Optimize sampling strategies
   - Implement caching mechanisms
   - Use traditional NLP for bulk processing

3. **Performance Bottlenecks**
   - Implement parallel processing
   - Use batch operations for API calls
   - Optimize data structures and algorithms

4. **Language Support Issues**
   - Verify language code settings
   - Check model availability for target languages
   - Implement language-specific preprocessing

### Performance Optimization
- **Memory Management**: Process large datasets in chunks
- **CPU Optimization**: Use vectorized operations and parallel processing
- **I/O Optimization**: Minimize disk access and use efficient data formats
- **Network Optimization**: Batch API calls and handle network issues

## Extension Possibilities

- **Multi-modal Analysis**: Analyze text + images + videos together
- **Real-time Processing**: Stream processing for live content analysis
- **Custom Models**: Fine-tune models for specific domains or industries
- **Advanced Visualization**: Interactive dashboards and 3D visualizations
- **Automated Actions**: Trigger automated responses based on analysis results

## Security and Privacy

- **Data Protection**: Ensure compliance with GDPR, CCPA, and other regulations
- **API Security**: Secure API key management and access control
- **Content Filtering**: Remove sensitive or private information
- **Audit Trails**: Track analysis operations and results