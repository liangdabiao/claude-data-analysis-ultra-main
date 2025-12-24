"""
评论内容分析 (Content Analysis)
分析产品评论的情感、评分模式和主题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import Counter
import re
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""

    # 转换为字符串
    text = str(text)

    # 转换为小写
    text = text.lower()

    # 移除特殊字符但保留空格
    text = re.sub(r'[^\w\s]', ' ', text)

    # 移除多余空格
    text = ' '.join(text.split())

    return text

def analyze_keywords(texts, n=20):
    """分析关键词"""
    all_words = []
    for text in texts:
        if text and len(text) > 2:
            words = text.split()
            # 过滤短词和常见词
            words = [w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had']]
            all_words.extend(words)

    word_freq = Counter(all_words)
    return word_freq.most_common(n)

def main():
    output_dir = 'do_more_analysis/skill_execution/content-analysis'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Review Content Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    reviews = pd.read_csv('data_storage/Reviews.csv',
                         parse_dates=['review_creation_date', 'review_answer_timestamp'])

    print(f"Total reviews: {len(reviews):,}")

    # Basic statistics
    print("\nBasic Statistics:")
    print(f"  Unique products: {reviews['order_id'].nunique():,}")
    print(f"  Review score range: {reviews['review_score'].min()} - {reviews['review_score'].max()}")

    # Sentiment based on scores
    reviews['sentiment'] = pd.cut(reviews['review_score'],
                                   bins=[0, 2, 3, 5],
                                   labels=['Negative', 'Neutral', 'Positive'])

    sentiment_counts = reviews['sentiment'].value_counts()
    print(f"\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count:,} ({count/len(reviews)*100:.1f}%)")

    # Analyze review comments
    print("\nAnalyzing review comments...")

    # Clean comment messages
    reviews['clean_comment'] = reviews['review_comment_message'].apply(clean_text)
    reviews['clean_title'] = reviews['review_comment_title'].apply(clean_text)

    # Combine title and message for full text analysis
    reviews['full_text'] = (reviews['clean_title'] + ' ' + reviews['clean_comment']).str.strip()

    # Text analysis
    comments_with_text = reviews[reviews['full_text'].str.len() > 0]
    print(f"  Reviews with comments: {len(comments_with_text):,} ({len(comments_with_text)/len(reviews)*100:.1f}%)")

    if len(comments_with_text) > 0:
        # Overall keywords
        all_keywords = analyze_keywords(comments_with_text['full_text'], n=30)

        # Keywords by sentiment
        positive_reviews = comments_with_text[comments_with_text['sentiment'] == 'Positive']
        negative_reviews = comments_with_text[comments_with_text['sentiment'] == 'Negative']

        positive_keywords = analyze_keywords(positive_reviews['full_text'], n=20) if len(positive_reviews) > 0 else []
        negative_keywords = analyze_keywords(negative_reviews['full_text'], n=20) if len(negative_reviews) > 0 else []

        print(f"\n  Total keywords extracted: {len(all_keywords)}")
        print(f"  Positive review keywords: {len(positive_keywords)}")
        print(f"  Negative review keywords: {len(negative_keywords)}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Score distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Score distribution
    score_counts = reviews['review_score'].value_counts().sort_index()
    axes[0, 0].bar(score_counts.index, score_counts.values,
                   color=['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71'],
                   alpha=0.8, edgecolor='black')
    axes[0, 0].set_xlabel('Review Score', fontsize=12)
    axes[0, 0].set_ylabel('Number of Reviews', fontsize=12)
    axes[0, 0].set_title('Review Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(score_counts.values):
        axes[0, 0].text(score_counts.index[i], v + max(score_counts.values)*0.01,
                       f'{v:,}', ha='center', fontweight='bold')

    # Sentiment distribution (pie)
    sentiment_colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
    axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=[sentiment_colors[s] for s in sentiment_counts.index],
                   startangle=90)
    axes[0, 1].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

    # Review length distribution
    review_lengths = comments_with_text['full_text'].str.len()
    axes[1, 0].hist(review_lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(review_lengths.mean(), color='red', linestyle='--', linewidth=2,
                      label=f"Mean: {review_lengths.mean():.0f} chars")
    axes[1, 0].set_xlabel('Comment Length (characters)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Review Comment Length Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Score vs Comment Length
    score_length_data = reviews.copy()
    score_length_data['comment_length'] = score_length_data['full_text'].str.len().fillna(0)

    score_length_box = []
    for score in sorted(score_length_data['review_score'].unique()):
        data = score_length_data[score_length_data['review_score'] == score]['comment_length']
        data = data[data > 0]  # Only reviews with comments
        score_length_box.append(data.values)

    bp = axes[1, 1].boxplot(score_length_box, labels=sorted(score_length_data['review_score'].unique()),
                             patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1, 1].set_xlabel('Review Score', fontsize=12)
    axes[1, 1].set_ylabel('Comment Length (characters)', fontsize=12)
    axes[1, 1].set_title('Comment Length by Score', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'review_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] review_distributions.png")

    # 2. Keyword analysis
    if len(comments_with_text) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Top keywords overall
        if all_keywords:
            words, counts = zip(*all_keywords[:15])
            axes[0].barh(range(len(words)), counts, color='steelblue', alpha=0.8)
            axes[0].set_yticks(range(len(words)))
            axes[0].set_yticklabels(words)
            axes[0].set_xlabel('Frequency', fontsize=12)
            axes[0].set_title('Top 15 Keywords (All Reviews)', fontsize=14, fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(axis='x', alpha=0.3)

        # Positive keywords
        if positive_keywords:
            words, counts = zip(*positive_keywords[:10])
            axes[1].barh(range(len(words)), counts, color='#2ecc71', alpha=0.8)
            axes[1].set_yticks(range(len(words)))
            axes[1].set_yticklabels(words)
            axes[1].set_xlabel('Frequency', fontsize=12)
            axes[1].set_title('Top 10 Keywords (Positive Reviews)', fontsize=14, fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(axis='x', alpha=0.3)

        # Negative keywords
        if negative_keywords:
            words, counts = zip(*negative_keywords[:10])
            axes[2].barh(range(len(words)), counts, color='#e74c3c', alpha=0.8)
            axes[2].set_yticks(range(len(words)))
            axes[2].set_yticklabels(words)
            axes[2].set_xlabel('Frequency', fontsize=12)
            axes[2].set_title('Top 10 Keywords (Negative Reviews)', fontsize=14, fontweight='bold')
            axes[2].invert_yaxis()
            axes[2].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_analysis.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("  [OK] keyword_analysis.png")

    # 3. Word cloud (skip - wordcloud module not available)
    print("  [SKIP] wordcloud (module not available)")

    # 4. Sentiment dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Score distribution by percentage
    ax1 = fig.add_subplot(gs[0, 0])
    score_pct = (reviews['review_score'].value_counts(normalize=True) * 100).sort_index()
    colors = ['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']
    ax1.bar(score_pct.index, score_pct.values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Review Score', fontsize=11)
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_title('Review Score Percentage', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate(score_pct.values):
        ax1.text(score_pct.index[i], v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    # Response time analysis
    ax2 = fig.add_subplot(gs[0, 1])
    reviews['response_time_days'] = (
        reviews['review_answer_timestamp'] - reviews['review_creation_date']
    ).dt.total_seconds() / 86400

    response_time = reviews[reviews['response_time_days'].notna()]['response_time_days']
    ax2.hist(response_time, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(response_time.mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {response_time.mean():.1f} days")
    ax2.set_xlabel('Response Time (days)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Seller Response Time Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Comment rate by score
    ax3 = fig.add_subplot(gs[1, :])
    has_comment_by_score = []

    for score in sorted(reviews['review_score'].unique()):
        score_reviews = reviews[reviews['review_score'] == score]
        comment_rate = (score_reviews['full_text'].str.len() > 0).sum() / len(score_reviews) * 100
        has_comment_by_score.append(comment_rate)

    bars = ax3.bar(sorted(reviews['review_score'].unique()), has_comment_by_score,
                   color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Review Score', fontsize=11)
    ax3.set_ylabel('Comment Rate (%)', fontsize=11)
    ax3.set_title('Comment Rate by Review Score', fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, has_comment_by_score):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', fontweight='bold')

    plt.savefig(os.path.join(output_dir, 'sentiment_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] sentiment_dashboard.png")

    # Export results
    print("\nExporting results...")

    # Reviews with sentiment
    reviews_export = reviews[['review_id', 'order_id', 'review_score', 'sentiment',
                              'review_comment_title', 'review_comment_message',
                              'full_text', 'review_creation_date']].copy()
    reviews_export.to_csv(os.path.join(output_dir, 'reviews_with_sentiment.csv'),
                         index=False, encoding='utf-8-sig')
    print("  [OK] reviews_with_sentiment.csv")

    # JSON summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_reviews': int(len(reviews)),
        'avg_score': float(reviews['review_score'].mean()),
        'sentiment_distribution': {
            'positive': int(sentiment_counts.get('Positive', 0)),
            'neutral': int(sentiment_counts.get('Neutral', 0)),
            'negative': int(sentiment_counts.get('Negative', 0))
        },
        'reviews_with_comments': int(len(comments_with_text)),
        'comment_rate': float(len(comments_with_text) / len(reviews) * 100),
        'avg_response_time_days': float(response_time.mean()) if len(response_time) > 0 else None,
        'top_keywords': all_keywords[:10] if len(comments_with_text) > 0 else []
    }

    with open(os.path.join(output_dir, 'content_analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  [OK] content_analysis_summary.json")

    # Markdown report
    report = f"""# Review Content Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the analysis of customer reviews for the e-commerce platform.

### Key Metrics

- **Total Reviews**: {len(reviews):,}
- **Average Review Score**: {reviews['review_score'].mean():.2f}/5.0
- **Reviews with Comments**: {len(comments_with_text):,} ({len(comments_with_text)/len(reviews)*100:.1f}%)
- **Average Response Time**: {response_time.mean():.1f} days

## Review Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
"""

    for score in sorted(reviews['review_score'].unique()):
        count = (reviews['review_score'] == score).sum()
        pct = count / len(reviews) * 100
        bar = '█' * int(pct / 2)
        report += f"| {score} | {count:,} | {pct:.1f}% {bar} |\n"

    report += f"""

## Sentiment Analysis

### Sentiment Distribution

- **Positive** (4-5 stars): {sentiment_counts.get('Positive', 0):,} ({sentiment_counts.get('Positive', 0)/len(reviews)*100:.1f}%)
- **Neutral** (3 stars): {sentiment_counts.get('Neutral', 0):,} ({sentiment_counts.get('Neutral', 0)/len(reviews)*100:.1f}%)
- **Negative** (1-2 stars): {sentiment_counts.get('Negative', 0):,} ({sentiment_counts.get('Negative', 0)/len(reviews)*100:.1f}%)

### Key Findings

1. **Overall Satisfaction**
   - Average score of {reviews['review_score'].mean():.2f} indicates {"high" if reviews['review_score'].mean() >= 4 else "moderate" if reviews['review_score'].mean() >= 3 else "low"} customer satisfaction
   - {sentiment_counts.get('Positive', 0)/len(reviews)*100:.1f}% of reviews are positive (4-5 stars)

2. **Customer Engagement**
   - {len(comments_with_text)/len(reviews)*100:.1f}% of reviews include written comments
   - Higher scores correlate with {"higher" if has_comment_by_score[4] > has_comment_by_score[0] else "lower"} comment rates
   - Average seller response time: {response_time.mean():.1f} days

3. **Areas for Improvement**
   - Focus on addressing {sentiment_counts.get('Negative', 0)/len(reviews)*100:.1f}% negative reviews
   - Investigate common issues in 1-2 star reviews
   - Improve response time to customer concerns

"""

    if all_keywords:
        report += "## Top Keywords\n\n"
        report += "### Most Frequent Terms (All Reviews)\n\n"
        for word, count in all_keywords[:10]:
            report += f"- **{word}**: {count} mentions\n"

        report += "\n"

        if positive_keywords:
            report += "### Positive Review Keywords\n\n"
            for word, count in positive_keywords[:5]:
                report += f"- **{word}**: {count} mentions\n"
            report += "\n"

        if negative_keywords:
            report += "### Negative Review Keywords\n\n"
            for word, count in negative_keywords[:5]:
                report += f"- **{word}**: {count} mentions\n"
            report += "\n"

    report += f"""
## Recommendations

### Customer Experience
1. **Address Negative Feedback**
   - Respond promptly to negative reviews
   - Investigate common pain points
   - Implement corrective actions

2. **Encourage Positive Reviews**
   - Request reviews from satisfied customers
   - Make review process easy and accessible
   - Incentivize detailed feedback

3. **Improve Response Time**
   - Current average: {response_time.mean():.1f} days
   - Aim to respond within 24-48 hours
   - Implement automated acknowledgments

### Product Quality
1. **Monitor Common Issues**
   - Track keyword trends in negative reviews
   - Identify product categories with low scores
   - Prioritize quality improvements

2. **Leverage Positive Feedback**
   - Highlight positive reviews in marketing
   - Showcase customer testimonials
   - Use positive keywords in product descriptions

### Review Management
1. **Implement Review Monitoring**
   - Set up alerts for negative reviews
   - Track review score trends over time
   - Monitor competitor reviews

2. **Enhance Customer Communication**
   - Follow up on negative experiences
   - Thank customers for positive reviews
   - Provide solutions to reported issues

## Files Generated

- reviews_with_sentiment.csv: Complete reviews with sentiment labels
- content_analysis_summary.json: Analysis summary in JSON format
- review_distributions.png: Score and sentiment distributions
- keyword_analysis.png: Top keywords by sentiment
- sentiment_dashboard.png: Comprehensive sentiment dashboard
"""

    if len(comments_with_text) > 50:
        report += "- wordcloud.png: Word cloud of review comments\n"

    report += """
---

*This analysis provides insights for improving customer satisfaction and product quality.*
"""

    with open(os.path.join(output_dir, 'Content_Analysis_Report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("  [OK] Content_Analysis_Report.md")

    print("\n" + "="*80)
    print("Content Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
