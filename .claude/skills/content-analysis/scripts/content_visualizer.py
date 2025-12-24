"""
Content Visualization Tools

This module provides comprehensive visualization capabilities for content analysis
including sentiment plots, topic visualizations, and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import List, Dict, Optional, Tuple, Union
from wordcloud import WordCloud
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

# Set matplotlib parameters for better Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ContentVisualizer:
    """
    Comprehensive content visualization toolkit.

    Provides various visualization types for content analysis results
    including sentiment analysis, topic modeling, and performance metrics.
    """

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the ContentVisualizer.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)

        # Set matplotlib style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Configure seaborn
        sns.set_palette("husl")

    def plot_sentiment_distribution(self, sentiment_data: List[Dict],
                                   title: str = "Sentiment Analysis Results",
                                   interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot sentiment distribution analysis.

        Args:
            sentiment_data: List of sentiment analysis results
            title: Plot title
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib or Plotly figure
        """
        if not sentiment_data:
            return None

        # Extract sentiment labels
        sentiments = [result.get('label', result.get('sentiment', 'Neutral'))
                     for result in sentiment_data]

        # Count sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()

        if interactive:
            # Create interactive plot with Plotly
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=("Sentiment Distribution", "Sentiment Counts"),
                horizontal_spacing=0.1
            )

            # Pie chart
            colors = ['#2E8B57', '#DC143C', '#708090']  # Green, Red, Gray
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment",
                    marker_colors=colors[:len(sentiment_counts)]
                ),
                row=1, col=1
            )

            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    marker_color=colors[:len(sentiment_counts)],
                    name="Count"
                ),
                row=1, col=2
            )

            fig.update_layout(
                title=title,
                showlegend=False,
                height=500
            )

            return fig

        else:
            # Create static plot with Matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Pie chart
            colors = ['#2E8B57', '#DC143C', '#708090']
            wedges, texts, autotexts = ax1.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                colors=colors[:len(sentiment_counts)],
                autopct='%1.1f%%',
                startangle=90
            )
            ax1.set_title('Sentiment Distribution', fontweight='bold')

            # Bar chart
            bars = ax2.bar(sentiment_counts.index, sentiment_counts.values,
                          color=colors[:len(sentiment_counts)])
            ax2.set_title('Sentiment Counts', fontweight='bold')
            ax2.set_ylabel('Number of Texts')

            # Add value labels on bars
            for bar, count in zip(bars, sentiment_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')

            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()

            return fig

    def plot_sentiment_timeline(self, sentiment_data: List[Dict],
                              timestamps: List[str] = None,
                              time_window: str = 'daily',
                              title: str = "Sentiment Trends Over Time") -> Union[plt.Figure, go.Figure]:
        """
        Plot sentiment trends over time.

        Args:
            sentiment_data: List of sentiment analysis results
            timestamps: List of timestamps corresponding to data
            time_window: Time aggregation window ('daily', 'weekly', 'monthly')
            title: Plot title

        Returns:
            Matplotlib or Plotly figure
        """
        if not sentiment_data:
            return None

        # Create DataFrame
        df = pd.DataFrame(sentiment_data)

        # Extract sentiment scores
        if 'compound' in df.columns:
            df['sentiment_score'] = df['compound']
        elif 'confidence' in df.columns:
            # Convert sentiment to numeric
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            df['sentiment_score'] = df['sentiment'].map(sentiment_map) * df['confidence']
        else:
            # Simple sentiment mapping
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            df['sentiment_score'] = df['sentiment'].map(sentiment_map)

        # Add timestamps if provided
        if timestamps:
            df['timestamp'] = pd.to_datetime(timestamps)
            df.set_index('timestamp', inplace=True)

            # Aggregate by time window
            if time_window == 'daily':
                df_resampled = df.resample('D')
            elif time_window == 'weekly':
                df_resampled = df.resample('W')
            elif time_window == 'monthly':
                df_resampled = df.resample('M')
            else:
                df_resampled = df.resample('D')

            sentiment_trends = df_resampled['sentiment_score'].mean()

            # Create interactive plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sentiment_trends.index,
                y=sentiment_trends.values,
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                hovermode='x unified'
            )

            return fig

        else:
            # Simple line plot
            fig, ax = plt.subplots(figsize=self.figsize)

            ax.plot(range(len(df)), df['sentiment_score'], marker='o', linewidth=2, markersize=4)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Text Index')
            ax.set_ylabel('Sentiment Score')
            ax.grid(True, alpha=0.3)

            return fig

    def plot_topic_modeling_results(self, topic_results: Dict,
                                   title: str = "Topic Modeling Analysis") -> plt.Figure:
        """
        Plot comprehensive topic modeling results.

        Args:
            topic_results: Results from topic modeling
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if not topic_results:
            return None

        num_topics = topic_results.get('num_topics', 0)
        if num_topics == 0:
            return None

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Topic prevalence (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'doc_topic_distribution' in topic_results:
            doc_topic_dist = pd.DataFrame(topic_results['doc_topic_distribution'])
            avg_topic_dist = doc_topic_dist.mean()

            ax1.bar(range(len(avg_topic_dist)), avg_topic_dist.values, color=self.color_palette[:len(avg_topic_dist)])
            ax1.set_title('Average Topic Prevalence', fontweight='bold')
            ax1.set_xlabel('Topic ID')
            ax1.set_ylabel('Average Probability')
            ax1.set_xticks(range(len(avg_topic_dist)))
            ax1.set_xticklabels([f'T{i}' for i in range(len(avg_topic_dist))])

        # 2. Dominant topic distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'doc_topic_distribution' in topic_results:
            doc_topic_dist = pd.DataFrame(topic_results['doc_topic_distribution'])
            dominant_topics = doc_topic_dist.idxmax(axis=1)
            topic_counts = dominant_topics.value_counts().sort_index()

            ax2.bar(topic_counts.index, topic_counts.values, color=self.color_palette[:len(topic_counts)])
            ax2.set_title('Dominant Topic Distribution', fontweight='bold')
            ax2.set_xlabel('Topic ID')
            ax2.set_ylabel('Number of Documents')

        # 3. Topic correlation heatmap (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'doc_topic_distribution' in topic_results:
            doc_topic_dist = pd.DataFrame(topic_results['doc_topic_distribution'])
            topic_corr = doc_topic_dist.corr()

            im = ax3.imshow(topic_corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(topic_corr.columns)))
            ax3.set_yticks(range(len(topic_corr.columns)))
            ax3.set_xticklabels([f'T{i}' for i in range(len(topic_corr.columns))])
            ax3.set_yticklabels([f'T{i}' for i in range(len(topic_corr.columns))])
            ax3.set_title('Topic Correlation Matrix', fontweight='bold')

            # Add colorbar
            plt.colorbar(im, ax=ax3, shrink=0.8)

        # 4-6. Top words for topics (middle row)
        topics = topic_results.get('topics', {})
        for i in range(min(3, num_topics)):
            ax = fig.add_subplot(gs[1, i])
            topic_key = f'topic_{i}'
            if topic_key in topics:
                topic_data = topics[topic_key]
                words = topic_data.get('words', [])[:10]
                weights = topic_data.get('weights', [])[:10]

                if words and weights:
                    y_pos = range(len(words))
                    bars = ax.barh(y_pos, weights, color=self.color_palette[i])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words)
                    ax.set_title(f'Topic {i} Top Words', fontweight='bold')
                    ax.set_xlabel('Weight')
                    ax.invert_yaxis()

        # 7-9. Topic word clouds (bottom row)
        for i in range(min(3, num_topics)):
            ax = fig.add_subplot(gs[2, i])
            topic_key = f'topic_{i}'
            if topic_key in topics:
                topic_data = topics[topic_key]
                words = topic_data.get('words', [])
                weights = topic_data.get('weights', [])

                if words and weights:
                    # Create word frequency dictionary
                    word_freq = {word: weight for word, weight in zip(words, weights)}

                    # Create word cloud
                    wordcloud = WordCloud(
                        width=400, height=300,
                        background_color='white',
                        colormap='viridis',
                        max_words=20
                    ).generate_from_frequencies(word_freq)

                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Topic {i} Word Cloud', fontweight='bold')

        plt.suptitle(title, fontsize=18, fontweight='bold')
        return fig

    def plot_content_quality_analysis(self, quality_data: List[Dict],
                                    title: str = "Content Quality Analysis") -> plt.Figure:
        """
        Plot content quality analysis results.

        Args:
            quality_data: List of quality analysis results
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if not quality_data:
            return None

        # Extract quality metrics
        df = pd.DataFrame(quality_data)

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Overall quality distribution
        ax1 = axes[0]
        if 'overall_score' in df.columns:
            ax1.hist(df['overall_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Overall Quality Score Distribution', fontweight='bold')
            ax1.set_xlabel('Quality Score')
            ax1.set_ylabel('Frequency')
            ax1.axvline(df['overall_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["overall_score"].mean():.3f}')
            ax1.legend()

        # 2. Criteria scores radar chart
        ax2 = axes[1]
        if 'criteria_scores' in df.columns and not df['criteria_scores'].empty:
            # Extract criteria names and average scores
            all_criteria = set()
            for scores in df['criteria_scores']:
                all_criteria.update(scores.keys())

            if all_criteria:
                avg_scores = {}
                for criterion in all_criteria:
                    scores = [item.get(criterion, {}).get('score', 0)
                             for item in df['criteria_scores'] if criterion in item]
                    avg_scores[criterion] = np.mean(scores) if scores else 0

                # Create radar chart
                criteria = list(avg_scores.keys())
                scores = list(avg_scores.values())

                # Add first point to close the circle
                criteria += [criteria[0]]
                scores += [scores[0]]

                angles = np.linspace(0, 2 * np.pi, len(criteria))

                ax2.plot(angles, scores, 'o-', linewidth=2, color='green')
                ax2.fill(angles, scores, alpha=0.25, color='green')
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(criteria[:-1], rotation=45, ha='right')
                ax2.set_title('Average Criteria Scores', fontweight='bold')
                ax2.set_ylim(0, 1)
                ax2.grid(True)

        # 3. Quality vs Length scatter plot
        ax3 = axes[2]
        if 'overall_score' in df.columns:
            # Calculate text lengths if available
            text_lengths = []
            for item in quality_data:
                if 'text' in item:
                    text_lengths.append(len(item['text']))
                else:
                    text_lengths.append(0)

            if text_lengths:
                ax3.scatter(text_lengths, df['overall_score'], alpha=0.6, color='purple')
                ax3.set_title('Quality vs Text Length', fontweight='bold')
                ax3.set_xlabel('Text Length (characters)')
                ax3.set_ylabel('Quality Score')

                # Add trend line
                z = np.polyfit(text_lengths, df['overall_score'], 1)
                p = np.poly1d(z)
                ax3.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)

        # 4. Quality categories
        ax4 = axes[3]
        if 'overall_score' in df.columns:
            # Categorize quality scores
            def categorize_score(score):
                if score >= 0.8:
                    return 'Excellent'
                elif score >= 0.6:
                    return 'Good'
                elif score >= 0.4:
                    return 'Fair'
                else:
                    return 'Poor'

            quality_categories = df['overall_score'].apply(categorize_score)
            category_counts = quality_categories.value_counts()

            colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
            bars = ax4.bar(category_counts.index, category_counts.values,
                           color=colors[:len(category_counts)])
            ax4.set_title('Quality Categories Distribution', fontweight='bold')
            ax4.set_ylabel('Number of Documents')

            # Add value labels
            for bar, count in zip(bars, category_counts.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')

        # 5. Improvement suggestions word cloud
        ax5 = axes[4]
        all_suggestions = []
        for item in quality_data:
            if 'suggestions' in item:
                all_suggestions.extend(item['suggestions'])

        if all_suggestions:
            # Join suggestions and create word cloud
            suggestions_text = ' '.join(all_suggestions)
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='plasma',
                max_words=50
            ).generate(suggestions_text)

            ax5.imshow(wordcloud, interpolation='bilinear')
            ax5.axis('off')
            ax5.set_title('Common Improvement Suggestions', fontweight='bold')

        # 6. Quality trend over time (if timestamps available)
        ax6 = axes[5]
        if 'overall_score' in df.columns:
            # Simple index-based trend
            ax6.plot(range(len(df)), df['overall_score'], marker='o', linewidth=2, markersize=4)
            ax6.set_title('Quality Trend (Document Order)', fontweight='bold')
            ax6.set_xlabel('Document Index')
            ax6.set_ylabel('Quality Score')
            ax6.grid(True, alpha=0.3)

            # Add moving average
            window = min(5, len(df) // 4)
            if window > 1:
                moving_avg = df['overall_score'].rolling(window=window).mean()
                ax6.plot(range(len(df)), moving_avg, 'r-', linewidth=2,
                        label=f'{window}-point Moving Average')
                ax6.legend()

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_content_performance_metrics(self, performance_data: Dict,
                                       title: str = "Content Performance Metrics") -> plt.Figure:
        """
        Plot content performance metrics dashboard.

        Args:
            performance_data: Dictionary with performance metrics
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if not performance_data:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # 1. Engagement metrics
        ax1 = axes[0]
        if 'engagement' in performance_data:
            engagement_data = performance_data['engagement']
            metrics = list(engagement_data.keys())
            values = list(engagement_data.values())

            bars = ax1.bar(metrics, values, color=self.color_palette[:len(metrics)])
            ax1.set_title('Engagement Metrics', fontweight='bold')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')

        # 2. Content categories performance
        ax2 = axes[1]
        if 'category_performance' in performance_data:
            cat_data = performance_data['category_performance']
            categories = list(cat_data.keys())
            avg_scores = [item.get('avg_score', 0) for item in cat_data.values()]

            bars = ax2.bar(categories, avg_scores, color=self.color_palette[:len(categories)])
            ax2.set_title('Performance by Category', fontweight='bold')
            ax2.set_ylabel('Average Score')
            ax2.tick_params(axis='x', rotation=45)

        # 3. Temporal performance trends
        ax3 = axes[2]
        if 'temporal_trends' in performance_data:
            trends = performance_data['temporal_trends']
            dates = list(trends.keys())
            scores = list(trends.values())

            ax3.plot(dates, scores, marker='o', linewidth=2, markersize=6)
            ax3.set_title('Performance Trends Over Time', fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Performance Score')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)

        # 4. Top performing content
        ax4 = axes[3]
        if 'top_content' in performance_data:
            top_content = performance_data['top_content']
            titles = [item.get('title', f'Content {i+1}') for i, item in enumerate(top_content)]
            scores = [item.get('score', 0) for item in top_content]

            # Truncate long titles
            titles = [title[:20] + '...' if len(title) > 20 else title for title in titles]

            bars = ax4.barh(titles, scores, color=self.color_palette[:len(titles)])
            ax4.set_title('Top Performing Content', fontweight='bold')
            ax4.set_xlabel('Performance Score')
            ax4.invert_yaxis()

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_interactive_dashboard(self, analysis_results: Dict,
                                   title: str = "Content Analysis Dashboard") -> go.Figure:
        """
        Create interactive dashboard with multiple analysis views.

        Args:
            analysis_results: Comprehensive analysis results
            title: Dashboard title

        Returns:
            Plotly figure with subplots
        """
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "indicator"}]
            ],
            subplot_titles=[
                "Sentiment Distribution", "Topic Prevalence",
                "Sentiment Timeline", "Quality Metrics",
                "Top Content", "Overall Performance"
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # 1. Sentiment distribution pie chart
        if 'sentiment_analysis' in analysis_results:
            sentiment_data = analysis_results['sentiment_analysis']
            if 'distribution' in sentiment_data:
                labels = list(sentiment_data['distribution'].keys())
                values = list(sentiment_data['distribution'].values())

                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        name="Sentiment"
                    ),
                    row=1, col=1
                )

        # 2. Topic prevalence bar chart
        if 'topic_analysis' in analysis_results:
            topic_data = analysis_results['topic_analysis']
            if 'topic_prevalence' in topic_data:
                topics = list(topic_data['topic_prevalence'].keys())
                prevalence = list(topic_data['topic_prevalence'].values())

                fig.add_trace(
                    go.Bar(
                        x=topics,
                        y=prevalence,
                        name="Topic Prevalence"
                    ),
                    row=1, col=2
                )

        # 3. Sentiment timeline
        if 'sentiment_timeline' in analysis_results:
            timeline_data = analysis_results['sentiment_timeline']
            dates = timeline_data.get('dates', [])
            scores = timeline_data.get('scores', [])

            if dates and scores:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=scores,
                        mode='lines+markers',
                        name="Sentiment Trend"
                    ),
                    row=2, col=1
                )

        # 4. Quality metrics
        if 'quality_analysis' in analysis_results:
            quality_data = analysis_results['quality_analysis']
            if 'metrics' in quality_data:
                metrics = list(quality_data['metrics'].keys())
                values = list(quality_data['metrics'].values())

                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=values,
                        name="Quality Metrics"
                    ),
                    row=2, col=2
                )

        # 5. Top content table
        if 'top_content' in analysis_results:
            top_data = analysis_results['top_content'][:5]  # Top 5 items
            headers = ["Title", "Score", "Category"]

            # Prepare table data
            table_data = []
            for item in top_data:
                row = [
                    item.get('title', 'N/A')[:30],  # Truncate long titles
                    f"{item.get('score', 0):.3f}",
                    item.get('category', 'N/A')
                ]
                table_data.append(row)

            fig.add_trace(
                go.Table(
                    header=dict(values=headers, fill_color='lightblue'),
                    cells=dict(values=list(zip(*table_data)) if table_data else [[], [], []],
                              fill_color='lightgray')
                ),
                row=3, col=1
            )

        # 6. Overall performance indicator
        if 'overall_performance' in analysis_results:
            overall_score = analysis_results['overall_performance'].get('score', 0)

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Performance"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.7], 'color': "gray"},
                            {'range': [0.7, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=False
        )

        return fig

    def create_word_cloud(self, text_data: Union[str, List[str]],
                         title: str = "Word Cloud",
                         width: int = 800,
                         height: int = 400,
                         colormap: str = 'viridis') -> plt.Figure:
        """
        Create word cloud visualization.

        Args:
            text_data: Text or list of texts to visualize
            title: Plot title
            width: Word cloud width
            height: Word cloud height
            colormap: Matplotlib colormap name

        Returns:
            Matplotlib figure
        """
        if not text_data:
            return None

        # Combine text if list provided
        if isinstance(text_data, list):
            text = ' '.join(text_data)
        else:
            text = text_data

        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontweight='bold', fontsize=16)

        return fig

    def save_visualization(self, fig: Union[plt.Figure, go.Figure],
                          filename: str,
                          format: str = 'png',
                          dpi: int = 300):
        """
        Save visualization to file.

        Args:
            fig: Figure to save
            filename: Output filename
            format: Output format ('png', 'jpg', 'svg', 'html')
            dpi: Resolution for raster formats
        """
        try:
            if isinstance(fig, go.Figure):
                if format.lower() == 'html':
                    fig.write_html(filename)
                else:
                    fig.write_image(filename, format=format, width=1200, height=800)
            else:
                fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        except Exception as e:
            print(f"Error saving visualization: {e}")