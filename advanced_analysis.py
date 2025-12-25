import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


def load_cleaned_data(csv_path):
    """
    Loads the cleaned Netflix dataset.
    """
    print("--- Loading Cleaned Data ---")
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}\n")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ============================================================================
# 1. CONTENT DISTRIBUTION ANALYSIS
# ============================================================================


def analyze_content_distribution(df):
    """
    Analyzes the distribution of content by type (Movie vs TV Show).
    """
    print("=" * 60)
    print("1. CONTENT DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Overall distribution
    type_counts = df['type'].value_counts()
    print("\nContent Distribution:")
    print(type_counts)
    print(f"\nPercentage:")
    print((type_counts / len(df) * 100).round(2))

    # By rating
    print("\n--- Content Distribution by Rating ---")
    rating_type = pd.crosstab(df['rating'], df['type'])
    print(rating_type)

    # By genre (listed_in)
    print("\n--- Top 15 Genres ---")
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])

    genre_counts = Counter(all_genres)
    top_genres = pd.Series(dict(genre_counts.most_common(15)))
    print(top_genres)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Content Type Distribution
    type_counts.plot(kind='bar', ax=axes[0, 0], color=['#E50914', '#221F1F'])
    axes[0, 0].set_title('Content Type Distribution',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xlabel('Type')
    axes[0, 0].tick_params(axis='x', rotation=0)

    # Content Type Pie Chart
    axes[0, 1].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
                   colors=['#E50914', '#221F1F'], startangle=90)
    axes[0, 1].set_title('Content Type Percentage',
                         fontsize=14, fontweight='bold')

    # Top Genres
    top_genres.plot(kind='barh', ax=axes[1, 0], color='#E50914')
    axes[1, 0].set_title('Top 15 Genres', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Count')

    # Rating Distribution
    rating_counts = df['rating'].value_counts().head(10)
    rating_counts.plot(kind='bar', ax=axes[1, 1], color='#221F1F')
    axes[1, 1].set_title('Top 10 Content Ratings',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('content_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Chart saved as 'content_distribution.png'")
    plt.show()

# ============================================================================
# 2. RELEASE DATE ANALYSIS
# ============================================================================


def analyze_release_dates(df):
    """
    Analyzes content release and addition dates.
    """
    print("\n" + "=" * 60)
    print("2. RELEASE DATE & ADDITION DATE ANALYSIS")
    print("=" * 60)

    # Release year distribution
    print("\n--- Content by Release Year (Top 10) ---")
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    release_year_counts = df['release_year'].value_counts(
    ).sort_index(ascending=False).head(10)
    print(release_year_counts)

    # Addition year distribution
    print("\n--- Content Added to Netflix by Year ---")
    year_added_counts = df['year_added'].value_counts().sort_index()
    print(year_added_counts)

    # Addition month distribution
    print("\n--- Content Added by Month ---")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month_added'].value_counts().reindex(month_order)
    print(month_counts)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Release Year Trend (top 20 years)
    top_release_years = df['release_year'].value_counts(
    ).sort_index(ascending=False).head(20).sort_index()
    axes[0, 0].plot(top_release_years.index, top_release_years.values,
                    marker='o', linewidth=2, markersize=6, color='#E50914')
    axes[0, 0].fill_between(top_release_years.index,
                            top_release_years.values, alpha=0.3, color='#E50914')
    axes[0, 0].set_title('Content Release Year Trend (Top 20)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Release Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

    # Year Added Trend
    year_added_counts.plot(kind='bar', ax=axes[0, 1], color='#221F1F')
    axes[0, 1].set_title('Content Added to Netflix by Year',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xlabel('Year Added')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Month Added Distribution
    month_counts.plot(kind='bar', ax=axes[1, 0], color='#E50914')
    axes[1, 0].set_title('Content Added by Month (All Years)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Content Type by Year Added
    type_by_year = pd.crosstab(df['year_added'], df['type'])
    type_by_year.plot(kind='bar', stacked=True,
                      ax=axes[1, 1], color=['#E50914', '#221F1F'])
    axes[1, 1].set_title('Movie vs TV Show Added by Year',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xlabel('Year Added')
    axes[1, 1].legend(title='Type')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('release_dates_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Chart saved as 'release_dates_analysis.png'")
    plt.show()

# ============================================================================
# 3. COUNTRY ANALYSIS
# ============================================================================


def analyze_countries(df):
    """
    Analyzes content production by country.
    """
    print("\n" + "=" * 60)
    print("3. COUNTRY ANALYSIS")
    print("=" * 60)

    # Parse countries (multiple countries separated by comma)
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend([c.strip() for c in countries.split(',')])

    country_counts = Counter(all_countries)
    top_countries = pd.Series(dict(country_counts.most_common(20)))

    print("\n--- Top 20 Countries by Content Production ---")
    print(top_countries)

    # Content type by top countries
    print("\n--- Content Type Distribution in Top 10 Countries ---")
    top_10_countries = top_countries.head(10).index.tolist()

    country_type_data = []
    for country in top_10_countries:
        country_mask = df['country'].str.contains(country, na=False)
        country_types = df[country_mask]['type'].value_counts()
        country_type_data.append(country_types)

    country_type_df = pd.DataFrame(
        country_type_data, index=top_10_countries).fillna(0)
    print(country_type_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top 20 Countries
    top_countries.plot(kind='barh', ax=axes[0], color='#E50914')
    axes[0].set_title('Top 20 Countries by Content Production',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Titles')
    axes[0].invert_yaxis()

    # Content Type by Top 10 Countries
    country_type_df.plot(kind='barh', stacked=True,
                         ax=axes[1], color=['#E50914', '#221F1F'])
    axes[1].set_title('Movie vs TV Show in Top 10 Countries',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Titles')
    axes[1].legend(title='Type', loc='lower right')

    plt.tight_layout()
    plt.savefig('country_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Chart saved as 'country_analysis.png'")
    plt.show()

# ============================================================================
# 4. CAST ANALYSIS
# ============================================================================


def analyze_cast(df):
    """
    Analyzes the most frequent actors and cast members.
    """
    print("\n" + "=" * 60)
    print("4. CAST ANALYSIS")
    print("=" * 60)

    # Parse cast (multiple actors separated by comma)
    all_cast = []
    for cast_list in df['cast'].dropna():
        all_cast.extend([actor.strip() for actor in cast_list.split(',')])

    cast_counts = Counter(all_cast)
    top_cast = pd.Series(dict(cast_counts.most_common(30)))

    print("\n--- Top 30 Most Frequent Actors ---")
    print(top_cast)

    # Cast by content type
    print("\n--- Top 10 Actors in Movies vs TV Shows ---")
    top_10_actors = top_cast.head(10).index.tolist()

    actor_type_data = []
    for actor in top_10_actors:
        actor_mask = df['cast'].str.contains(actor, na=False)
        actor_types = df[actor_mask]['type'].value_counts()
        actor_type_data.append(actor_types)

    actor_type_df = pd.DataFrame(
        actor_type_data, index=top_10_actors).fillna(0)
    print(actor_type_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Top 20 Actors
    top_cast.head(20).plot(kind='barh', ax=axes[0], color='#221F1F')
    axes[0].set_title('Top 20 Most Frequent Actors',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Titles')
    axes[0].invert_yaxis()

    # Content Type by Top 10 Actors
    actor_type_df.plot(kind='barh', stacked=True,
                       ax=axes[1], color=['#E50914', '#221F1F'])
    axes[1].set_title('Movie vs TV Show for Top 10 Actors',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Titles')
    axes[1].legend(title='Type', loc='lower right')

    plt.tight_layout()
    plt.savefig('cast_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Chart saved as 'cast_analysis.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def run_advanced_analysis(cleaned_csv_path):
    """
    Executes all advanced analyses.
    """
    df = load_cleaned_data(cleaned_csv_path)

    if df is not None:
        # Run all analyses
        analyze_content_distribution(df)
        analyze_release_dates(df)
        analyze_countries(df)
        analyze_cast(df)

        print("\n" + "=" * 60)
        print("✅ ALL ANALYSES COMPLETED!")
        print("=" * 60)
        print("\nGenerated visualizations:")
        print("  1. content_distribution.png")
        print("  2. release_dates_analysis.png")
        print("  3. country_analysis.png")
        print("  4. cast_analysis.png")


if __name__ == "__main__":
    CLEANED_FILE = "netflix_cleaned.csv"

    print("Starting Advanced Netflix Analysis...\n")
    run_advanced_analysis(CLEANED_FILE)
