"""
Netflix veri seti üzerinde istatistiksel analizler, aykırı değer tespiti
ve dağılım incelemeleri yapan modül.
"""
import warnings
from collections import Counter
from scipy import stats
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def load_cleaned_data(csv_path):
    """
    Load the cleaned Netflix dataset from CSV file.
    """
    print("--- Loading Cleaned Data ---")
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}\n")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def analyze_content_distribution(df):
    """
    Analyze content distribution by type, rating, and genre.
    Includes baseline, advanced, and data science levels.
    """
    print("\n" + "=" * 90)
    print("SECTION 1: CONTENT DISTRIBUTION ANALYSIS")
    print("=" * 90)

    print("\n[BASELINE] Simple Aggregations & Data Summary")
    print("-" * 90)

    # Content type distribution
    type_counts = df['type'].value_counts()
    total_titles = len(df)

    print(f"\nTotal Titles in Dataset: {total_titles}")
    print("Content Type Breakdown:")
    for content_type, count in type_counts.items():
        percentage = (count / total_titles) * 100
        print(f"  {content_type}: {count} ({percentage:.2f}%)")

    # Rating distribution
    print("\nRating Distribution (Top 10):")
    rating_counts = df['rating'].value_counts().head(10)
    for rating, count in rating_counts.items():
        print(f"  {rating}: {count} titles")

    # Genre analysis
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])

    genre_counts = Counter(all_genres)
    top_genres = pd.Series(dict(genre_counts.most_common(10)))

    print("\nTop 10 Genres:")
    for genre, count in top_genres.items():
        percentage = (count / len(df)) * 100
        print(f"  {genre}: {count} ({percentage:.2f}%)")

    # Summary statistics
    print("\nContent Distribution Summary:")
    print(f"  Total Movies: {type_counts.get('Movie', 0)}")
    print(f"  Total TV Shows: {type_counts.get('TV Show', 0)}")
    print(f"  Unique Ratings: {df['rating'].nunique()}")
    print(f"  Unique Genres: {len(genre_counts)}")

    print("\n[ADVANCED] Statistical Analysis - Content Distribution")
    print("-" * 90)

    # Quartile analysis for content distribution
    type_by_rating = pd.crosstab(df['rating'], df['type'])
    print("\nContent Type by Rating Distribution:")
    print(type_by_rating)

    # Distribution of genres
    print(f"\nGenre Distribution Statistics:")
    genre_series = pd.Series(dict(genre_counts))
    print(f"  Mean titles per genre: {genre_series.mean():.2f}")
    print(f"  Median titles per genre: {genre_series.median():.2f}")
    print(f"  Min titles per genre: {genre_series.min()}")
    print(f"  Max titles per genre: {genre_series.max()}")
    print(f"  Q1 (25%): {genre_series.quantile(0.25):.2f}")
    print(f"  Q3 (75%): {genre_series.quantile(0.75):.2f}")
    print(
        f"  IQR: {genre_series.quantile(0.75) - genre_series.quantile(0.25):.2f}")

    # Skewness and distribution shape
    genre_skewness = stats.skew(genre_series.values)
    genre_kurtosis = stats.kurtosis(genre_series.values)
    print("\nDistribution Shape:")
    print(
        f"  Skewness: {genre_skewness:.4f} (Right-skewed - few dominant genres)")
    print(f"  Kurtosis: {genre_kurtosis:.4f}")

    print("\n[DATA SCIENCE] Anomaly Detection - Content Distribution")
    print("-" * 90)

    # Identify unusual rating distributions
    q1_rating = type_counts.quantile(0.25) if hasattr(
        type_counts, 'quantile') else type_counts.values.mean() - type_counts.values.std()
    q3_rating = type_counts.quantile(0.75) if hasattr(
        type_counts, 'quantile') else type_counts.values.mean() + type_counts.values.std()

    # Identify dominant genres using statistical methods
    q1_genre = genre_series.quantile(0.25)
    q3_genre = genre_series.quantile(0.75)
    iqr_genre = q3_genre - q1_genre
    upper_bound_genre = q3_genre + 1.5 * iqr_genre

    dominant_genres = genre_series[genre_series >
                                   upper_bound_genre].sort_values(ascending=False)

    print("\nDominant Genres (Statistical Outliers using IQR method):")
    print(f"  Outlier threshold: > {upper_bound_genre:.0f} titles")
    print(f"  Dominant genres found: {len(dominant_genres)}")

    if len(dominant_genres) > 0:
        for genre, count in dominant_genres.items():
            z_score = (count - genre_series.mean()) / genre_series.std()
            print(f"  - {genre}: {count} titles (Z-score: {z_score:.2f})")

    # Z-score anomaly detection for ratings
    rating_series = type_counts
    rating_z_scores = np.abs(stats.zscore(rating_series.values))
    anomalous_ratings = rating_series[rating_z_scores > 2]

    print("\nAnomalous Rating Distribution (Z-score > 2):")
    if len(anomalous_ratings) > 0:
        for rating, count in anomalous_ratings.items():
            print(f"  - {rating}: {count} titles")
    else:
        print("  No anomalies detected in rating distribution.")

    # Data quality assessment
    print("\nData Quality Assessment:")
    missing_listed_in = df['listed_in'].isnull().sum()
    print(
        f"  Missing genre information: {missing_listed_in} ({missing_listed_in/len(df)*100:.2f}%)")


def analyze_release_dates(df):
    """
    Analyze content release dates and addition dates to Netflix.
    Includes baseline, advanced, and data science levels.
    """
    print("\n" + "=" * 90)
    print("SECTION 2: RELEASE DATE & ADDITION DATE ANALYSIS")
    print("=" * 90)

    print("\n[BASELINE] Simple Aggregations & Date Statistics")
    print("-" * 90)

    # Year added distribution
    year_added_counts = df['year_added'].value_counts().sort_index()

    print("\nContent Added to Netflix by Year:")
    print(f"  Min year: {year_added_counts.index.min()}")
    print(f"  Max year: {year_added_counts.index.max()}")
    print(f"  Total years in dataset: {len(year_added_counts)}")

    for year, count in year_added_counts.tail(5).sort_index(ascending=False).items():
        print(f"  {int(year)}: {count} titles")

    # Month added distribution
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month_added'].value_counts().reindex(month_order)

    print("\nContent Added by Month (All Years):")
    month_stats = {
        'Sum': month_counts.sum(),
        'Mean': month_counts.mean(),
        'Max': month_counts.max(),
        'Min': month_counts.min()
    }

    for stat, value in month_stats.items():
        print(f"  {stat}: {value:.2f}")

    # Release year distribution
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    release_year_counts = df['release_year'].value_counts(
    ).sort_index(ascending=False).head(10)

    print("\nTop 10 Release Years:")
    for year, count in release_year_counts.items():
        if not np.isnan(year):
            print(f"  {int(year)}: {count} titles")

    print("\n[ADVANCED] Statistical Analysis - Release Dates")
    print("-" * 90)

    # Quartile analysis for year added
    year_quartiles = df['year_added'].quantile([0.25, 0.5, 0.75])

    print("\nYear Added Quartile Analysis:")
    print(f"  Q1 (25%): {year_quartiles[0.25]:.0f}")
    print(f"  Q2 (50%/Median): {year_quartiles[0.5]:.0f}")
    print(f"  Q3 (75%): {year_quartiles[0.75]:.0f}")
    print(f"  IQR: {year_quartiles[0.75] - year_quartiles[0.25]:.0f}")

    # Distribution shape analysis
    year_skewness = stats.skew(df['year_added'].dropna())
    year_kurtosis = stats.kurtosis(df['year_added'].dropna())

    print("\nYear Added Distribution Shape:")
    print(f"  Mean: {df['year_added'].mean():.2f}")
    print(f"  Median: {df['year_added'].median():.2f}")
    print(f"  Std Dev: {df['year_added'].std():.2f}")
    print(f"  Skewness: {year_skewness:.4f}")
    print(f"  Kurtosis: {year_kurtosis:.4f}")

    # Month distribution statistics
    print("\nMonth Added Distribution Statistics:")
    print(f"  Mean titles per month: {month_counts.mean():.2f}")
    print(f"  Median: {month_counts.median():.2f}")
    print(
        f"  Max (busiest month): {month_counts.idxmax()} with {month_counts.max()} titles")
    print(
        f"  Min (slowest month): {month_counts.idxmin()} with {month_counts.min()} titles")

    print("\n[DATA SCIENCE] Anomaly Detection - Release Dates")
    print("-" * 90)

    # IQR method for outlier detection in year added
    q1_year = df['year_added'].quantile(0.25)
    q3_year = df['year_added'].quantile(0.75)
    iqr_year = q3_year - q1_year
    lower_bound_year = q1_year - 1.5 * iqr_year
    upper_bound_year = q3_year + 1.5 * iqr_year

    outlier_years = df[(df['year_added'] < lower_bound_year)
                       | (df['year_added'] > upper_bound_year)]

    print("\nOutliers in Year Added (IQR method):")
    print(f"  Normal bounds: [{lower_bound_year:.0f}, {upper_bound_year:.0f}]")
    print(
        f"  Outliers found: {len(outlier_years)} ({len(outlier_years)/len(df)*100:.2f}%)")

    if len(outlier_years) > 0:
        print(f"  Years with outliers:")
        outlier_year_counts = outlier_years['year_added'].value_counts(
        ).sort_index()
        for year, count in outlier_year_counts.items():
            print(f"    {int(year)}: {count} titles")

    # Z-score anomaly detection for months
    month_z_scores = np.abs(stats.zscore(month_counts.values))
    anomalous_months = month_counts[month_z_scores > 2]

    print("\nAnomalous Months (Z-score > 2):")
    if len(anomalous_months) > 0:
        for month, count in anomalous_months.items():
            print(f"  - {month}: {count} titles")
    else:
        print("  No significant anomalies in monthly distribution.")

    # Release year vs addition year analysis
    print("\nRelease Year Analysis:")
    release_year_clean = df['release_year'].dropna()
    print(
        f"  Release year range: {release_year_clean.min():.0f} - {release_year_clean.max():.0f}")
    print(f"  Mean release year: {release_year_clean.mean():.2f}")
    print(f"  Median release year: {release_year_clean.median():.2f}")

    # Normality test for dates
    stat, p_value = stats.shapiro(df['year_added'].dropna())
    print("\nNormality Test (Shapiro-Wilk) for Year Added:")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Distribution: {'Non-normal' if p_value < 0.05 else 'Normal'}")


def analyze_countries(df):
    """
    Analyze content production by country.
    Includes baseline, advanced, and data science levels.
    """
    print("\n" + "=" * 90)
    print("SECTION 3: COUNTRY ANALYSIS")
    print("=" * 90)

    # Parse countries from dataset
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend([c.strip() for c in countries.split(',')])

    country_counts = Counter(all_countries)
    country_series = pd.Series(dict(country_counts))

    print("\n[BASELINE] Simple Aggregations & Country Statistics")
    print("-" * 90)

    print("\nCountry Production Overview:")
    print(f"  Total unique countries: {len(country_series)}")
    print(
        f"  Total productions (with multiple countries): {len(all_countries)}")
    print(
        f"  Average: {len(all_countries) / len(df) * 100:.2f}% of content is international")

    print("\nTop 15 Countries by Title Count:")
    top_countries = country_series.nlargest(15)
    for country, count in top_countries.items():
        percentage = (count / len(df)) * 100
        print(f"  {country}: {count} titles ({percentage:.2f}%)")

    print("\nCountry Summary Statistics:")
    print(f"  Sum (total productions): {country_series.sum()}")
    print(f"  Mean (avg per country): {country_series.mean():.2f}")
    print(f"  Min (smallest): {country_series.min()}")
    print(f"  Max (largest): {country_series.max()}")

    print("\n[ADVANCED] Statistical Analysis - Country Distribution")
    print("-" * 90)

    # Quartile analysis
    q1_country = country_series.quantile(0.25)
    q2_country = country_series.quantile(0.50)
    q3_country = country_series.quantile(0.75)
    iqr_country = q3_country - q1_country

    print("\nCountry Production Quartile Analysis:")
    print(f"  Q1 (25%): {q1_country:.2f}")
    print(f"  Q2 (50%/Median): {q2_country:.2f}")
    print(f"  Q3 (75%): {q3_country:.2f}")
    print(f"  IQR: {iqr_country:.2f}")
    print(f"  Std Dev: {country_series.std():.2f}")

    # Distribution shape
    country_skewness = stats.skew(country_series.values)
    country_kurtosis = stats.kurtosis(country_series.values)

    print("\nCountry Distribution Shape:")
    print(
        f"  Skewness: {country_skewness:.4f} (Right-skewed - few dominant countries)")
    print(f"  Kurtosis: {country_kurtosis:.4f} (Heavy tails - extreme values)")

    # Content type by top countries
    print("\nContent Type Distribution in Top 10 Countries:")
    top_10_countries = top_countries.head(10).index.tolist()
    for country in top_10_countries:
        country_mask = df['country'].str.contains(country, na=False)
        country_df = df[country_mask]
        movie_count = len(country_df[country_df['type'] == 'Movie'])
        show_count = len(country_df[country_df['type'] == 'TV Show'])
        print(f"  {country}: {movie_count} Movies, {show_count} TV Shows")

    # ========================================================================
    # DATA SCIENCE: Anomaly detection in country distribution
    # ========================================================================
    print("\n[DATA SCIENCE] Anomaly Detection - Country Distribution")
    print("-" * 90)

    # IQR method for outlier detection
    upper_bound_country = q3_country + 1.5 * iqr_country
    lower_bound_country = q1_country - 1.5 * iqr_country

    outlier_countries = country_series[(country_series > upper_bound_country) |
                                       (country_series < lower_bound_country)]

    print("\nOutlier Countries (IQR method):")
    print(
        f"  Normal range: [{lower_bound_country:.2f}, {upper_bound_country:.2f}]")
    print(f"  Countries with unusual production: {len(outlier_countries)}")

    if len(outlier_countries) > 0:
        print(f"\nDominant Countries (Upper Outliers):")
        for country, count in outlier_countries[outlier_countries > upper_bound_country].sort_values(ascending=False).items():
            z_score = (count - country_series.mean()) / country_series.std()
            print(f"  - {country}: {count} titles (Z-score: {z_score:.2f})")

    # Z-score method
    print("\nZ-Score Anomaly Detection (|z| > 2.5):")
    country_z_scores = np.abs(stats.zscore(country_series.values))
    extreme_countries = country_series[country_z_scores > 2.5]

    print(f"  Extreme outliers found: {len(extreme_countries)}")
    if len(extreme_countries) > 0:
        for country, count in extreme_countries.sort_values(ascending=False).items():
            print(f"  - {country}: {count} titles")

    # Data quality
    print("\nData Quality Assessment:")
    missing_country = df['country'].isnull().sum()
    print(
        f"  Missing country information: {missing_country} ({missing_country/len(df)*100:.2f}%)")


def analyze_cast(df):
    """
    Analyze content cast and actors.
    Includes baseline, advanced, and data science levels.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: CAST ANALYSIS")
    print("=" * 70)

    # Parse cast from dataset
    all_cast = []
    for cast_list in df['cast'].dropna():
        all_cast.extend([actor.strip() for actor in cast_list.split(',')])

    cast_counts = Counter(all_cast)
    cast_series = pd.Series(dict(cast_counts))

    print("\n[BASELINE] Simple Aggregations & Cast Statistics")
    print("-" * 70)

    print("\nCast Overview:")
    print(f"  Total unique actors: {len(cast_series)}")
    print(f"  Total cast appearances: {len(all_cast)}")
    print(
        f"  Average appearances per actor: {len(all_cast) / len(cast_series):.2f}")

    print(f"\nTop 20 Most Frequent Actors:")
    top_cast = cast_series.nlargest(20)
    for actor, count in top_cast.items():
        percentage = (count / len(df)) * 100
        print(f"  {actor}: {count} titles ({percentage:.2f}% of catalog)")

    print("\nCast Summary Statistics:")
    print(f"  Sum (total appearances): {cast_series.sum()}")
    print(f"  Mean (avg per actor): {cast_series.mean():.2f}")
    print(f"  Min (lowest): {cast_series.min()}")
    print(f"  Max (highest): {cast_series.max()}")

    print("\n[ADVANCED] Statistical Analysis - Cast Distribution")
    print("-" * 90)

    # Quartile analysis
    q1_cast = cast_series.quantile(0.25)
    q2_cast = cast_series.quantile(0.50)
    q3_cast = cast_series.quantile(0.75)
    iqr_cast = q3_cast - q1_cast

    print("\nActor Appearance Quartile Analysis:")
    print(f"  Q1 (25%): {q1_cast:.2f}")
    print(f"  Q2 (50%/Median): {q2_cast:.2f}")
    print(f"  Q3 (75%): {q3_cast:.2f}")
    print(f"  IQR: {iqr_cast:.2f}")
    print(f"  Std Dev: {cast_series.std():.2f}")

    # Distribution shape
    cast_skewness = stats.skew(cast_series.values)
    cast_kurtosis = stats.kurtosis(cast_series.values)

    print("\nActor Distribution Shape:")
    print(
        f"  Skewness: {cast_skewness:.4f} (Right-skewed - few prolific actors)")
    print(f"  Kurtosis: {cast_kurtosis:.4f} (Heavy tails - extreme actors)")

    # Content type analysis for top actors
    print("\nContent Type Distribution for Top 10 Actors:")
    top_10_actors = top_cast.head(10).index.tolist()
    for actor in top_10_actors:
        actor_mask = df['cast'].str.contains(actor, na=False)
        actor_df = df[actor_mask]
        movie_count = len(actor_df[actor_df['type'] == 'Movie'])
        show_count = len(actor_df[actor_df['type'] == 'TV Show'])
        total = movie_count + show_count
        print(f"  {actor}: {movie_count} Movies ({movie_count/total*100:.1f}%), {show_count} Shows ({show_count/total*100:.1f}%)")

    print("\n[DATA SCIENCE] Anomaly Detection - Cast Distribution")
    print("-" * 90)

    # IQR method for outlier detection
    upper_bound_cast = q3_cast + 1.5 * iqr_cast
    lower_bound_cast = q1_cast - 1.5 * iqr_cast

    outlier_actors = cast_series[(cast_series > upper_bound_cast) |
                                 (cast_series < lower_bound_cast)]

    print("\nOutlier Actors (IQR method):")
    print(f"  Normal range: [{lower_bound_cast:.2f}, {upper_bound_cast:.2f}]")
    print(
        f"  Prolific actors detected: {len(outlier_actors[outlier_actors > upper_bound_cast])}")

    if len(outlier_actors[outlier_actors > upper_bound_cast]) > 0:
        print("\nProlific Actors (Upper Outliers):")
        for actor, count in outlier_actors[outlier_actors > upper_bound_cast].sort_values(ascending=False).items():
            z_score = (count - cast_series.mean()) / cast_series.std()
            print(f"  - {actor}: {count} titles (Z-score: {z_score:.2f})")

    # Z-score method
    print("\nZ-Score Anomaly Detection (|z| > 2.5):")
    cast_z_scores = np.abs(stats.zscore(cast_series.values))
    extreme_actors = cast_series[cast_z_scores > 2.5]

    print(f"  Extreme prolific actors found: {len(extreme_actors)}")
    if len(extreme_actors) > 0:
        for actor, count in extreme_actors.sort_values(ascending=False).items():
            print(f"  - {actor}: {count} titles")

    # Data quality
    print("\nData Quality Assessment:")
    missing_cast = df['cast'].isnull().sum()
    print(
        f"  Missing cast information: {missing_cast} ({missing_cast/len(df)*100:.2f}%)")

    # Statistical test
    if len(cast_series) > 3:
        stat, p_value = stats.shapiro(cast_series.values[:5000] if len(
            cast_series) > 5000 else cast_series.values)
        print("\nNormality Test (Shapiro-Wilk) for Actor Distribution:")
        print(f"  p-value: {p_value:.6f}")
        print(
            f"  Distribution: {'Non-normal' if p_value < 0.05 else 'Normal'}")


def main():
    """
    Main function to execute all analysis sections.
    """
    print("=" * 70)
    print("NETFLIX DATASET COMPREHENSIVE ANALYSIS")
    print("Analyzing: Content Distribution, Release Dates, Countries, and Cast")
    print("=" * 70)

    CLEANED_FILE = "netflix_cleaned.csv"

    df = load_cleaned_data(CLEANED_FILE)

    if df is not None:
        # Run all analysis sections
        analyze_content_distribution(df)
        analyze_release_dates(df)
        analyze_countries(df)
        analyze_cast(df)

        print("\n" + "=" * 70)
        print(" ANALYSIS COMPLETE")
        print("=" * 90)
        print("\nAnalysis Levels Applied:")
        print("  [BASELINE] Simple aggregations with mean, max, min, sum")
        print(
            "  [ADVANCED] Statistical analysis with quartiles and distribution analysis")
        print("  [DATA SCIENCE] Anomaly detection using IQR and Z-score methods")
        print("=" * 70)


if __name__ == "__main__":
    main()
