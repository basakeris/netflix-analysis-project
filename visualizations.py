"""
Generates charts for content growth, countries, genres, and ratings.
"""
from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset

try:
    df = pd.read_csv("netflix_cleaned.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'netflix_cleaned.csv' not found.")
    exit()

sns.set_theme(style="darkgrid")

# 1. Content growth over years
yc = df["release_year"].value_counts().sort_index().reset_index()
yc.columns = ["year", "count"]

plt.figure(figsize=(12, 6))
sns.lineplot(data=yc, x="year", y="count", marker="o")
plt.title("Netflix Content Growth Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Amount of Content")
plt.savefig("chart_1_growth.png")
plt.show()

# 2. Top 10 countries
c_data = df[df["country"] != "Unknown Country"]["country"]
c = c_data.astype(str).str.split(
    ", ").explode().value_counts().head(10).to_frame()

plt.figure(figsize=(10, 8))
sns.heatmap(c, annot=True, fmt="d", cmap="Reds")
plt.title("Top 10 Countries with Most Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.savefig("chart_2_countries.png")
plt.show()

# 3. Release year distribution by top genres
d2 = df.copy()
d2["genre"] = d2["listed_in"].astype(str).str.split(",").str[0]
g = d2["genre"].value_counts().head(8).index
d2 = d2[d2["genre"].isin(g)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=d2, x="genre", y="release_year")
plt.title("Release Year Distribution by Top Genres")
plt.xticks(rotation=45)
plt.xlabel("Genre")
plt.ylabel("Release Year")
plt.savefig("chart_3_genres.png")
plt.show()

# 4. Movies vs TV Shows count
t = df["type"].value_counts().reset_index()
t.columns = ["type", "count"]

plt.figure(figsize=(8, 5))
sns.barplot(data=t, x="type", y="count")
plt.title("Movies vs TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.savefig("chart_4_types.png")
plt.show()

# 5. Rating distribution (Top 12)
r = df["rating"].value_counts().head(12).reset_index()
r.columns = ["rating", "count"]

plt.figure(figsize=(12, 5))
sns.barplot(data=r, x="rating", y="count")
plt.title("Content Rating Distribution (Top 12)")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.savefig("chart_5_ratings.png")
plt.show()

# 6. Rating vs Type heatmap
x = df[["rating", "type"]]
tab = pd.crosstab(x["rating"], x["type"])

plt.figure(figsize=(12, 6))
sns.heatmap(tab, annot=True, fmt="d", cmap="Blues")
plt.title("Rating vs Type Relationship")
plt.xlabel("Type")
plt.ylabel("Rating")
plt.savefig("chart_6_heatmap.png")
plt.show()

# 7. Content added by month
order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
m = df["month_added"].value_counts().reindex(order).reset_index()
m.columns = ["month", "count"]

plt.figure(figsize=(12, 5))
sns.barplot(data=m, x="month", y="count")
plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("chart_7_months.png")
plt.show()

# 8. Top 20 actors
cs_data = df[df["cast"] != "Unknown Cast"]["cast"]
cs = cs_data.astype(str).str.split(", ").explode()
top = cs.value_counts().head(20).reset_index()
top.columns = ["name", "count"]

plt.figure(figsize=(12, 8))
sns.barplot(data=top, y="name", x="count")
plt.title("Top 20 Most Frequent Actors & Actresses")
plt.xlabel("Number of Titles")
plt.ylabel("Actor / Actress")
plt.savefig("chart_8_actors.png")
plt.show()

# ============ ADVANCED LEVEL: BOX PLOTS ============
print("\n=== ADVANCED ANALYSIS: Outlier Detection ===\n")

# 9. Release Year - Outlier Detection (Box Plot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, y='release_year', color='skyblue')
plt.title("Release Year Distribution - Outlier Detection (Box Plot)")
plt.ylabel("Release Year")
plt.axhline(y=df['release_year'].quantile(0.25),
            color='red', linestyle='--', label='Q1')
plt.axhline(y=df['release_year'].quantile(0.75),
            color='green', linestyle='--', label='Q3')
plt.legend()
plt.savefig("chart_9_outliers_boxplot.png")
plt.show()

# 10. Duration - Outlier Detection
df_with_duration = df[df['duration_numeric'].notna()]
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_with_duration, y='duration_numeric', color='lightcoral')
plt.title("Movie Duration Distribution - Outlier Detection (Box Plot)")
plt.ylabel("Duration (minutes)")
plt.savefig("chart_10_duration_outliers.png")
plt.show()

# ============ DATA SCIENCE LEVEL: ANOMALY DETECTION ============
print("\n=== DATA SCIENCE: Z-Score Anomaly Detection ===\n")


# 11. Z-Score Anomaly Detection for Release Year
z_scores = np.abs(stats.zscore(df['release_year'].dropna()))
anomalies = df[np.abs(stats.zscore(df['release_year'].dropna())) > 2.5]

print(f"Anomalies Found (Z-score > 2.5): {len(anomalies)}")
if len(anomalies) > 0:
    print(f"Anomalous Years: {anomalies['release_year'].unique()}")

plt.figure(figsize=(12, 6))
plt.scatter(df.index, df['release_year'], alpha=0.6, label='Normal Data', s=30)
plt.scatter(anomalies.index, anomalies['release_year'],
            color='red', label='Anomalies (Z-score > 2.5)', s=100, marker='X')
plt.title("Release Year - Z-Score Anomaly Detection")
plt.xlabel("Index")
plt.ylabel("Release Year")
plt.legend()
plt.savefig("chart_11_zscore_anomalies.png")
plt.show()

# 12. IQR Method - Content Count by Type (Anomaly Detection)
type_counts = df['type'].value_counts()
Q1 = type_counts.quantile(0.25)
Q3 = type_counts.quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

outlier_types = type_counts[(type_counts > upper_bound)
                            | (type_counts < lower_bound)]

print(f"\nIQR Method - Content Type Analysis:")
print(f"Upper Bound: {upper_bound}, Lower Bound: {lower_bound}")
print(f"Outliers: {outlier_types.to_dict()}")

plt.figure(figsize=(10, 6))
colors = ['red' if x > upper_bound or x < lower_bound else 'skyblue'
          for x in type_counts.values]
sns.barplot(x=type_counts.index, y=type_counts.values, palette=colors)
plt.title("Content Type Distribution - IQR Anomaly Detection")
plt.axhline(y=upper_bound, color='red', linestyle='--', label='Upper Bound')
plt.axhline(y=lower_bound, color='orange', linestyle='--', label='Lower Bound')
plt.legend()
plt.savefig("chart_12_iqr_anomalies.png")
plt.show()

# 13. Genre Distribution - Statistical Anomalies

all_genres = []
for genres in df['listed_in'].dropna():
    all_genres.extend([g.strip() for g in genres.split(',')])

genre_counts = Counter(all_genres)
genre_series = pd.Series(dict(genre_counts))

# IQR for genres
q1_genre = genre_series.quantile(0.25)
q3_genre = genre_series.quantile(0.75)
iqr_genre = q3_genre - q1_genre
upper_bound_genre = q3_genre + 1.5 * iqr_genre

dominant_genres = genre_series[genre_series >
                               upper_bound_genre].sort_values(ascending=False)

print(f"\nDominant Genres (Statistical Outliers):")
for genre, count in dominant_genres.items():
    z_score = (count - genre_series.mean()) / genre_series.std()
    print(f"  {genre}: {count} titles (Z-score: {z_score:.2f})")

plt.figure(figsize=(14, 6))
colors_genre = ['red' if x > upper_bound_genre else 'lightblue'
                for x in genre_series.values]
genre_series.sort_values().plot(kind='barh', color=colors_genre, figsize=(12, 8))
plt.title("Genre Distribution - Anomaly Detection (IQR Method)")
plt.xlabel("Count")
plt.axvline(x=upper_bound_genre, color='red',
            linestyle='--', label='Anomaly Threshold')
plt.legend()
plt.savefig("chart_13_genre_anomalies.png")
plt.tight_layout()
plt.show()

# 14. Country Analysis - Anomaly Detection
all_countries = []
for countries in df['country'].dropna():
    if countries != "Unknown Country":
        all_countries.extend([c.strip() for c in countries.split(',')])

country_counts = Counter(all_countries)
country_series = pd.Series(dict(country_counts))

# IQR for countries
q1_country = country_series.quantile(0.25)
q3_country = country_series.quantile(0.75)
iqr_country = q3_country - q1_country
upper_bound_country = q3_country + 1.5 * iqr_country

outlier_countries = country_series[country_series >
                                   upper_bound_country].sort_values(ascending=False)

print(f"\nDominant Countries (Anomaly Detection):")
for country, count in outlier_countries.items():
    z_score = (count - country_series.mean()) / country_series.std()
    print(f"  {country}: {count} titles (Z-score: {z_score:.2f})")

plt.figure(figsize=(12, 8))
top_20_countries = country_series.nlargest(20)
colors_country = ['red' if x > upper_bound_country else 'lightgreen'
                  for x in top_20_countries.values]
top_20_countries.plot(kind='barh', color=colors_country, figsize=(12, 8))
plt.title("Top 20 Countries - Anomaly Detection (IQR Method)")
plt.xlabel("Count")
plt.axvline(x=upper_bound_country, color='red',
            linestyle='--', label='Anomaly Threshold')
plt.legend()
plt.savefig("chart_14_countries_anomalies.png")
plt.tight_layout()
plt.show()

print("\n=== All visualizations saved successfully! ===")
