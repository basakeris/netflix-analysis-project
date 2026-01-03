"""
Generates charts for content growth, countries, genres, and ratings.
"""
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
c = c_data.astype(str).str.split(", ").explode().value_counts().head(10).to_frame()

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
# Using pre-processed 'month_added' column from clean data
order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
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
