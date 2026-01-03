import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_titles.csv")
sns.set_theme(style="darkgrid")

yc = df["release_year"].value_counts().sort_index().reset_index()
yc.columns = ["year", "count"]

plt.figure(figsize=(12, 6))
sns.lineplot(data=yc, x="year", y="count", marker="o")
plt.title("Netflix Content Growth Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Amount of Content")
plt.show()


c = df["country"].dropna().astype(str).str.split(", ").explode().value_counts().head(10).to_frame()

plt.figure(figsize=(10, 8))
sns.heatmap(c, annot=True, fmt="d", cmap="Reds")
plt.title("Top 10 Countries with Most Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()


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
plt.show()


t = df["type"].value_counts().reset_index()
t.columns = ["type", "count"]

plt.figure(figsize=(8, 5))
sns.barplot(data=t, x="type", y="count")
plt.title("Movies vs TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()


r = df["rating"].value_counts().head(12).reset_index()
r.columns = ["rating", "count"]

plt.figure(figsize=(12, 5))
sns.barplot(data=r, x="rating", y="count")
plt.title("Content Rating Distribution (Top 12)")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.show()


x = df[["rating", "type"]].dropna()
tab = pd.crosstab(x["rating"], x["type"])

plt.figure(figsize=(12, 6))
sns.heatmap(tab, annot=True, fmt="d", cmap="Blues")
plt.title("Rating vs Type Relationship")
plt.xlabel("Type")
plt.ylabel("Rating")
plt.show()


ad = pd.to_datetime(df["date_added"], errors="coerce")
m = ad.dt.month_name().value_counts()
order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
m = m.reindex(order).dropna().reset_index()
m.columns = ["month", "count"]

plt.figure(figsize=(12, 5))
sns.barplot(data=m, x="month", y="count")
plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


cs = df["cast"].dropna().astype(str).str.split(", ").explode()
top = cs.value_counts().head(20).reset_index()
top.columns = ["name", "count"]

plt.figure(figsize=(12, 8))
sns.barplot(data=top, y="name", x="count")
plt.title("Top 20 Most Frequent Actors & Actresses")
plt.xlabel("Number of Titles")
plt.ylabel("Actor / Actress")
plt.show()
