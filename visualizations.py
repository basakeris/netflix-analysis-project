import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

yearly_counts = df["release_year"].value_counts().sort_index().reset_index()
yearly_counts.columns = ["year", "count"]

plt.figure(figsize=(12, 6))
sns.lineplot(x="year", y="count", data=yearly_counts, marker="o")
plt.title("Netflix Content Growth Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Amount of Content")
plt.show()


top_countries = df["country"].dropna().str.split(", ").explode().value_counts().head(10).to_frame()

plt.figure(figsize=(10, 8))
sns.heatmap(top_countries, annot=True, fmt="d", cmap="Reds")
plt.title("Top 10 Countries with Most Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

df_copy = df.copy()
df_copy["first_genre"] = df_copy["listed_in"].str.split(",").str[0]

top_8_genres = df_copy["first_genre"].value_counts().head(8).index
filtered_df = df_copy[df_copy["first_genre"].isin(top_8_genres)]

plt.figure(figsize=(12, 6))
sns.boxplot(x="first_genre", y="release_year", data=filtered_df)
plt.title("Release Year Distribution by Top Genres")
plt.xticks(rotation=45)
plt.xlabel("Genre")
plt.ylabel("Release Year")
plt.show()
