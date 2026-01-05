# Netflix Data Analysis Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-013243?style=for-the-badge&logo=matplotlib&logoColor=white)

**Comprehensive analysis of Netflix catalog combining data cleaning, statistical analysis, and advanced visualizations**

</div>

---

## 📋 Overview

Netflix Data Analysis Project transforms raw Netflix catalog data into meaningful insights. The application processes datasets through a comprehensive pipeline: data cleaning, advanced statistical analysis, and anomaly detection, ultimately producing publication-ready visualizations.

**Key Features:**
- Multi-stage data cleaning pipeline with detailed documentation
- Advanced statistical analysis including distribution and anomaly detection
- Professional-grade visualizations with Seaborn and Matplotlib
- Robust error handling and reproducible workflows
- Modular architecture for independent component execution

---

## 🛠️ Technologies Used

```
Language              Python 3.7+
Data Processing       Pandas, NumPy
Statistical Analysis  SciPy, statsmodels
Visualization         Matplotlib, Seaborn
Testing              pytest
Data Format          CSV
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Steps

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/netflix-analysis-project.git
cd netflix-analysis-project
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add dataset:**
```bash
# Place netflix_titles.csv in the project root directory
cp /path/to/netflix_titles.csv .
```

---

## 🚀 Quick Start

### Run Complete Pipeline
```bash
python main.py
```
Executes all stages automatically: cleaning → analysis → visualization

### Run Individual Modules
```bash
python data_cleaning.py      # Preprocessing only
python advanced_analysis.py  # Analysis only
python visualizations.py     # Charts only
```

---

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  netflix_titles.csv (raw data)                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   data_cleaning.py         │
        │ ✓ Load & validate          │
        │ ✓ Handle missing values    │
        │ ✓ Standardize formats      │
        │ ✓ Normalize durations      │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │  netflix_cleaned.csv (clean)     │
        └────────────┬─────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────────┐  ┌────────────────────┐
│advanced_analysis.py │  │visualizations.py   │
│ • Distributions     │  │ • Growth charts    │
│ • Statistics        │  │ • Country heatmaps │
│ • Anomaly detect.   │  │ • Genre trends     │
│ • Temporal trends   │  │ • PNG exports      │
└─────────────────────┘  └────────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   Insights & Reports       │
        │ • Statistical summaries    │
        │ • Visualization files      │
        │ • Analysis documentation   │
        └────────────────────────────┘
```

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration and error handling |
| `data_cleaning.py` | Data loading, cleaning, normalization |
| `advanced_analysis.py` | Statistical analysis and anomaly detection |
| `visualizations.py` | Chart generation and export |
| `netflix_titles.csv` | Raw dataset (input) |
| `netflix_cleaned.csv` | Processed dataset (output) |
| `requirements.txt` | Python dependencies |

---

## 💡 Module Details

### data_cleaning.py
Implements a 6-step cleaning pipeline:
1. **Data Loading & Validation** - CSV import and integrity checks
2. **Missing Value Handling** - Intelligent NaN handling (director, cast, country, dates)
3. **Date Standardization** - Format conversion for temporal analysis
4. **Duration Normalization** - Movies (minutes) vs TV shows (seasons)
5. **Type Standardization** - Consistent content type values
6. **Statistical Reporting** - Logs removed rows and quality metrics

### advanced_analysis.py
Four-tier analysis framework:
- **Distribution Analysis** - Content types, ratings, genres, countries
- **Descriptive Statistics** - Mean, median, std dev, percentiles, skewness, kurtosis
- **Anomaly Detection** - IQR and Z-score outlier identification
- **Temporal Analysis** - Year-over-year trends, production patterns, rating changes

### visualizations.py
Three main charts generated:
1. **Content Growth** - Line plot of catalog expansion over years
2. **Top Countries** - Heatmap of production distribution
3. **Genre Evolution** - Temporal trends of main genres

---

## 📈 Use Cases

| Use Case | Command | Output |
|----------|---------|--------|
| Full Analysis | `python main.py` | Complete pipeline execution |
| Data Cleaning | `python data_cleaning.py` | Cleaned CSV ready for analysis |
| Generate Charts | `python visualizations.py` | PNG visualization files |
| Statistical Report | `python advanced_analysis.py` | Console analysis output |

---

## 📤 Generated Output

| File | Type | Purpose |
|------|------|---------|
| `netflix_cleaned.csv` | CSV | Processed dataset for analysis |
| `chart_1_growth.png` | PNG | Content growth timeline |
| `chart_2_countries.png` | PNG | Top 10 countries distribution |
| `chart_3_genres.png` | PNG | Genre trends over time |

---

## 🏆 Course Requirements & Extra Credit

<div align="center">

| Tier | Feature | Status | Points |
|------|---------|--------|--------|
| **Visualization** | Matplotlib + Seaborn | ✅ Implemented | +5 |
| **Dataset** | Messy data with documented cleaning | ✅ Implemented | +5 |
| **Analysis** | Advanced stats + Anomaly detection | ✅ Implemented | +15 |

</div>

**Cleaning Documentation:** See `data_cleaning.py` for step-by-step explanations of how messy data is transformed into analysis-ready datasets.

**Analysis Documentation:** See `advanced_analysis.py` for implementation of IQR/Z-score outlier detection and advanced statistical reporting.

---

## ⚠️ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: netflix_titles.csv` | Dataset missing from root | Place `netflix_titles.csv` in project root directory |
| `ModuleNotFoundError` | Missing dependencies | Run `pip install -r requirements.txt` |
| Charts not generating | matplotlib/seaborn issue | Verify installation: `pip install --upgrade matplotlib seaborn` |
| Analysis produces no output | Data not cleaned | Run `python data_cleaning.py` first |
| Script hangs on large dataset | Processing delay | Normal for 50K+ records (5-15 seconds typical) |

---

## 📝 Requirements

```
pandas>=1.3.0           # Data manipulation
numpy>=1.21.0           # Numerical computing
scipy>=1.7.0            # Statistical functions
matplotlib>=3.4.0       # Base visualization
seaborn>=0.11.0         # Statistical plots
pytest>=6.0.0           # Testing framework
```

---

## 📌 Notes

- Dataset must be placed as `netflix_titles.csv` in project root
- Requires Python 3.7 or higher for compatibility
- All charts exported as PNG for universal compatibility
- Console warnings suppressed for clean output
- All file paths use relative imports for portability

---

<div align="center">

Made with ❤️ to understand Netflix

[Report Issue](https://github.com/yourusername/netflix-analysis-project/issues) • [Documentation](./README.md)

</div>
