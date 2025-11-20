# Hacker News Analysis Suite

A complete data pipeline for collecting, storing, and analyzing the entire history of Hacker News. This project combines high-performance data scraping with powerful analytical tools to uncover trends in programming languages, popular topics, and influential users.

## 🎯 Project Components

This suite consists of three main components:

### 1. **[Data Scraper](Scraper/Scraper_README.md)** 
High-performance parallel scraper that downloads all Hacker News items into PostgreSQL.
- **What it does:** Downloads stories, comments, and other items from the Hacker News API
- **Key features:** Parallel processing, fault tolerance, resumable downloads
- **Output:** Complete HN database with 40M+ items

### 2. **[Analysis Tools](Analysis/README.md)**
Suite of analytical tools for extracting insights from the data.
- **Temporal Analysis:** Track keyword frequency over time (Flask web app)
- **Topic Modeling:** Discover popular topics using BERTopic (ML-based)
- **User Influence:** Identify top contributors and analyze posting patterns

---

## 🚀 Quick Start

### Step 1: Scrape the Data
```bash
cd scraper
docker-compose up -d
python dispatcher.py
```
⏱️ Takes ~3-20 hours to download full history. You can read the [Data Scraper README](Scraper/Scraper_README.md) for more details.

### Step 2: Run Analysis
```bash
cd ../analysis

# Temporal keyword analysis (web interface)
cd temporal
python app.py

# Topic modeling
cd ../topics
python bertopic_analysis.py --days 365 --max-items 5000

# User influence ranking
cd ../users
python top_users.py --limit 100
```

---

## What You Can Discover

- **Programming Language Trends:** Which languages are gaining/losing popularity?
- **Topic Evolution:** What topics dominate HN discussions?
- **Influential Users:** Who are the most impactful contributors?
- **Temporal Patterns:** How do discussions change over time?
- **Community Dynamics:** Posting patterns, engagement metrics, and more!

---

## 📋 Prerequisites

- **Docker** - For PostgreSQL database
- **Python 3.10+** - For all scripts
- **50-100 GB disk space** - For complete dataset
- **8+ GB RAM** - Recommended for analysis tools

---

## 📖 Documentation

- **[Scraper Setup](scraper/Scaper_README.md)** - Detailed installation and configuration
- **[Analysis Guide](analysis/README.md)** - How to use each analysis tool
- **[Database Schema](database/schema.md)** - Database structure reference


## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---



## 🙏 Acknowledgments

- Hacker News team for the free, open API!
- BERTopic and sentence-transformers communities!

---

## 📚 References

- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [PostgreSQL](https://www.postgresql.org/)
- [DBeaver](https://dbeaver.io/)
- [Docker](https://www.docker.com/)