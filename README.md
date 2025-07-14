# 🧠 FeedShift: Rewiring your feed

FeedShift is an advanced recommendation engine that transforms noisy social feeds into personalized, high-quality content streams - even with minimal user history.

> 🚀 "From chaos to clarity - one semantically-aware recommendation at a time."

---

## 🔧 Key Features

### Core Ranking Engine
- 🧠 **Cold-start personalization** with interest-based recommendations
- 🧪 **Advanced embedding techniques** (Sentence Transformers) for semantic understanding
- 📊 **Multi-factor scoring**:
  - Interest relevance (semantic similarity)
  - Content freshness (time-decay)
  - Toxicity filtering (Detoxify integration)
  - Content uniqueness (cluster-based scoring)
  - Diversity control (user-tunable)

### Dashboard Features
- 📥 CSV upload with real-time processing
- 🎚️ Interactive controls:
  - Toxicity strictness slider
  - Diversity strength control
  - Interest selection panel
- 📊 Visual ranking of top 10 recommended posts
- ⚡ Real-time updates with debounced processing

### Architecture
- 🧩 Modular design with clear separation:
  - Embedding models
  - Ranking algorithms
  - UI components
- 🚀 Optimized for future expansion:
  - User profile persistence
  - Real-time APIs
  - Cross-platform support

---

## 📁 Input Format

The input should be a CSV with the following schema (Reddit-style):

| Column Name | Type     | Description                         |
|-------------|----------|-------------------------------------|
| `id`        | string   | Unique post ID                      |
| `title`     | string   | Post title                          |
| `selftext`  | string   | Post content                        |
| `author`    | string   | Author username                     |
| `score`     | integer  | Post score (upvotes)                |
| `created_utc` | datetime | UTC timestamp (Unix epoch)          |
| `subreddit` | string   | Subreddit/category                  |
| `url`       | string   | Post URL                            |
| `num_comments` | integer  | Number of comments                 |
| `upvote_ratio` | float    | Upvote ratio (0-1)                  |

---

## 🚀 Quickstart

1. **Clone the repo**
```bash
git clone -b V0.5/recommendations-improvement https://github.com/amitjoshi9627/FeedShift.git
cd FeedShift
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Reflex dashboard**
```bash
reflex run
```

4. **Access the dashboard** at `http://localhost:3000`

5. **Upload your CSV** and customize your feed!

---

## 🧩 V0.5 Features

- ✅ **Interest-based personalization** with semantic matching
- ✅ **Toxicity-aware filtering** with adjustable strictness
- ✅ **Content uniqueness scoring** using cluster-based algorithms
- ✅ **Diversity control** with smart similarity adjustments
- ✅ **Interactive dashboard** with real-time recommendations
- ✅ **Performance optimizations** for faster ranking

---

## 🔮 Roadmap

* [ ] v0.6 - Real-time API integration
* [ ] v0.7 - Chrome extension MVP
* [ ] v0.8 - Cross-platform user embeddings
* [ ] v0.9 - Personalized topic discovery
* [ ] v1.0 - Production deployment pipeline

---

## 🛠️ Development

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt

# Run with hot reloading
reflex run
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss proposed changes.

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## ✨ Made with passion by [Amit Joshi](https://github.com/amitjoshi9627)
