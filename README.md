# 🧠 FeedShift: Rewiring your feed

FeedShift is an intelligent recommendation engine that transforms noisy social feeds into personalized, semantically relevant, and high-quality content streams — even with minimal user history.

> 🚀 "From chaos to clarity — one semantically-aware recommendation at a time."

---

## 🔧 Key Features

### 🧠 Core Recommendation Engine
- 💡 **Cold-start personalization** using interest-aware semantic filtering
- 🧬 **Advanced embeddings** (e.g., Sentence Transformers) for meaningful content representation
- 🎯 **Multi-factor scoring**:
  - Semantic similarity to user interests
  - Time decay for freshness
  - Toxicity filtering (Detoxify-based)
  - Uniqueness via clustering
  - User-controlled diversity

### 📥 Real-time Reddit Ingestion (v0.6)
- ⛓️ Pulls latest content from selected subreddits
- 🛠️ Uses modular `EngineFactory` for dynamic platform loading
- 🗂️ Outputs in clean tabular format for downstream use

### 🎛️ Reflex Dashboard Overhaul
- 🧭 No more CSV upload — fetch content directly via live subreddit inputs!
- 🎚️ Interactive controls:
  - Toxicity strictness
  - Diversity strength
  - Interest selection
- 📊 Live top-10 recommendation preview panel
- 💨 Debounced updates for snappy UX

---

## 📁 Input Format (for developers)

The internal recommendation engine expects tabular input with the following columns (Reddit format):

| Column Name    | Type      | Description                        |
|----------------|-----------|------------------------------------|
| `id`           | string    | Unique post ID                     |
| `title`        | string    | Post title                         |
| `selftext`     | string    | Post body                          |
| `author`       | string    | Reddit username                    |
| `score`        | integer   | Upvotes                            |
| `created_utc`  | datetime  | Unix timestamp                     |
| `subreddit`    | string    | Subreddit category                 |
| `url`          | string    | Post URL                           |
| `num_comments` | integer   | Number of comments                 |
| `upvote_ratio` | float     | Ratio of upvotes (0-1)             |

---

## 🚀 Quickstart

1. **Clone the repo**
```bash
git clone https://github.com/amitjoshi9627/FeedShift.git
cd FeedShift
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Reflex dashboard**

```bash
reflex run
```

4. **Use the Dashboard**

* Select platform (currently supports `Reddit`)
* Choose your subreddits
* Adjust sliders to customize recommendations

---

## 🧪 Version Progress

### ✅ **v0.6 - Reddit Real-time Feed Integration**

* ✅ Reddit ingestion via live API
* ✅ Subreddit selector UI
* ✅ Modular engine loading via factory pattern
* ✅ Dashboard revamp: No file upload needed

### 🧩 Previous: v0.5

* ✅ Cold-start recommendations with embeddings
* ✅ Toxicity-aware filtering
* ✅ Diversity & uniqueness scoring
* ✅ CSV upload-based input

---

## 📅 Upcoming Roadmap

| Version     | Features                                                 |
| ----------- | -------------------------------------------------------- |
| 🔜 **v0.7** | Embedding generation + vector search for similar content |
| 🔜 **v0.8** | User profile memory + feedback loop (👍👎)               |
| 🔜 **v0.9** | Improved diversity + smarter toxicity filtering          |
| 🚀 **v1.0** | Production-ready API, CI/CD pipeline, and packaging      |

---

## 🛠️ Dev Setup

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Reflex frontend
reflex run
```

---

## ✅ Tech Stack

* ⚙️ **Backend**: Python · Reddit API · Pandas
* 📈 **Ranking**: Sentence Transformers · Detoxify · Clustering
* 🖥️ **Frontend**: Reflex (Next.js-like UI in Python)
* 🧪 **Testing**: Pytest (in-progress)
* 🚀 **Future**: FastAPI · FAISS · Qdrant · Chrome Extension

---

## 🧪 In Progress: Testing & Production Readiness

* [ ] Unit tests for ingestion, ranking, vectorization
* [ ] Integration tests with live endpoints
* [ ] CI/CD with GitHub Actions
* [ ] Auto-docs via `pdoc` or `mkdocs`
* [ ] Packaging with `pyproject.toml`

---

## 🤝 Contributing

Open to contributors! Start by opening an issue or suggesting a feature.

---

## 📜 License

MIT License. See `LICENSE` for full terms.

---

## ✨ Made with passion by [Amit Joshi](https://github.com/amitjoshi9627)

