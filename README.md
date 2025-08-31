# 🧠 FeedShift: Rewiring your feed

FeedShift is an intelligent recommendation engine that transforms noisy social feeds into personalized, semantically relevant, and high-quality content streams — even with minimal user history.

> 🚀 "From chaos to clarity — one semantically-aware recommendation at a time."

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

### 📥 Real-time Reddit Ingestion
- ⛓️ Pulls latest content from selected subreddits
- 🛠️ Uses modular `EngineFactory` for dynamic platform loading
- 🗂️ Hash-based duplicate detection for data integrity
- 🔄 Auto-updates with parameter changes

### 🎛️ Interactive Dashboard
- 🧭 Live subreddit content fetching (no CSV uploads needed!)
- 🎚️ Real-time controls:
  - Toxicity strictness slider
  - Diversity strength adjustment
  - Multi-select interest categories
- 📊 Instant top-10 recommendation preview
- 💨 Responsive UI with auto-updates

### ✅ Production-Ready Quality (v0.7)
- 🧪 **Comprehensive testing** with pytest and coverage reporting
- 🔍 **Automated code quality** via pre-commit hooks (Black, isort, flake8)
- 🚀 **CI/CD pipeline** with GitHub Actions (multi-OS testing)
- 📦 **Professional packaging** with pyproject.toml
- 📚 **Complete documentation** with type hints and docstrings

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

## 🚀 Quickstart

### 1. **Clone & Setup**
```bash
git clone https://github.com/amitjoshi9627/FeedShift.git
cd FeedShift

# Install in development mode with all tools
pip install -e ".[dev]"
```

### 2. **Set up Environment**
```bash
# Copy environment template
cp .env.example .env

# Add your Reddit API credentials to .env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

### 3. **Run the Application**
```bash
# Start the dashboard
reflex run

# Or run specific components
python -m src.app.home  # CLI version
```

### 4. **Development Workflow**
```bash
# Run tests
pytest

# Check code quality
pre-commit run --all-files

# Build package
python -m build
```

## 🧪 Version Progress

### ✅ **v0.7 - Production Readiness & Quality Assurance**

**🔧 Testing & Quality:**
- ✅ Comprehensive unit tests for all modules (database, models, ranking, engine)
- ✅ Integration tests for end-to-end workflows
- ✅ 80%+ test coverage with pytest-cov
- ✅ Automated code formatting (Black, isort, flake8)
- ✅ Pre-commit hooks for code quality enforcement

**🚀 CI/CD & Automation:**
- ✅ GitHub Actions pipeline with multi-OS testing (Ubuntu + macOS)
- ✅ Automated linting, testing, and packaging
- ✅ Coverage reporting with Codecov integration
- ✅ Branch protection rules and PR validation

**📦 Professional Packaging:**
- ✅ Modern pyproject.toml configuration
- ✅ Proper dependency management with optional dev dependencies
- ✅ Automated package building and validation
- ✅ Professional project structure and documentation

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

## 📅 Upcoming Roadmap

| Version     | Features                                                 |
| ----------- | -------------------------------------------------------- |
| 🔜 **v0.8** | User profile memory + feedback loop (👍👎)               |
| 🔜 **v0.9** | Multiple platform support (Twitter, HackerNews)          |
| 🚀 **v1.0** | Production API, advanced vectorization, Chrome extension |

## 🛠️ Development Setup

### **Prerequisites**
- Python 3.11+
- Git
- Reddit API credentials

### **Development Installation**
```bash
# Clone repository
git clone https://github.com/amitjoshi9627/FeedShift.git
cd FeedShift

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your Reddit API credentials
```

### **Available Commands**
```bash
# Testing
pytest                    # Run all tests
pytest --cov=src         # Run tests with coverage
pytest -m unit           # Run only unit tests
pytest -m integration    # Run only integration tests

# Code Quality
black src dashboard tests          # Format code
isort src dashboard tests          # Sort imports
flake8 src dashboard tests         # Check code quality
pre-commit run --all-files        # Run all checks

# Development
reflex run                # Start dashboard
python -m src.app.home    # Run CLI version

# Building
python -m build          # Build package
twine check dist/*       # Validate package
```

## ✅ Tech Stack

### **Core Technologies**
* ⚙️ **Backend**: Python 3.11+ · Reddit API (PRAW) · Pandas · NumPy
* 📈 **ML/AI**: Sentence Transformers · Detoxify · scikit-learn · Torch
* 🖥️ **Frontend**: Reflex (React-like UI in Python)
* 🗄️ **Data**: SQLite · CSV processing · Real-time ingestion

### **Development & Quality**
* 🧪 **Testing**: pytest · pytest-cov · pytest-mock
* 🔍 **Code Quality**: Black · isort · flake8 · pre-commit
* 🚀 **CI/CD**: GitHub Actions · Multi-OS testing · Automated deployment
* 📦 **Packaging**: pyproject.toml · setuptools · build · twine

### **Future Stack**
* 🚀 **API**: FastAPI · Pydantic · SQLAlchemy
* 🔍 **Vector Search**: FAISS · Qdrant · Pinecone
* 🌐 **Extension**: Chrome Extension · Browser integration

## 📊 Quality Metrics

### **Test Coverage**
- **Unit Tests**: Database operations, text processing, ranking algorithms
- **Integration Tests**: End-to-end recommendation pipeline
- **Coverage Target**: 80%+ maintained automatically via CI
- **Cross-Platform**: Tested on Ubuntu and macOS

### **Code Quality Standards**
- **Line Length**: 120 characters (modern standard)
- **Code Style**: Black formatter with consistent formatting
- **Import Organization**: isort with Black compatibility
- **Linting**: flake8 with complexity checking
- **Pre-commit**: Automatic quality checks before commits

### **CI/CD Pipeline**
- **Multi-OS Testing**: Ubuntu 22.04 + macOS Latest
- **Automated Quality Checks**: Linting, formatting, testing
- **Security Scanning**: Dependency vulnerability checks (planned)
- **Automated Packaging**: Build validation on main branch
- **Branch Protection**: Requires PR review and passing tests

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### **Quick Setup**
```bash
# Fork the repository
git clone https://github.com/your-username/FeedShift.git
cd FeedShift

# Set up development environment
pip install -e ".[dev]"
pre-commit install

# Make your changes and run tests
pytest
pre-commit run --all-files

# Submit a pull request
```

### **Contribution Guidelines**
- 🧪 **Add tests** for new functionality
- 📝 **Update documentation** for API changes
- ✅ **Ensure all checks pass** (tests, linting, formatting)
- 📋 **Follow conventional commit messages**
- 🔍 **Request review** for significant changes

### **Areas We Need Help**
- [ ] Additional platform integrations (Twitter, HackerNews)
- [ ] Advanced ML models for recommendation
- [ ] Performance optimizations
- [ ] UI/UX improvements
- [ ] Documentation and tutorials

## 📜 License

MIT License. See `LICENSE` for full terms.

## 🏆 Acknowledgments

- **Sentence Transformers** team for semantic embeddings
- **Detoxify** contributors for toxicity detection
- **Reflex** team for the amazing Python web framework
- **Reddit API** for providing accessible social data

## ✨ Created with passion by [Amit Joshi](https://github.com/amitjoshi9627)

### **Connect & Follow**
- 🐙 **GitHub**: [@amitjoshi9627](https://github.com/amitjoshi9627)

*⭐ Star this repo if FeedShift helps organize your digital content consumption!*
