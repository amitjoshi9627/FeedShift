# 🧠 FeedShift: Rewiring Your Feed

FeedShift is a cold-start ranking engine designed to help users discover high-quality content in noisy social feeds — even with zero user history.

> 🚀 “From chaos to clarity — one post at a time.”

---

## 🔧 Features

- 📥 Upload raw social media CSV data
- 📊 Rank posts using content-based similarity (cosine similarity)
- 📝 View top 10 recommended posts in a Reddit/Facebook-style feed
- ❤️ Shows likes, authors, post IDs, and timestamps
- 💡 Modular design for future expansion:
  - User profile preferences
  - Toxicity filtering
  - Topic clustering (BERTopic)

---

## 📁 Input Format

The input should be a CSV with the following schema:

| Column Name | Type     | Description                         |
|-------------|----------|-------------------------------------|
| `post_id`   | string   | Unique post ID (`tw_123`, `rd_456`) |
| `platform`  | string   | Source platform (`twitter`, `reddit`, `blog`) |
| `text`      | string   | Post content                        |
| `author`    | string   | Author username (`@user123`)        |
| `likes`     | integer  | Number of likes                     |
| `timestamp` | datetime | ISO 8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) |
| `category`  | string   | Topic category (`AI`, `Health`, etc) |
| `has_image` | boolean  | Whether the post contains an image  |
| `language`  | string   | Language code (`en`, `es`, etc.)    |

---

## 🚀 Quickstart

1. **Clone the repo**
```bash
git clone https://github.com/your-username/feedshift.git
cd feedshift
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. **Upload your CSV and see the magic!** 🧙‍♂️

---

## 🔮 Roadmap

* [ ] v0.2 – Add user profile preferences (`e.g., prefers AI news`)
* [ ] v0.3 – Replace mock toxicity filter with  [Detoxify](https://github.com/unitaryai/detoxify)
* [ ] v0.4 – Use [BERTopic](https://github.com/MaartenGr/BERTopic) for topic-based ranking

---

## 🤝 Contributing

Pull requests and feedback are welcome! Open an issue or submit a PR.

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## ✨ Made with love by [Amit Joshi](https://github.com/amitjoshi9627)
