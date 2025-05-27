# Dewey 🗂️

**Intelligent GitHub Repository Clustering and Visualization**

Dewey is a sophisticated tool that automatically discovers, analyzes, and visualizes patterns in GitHub repositories using machine learning. It fetches repositories from a GitHub user, generates AI-powered summaries, creates semantic embeddings, clusters similar repositories, and produces interactive 3D/2D visualizations.

## 🎯 What Dewey Does

1. **Repository Discovery**: Fetches all public repositories from a specified GitHub user
2. **AI-Powered Summarization**: Uses local LLMs (via Ollama) to generate intelligent summaries of each repository
3. **Semantic Analysis**: Creates embeddings using sentence transformers to capture repository meaning
4. **Intelligent Clustering**: Groups similar repositories using UMAP dimensionality reduction and HDBSCAN clustering
5. **Interactive Visualization**: Generates beautiful 3D/2D visualizations that let you explore repository relationships

## 🚀 Features

- **🤖 AI-Powered**: Uses Mistral LLM for repository summarization and cluster labeling
- **📊 Advanced ML Pipeline**: UMAP + HDBSCAN for high-quality clustering
- **🎨 Interactive Visualizations**: Plotly-based 3D scatter plots with hover information
- **⚡ Efficient**: Parallel processing for repository fetching and analysis
- **🔧 Configurable**: Extensive hyperparameter tuning options
- **💾 Persistent**: Caches embeddings and summaries for faster re-runs

## 📋 Requirements

- [UV](https://docs.astral.sh/uv/) - Modern Python package manager
- [Ollama](https://ollama.com/) - Local LLM runtime
- GitHub Personal Access Token
- Python 3.13+

## 🛠️ Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies
uv sync
```

### 2. Set Up Ollama

```bash
# Install and start Ollama
ollama serve

# Pull the Mistral model (in another terminal)
ollama pull mistral
```

### 3. Configure Environment

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_github_personal_access_token"
```

### 4. Run Dewey

```bash
# Run with default settings (analyzes joaofnds' repositories)
python main.py

# The visualization will be saved as cluster_visualization.html
```

## 🎛️ Configuration

Dewey offers extensive configuration options in `main.py`:

### Core Settings

- `USERNAME`: GitHub username to analyze
- `SUMMARY_LLM_MODEL`: LLM model for summarization (default: "mistral")
- `EMBEDDING_MODEL_NAME`: Sentence transformer model (default: "all-MiniLM-L6-v2")

### UMAP Parameters (Dimensionality Reduction)

- `UMAP_COMPONENTS`: Output dimensions (2 or 3)
- `UMAP_N_NEIGHBORS`: Local neighborhood size
- `UMAP_METRIC`: Distance metric ("cosine", "euclidean", etc.)
- `UMAP_MIN_DIST`: Minimum distance between points

### HDBSCAN Parameters (Clustering)

- `HDBSCAN_MIN_CLUSTER_SIZE`: Minimum cluster size
- `HDBSCAN_MIN_SAMPLES`: Core point threshold
- `HDBSCAN_EPSILON`: Distance threshold

## 📁 Project Structure

```
dewey/
├── main.py                 # Main execution script
├── lib/
│   ├── repo_fetcher.py     # GitHub API interaction
│   ├── generate_summaries.py # LLM-based summarization
│   ├── embbed.py           # Sentence embedding generation
│   ├── reducer.py          # UMAP dimensionality reduction
│   ├── clusterer.py        # HDBSCAN clustering
│   ├── labe_namer.py       # LLM-based cluster labeling
│   ├── ollama.py           # Ollama LLM client
│   └── viz.py              # Plotly visualization
├── data/
│   ├── repos/              # Cached repository data
│   ├── embeddings/         # Cached embeddings
│   └── labels/             # Cached cluster labels
└── cluster_visualization.html # Generated visualization
```

## 🔬 How It Works

1. **Data Collection**: Fetches repository metadata from GitHub API
2. **Content Analysis**: Generates summaries using repository README, description, and metadata
3. **Vectorization**: Creates semantic embeddings using sentence transformers
4. **Dimensionality Reduction**: Uses UMAP to reduce to 2D/3D while preserving structure
5. **Clustering**: Applies HDBSCAN to identify repository groups
6. **Labeling**: Uses LLM to generate meaningful cluster names
7. **Visualization**: Creates interactive plots showing repository relationships

## 🎨 Visualization Features

The generated HTML visualization includes:

- **Interactive 3D/2D scatter plot** of repository clusters
- **Hover information** showing repository names and summaries
- **Color-coded clusters** with meaningful labels
- **Zoom and pan** capabilities for detailed exploration

## 🔧 Customization

### Analyzing Different Users

```python
USERNAME = "your_target_username"
```

### Adjusting Clustering Sensitivity

```python
# For tighter clusters
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_EPSILON = 0.15

# For looser clusters
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_EPSILON = 0.35
```

### Using Different Models

```python
# Different embedding models
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Better quality, slower
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"  # Good balance

# Different LLM models (ensure they're available in Ollama)
SUMMARY_LLM_MODEL = "llama2"
LABLER_LLM_MODEL = "codellama"
```

## 📊 Example Use Cases

- **Portfolio Analysis**: Visualize patterns in your GitHub repositories
- **Technology Exploration**: Discover clusters of repositories by language or domain
- **Research**: Analyze open-source project ecosystems
- **Code Discovery**: Find similar projects to ones you're interested in

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error**

```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**GitHub API Rate Limiting**

- Ensure your `GITHUB_TOKEN` is set correctly
- The tool respects rate limits automatically

**Memory Issues with Large Datasets**

- Reduce the number of repositories or adjust worker count in `repo_fetcher.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

---

**Built with ❤️ using Python, UMAP, HDBSCAN, Sentence Transformers, and Ollama**
