# 🧠 SmartBiz-KG: Knowledge Graph & Graph-RAG Engine

SmartBiz-KG is a Python-based Knowledge Graph architecture integrated with **Graph-RAG** and **Neo4j** for supply chain risk analysis and automated decision impact reasoning.

## 🚀 Key Features

* **In-Memory Knowledge Graph (`NetworkX`):** Build, query, and trace paths across domain entities.
* **Graph-RAG Agent:** Combine graph topological context with LLMs for root-cause and downstream risk analysis.
* **Neo4j Integration:** Synchronize entities and triplets directly with an enterprise graph database using Cypher queries.
* **Interactive Visualization:** Generate dynamic HTML dashboards (`PyVis`) and static graphs (`Matplotlib`).

## 🛠️ Project Structure

* `networkx_triplets.py`: Core Knowledge Graph engine, Triplet structures, and PyVis visualizer.
* `llm_agent.py`: Graph-RAG agent with LLM integration and fallback mechanisms.
* `neo4j_manager.py`: Neo4j driver connection and Cypher query execution.
* `main.py`: End-to-end integration pipeline.

## 📦 Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/knowledge_graph.push](https://github.com/YOUR_USERNAME/knowledge_graph.push)
   cd knowledge_graph