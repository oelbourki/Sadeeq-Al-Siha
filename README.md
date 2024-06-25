# Medical RAG QA App using Meditron 7B LLM

This project is a Medical RAG (Retrieval-Augmented Generation) QA (Question-Answering) application using the Meditron 7B LLM (Language Learning Model), Qdrant Vector Database, and PubMedBERT Embedding Model. This app is designed to provide accurate and relevant medical information by leveraging state-of-the-art NLP (Natural Language Processing) and vector search technologies.

## Features

- **Meditron 7B LLM**: A powerful language model tailored for medical question answering.
- **Qdrant Vector Database**: Efficient and scalable vector search to store and retrieve embeddings.
- **PubMedBERT Embedding Model**: Embedding model specifically trained on medical literature for better relevance and accuracy.
- **Chainlit Interface**: User-friendly interface for interacting with the QA system.

## Installation

### Prerequisites

- Python 3.8+
- Docker
- NVIDIA GPU and Docker (if you want to utilize GPU)

### Setup

1. **Create a Python Virtual Environment**:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # For Linux/macOS
    .\.venv\Scripts\activate   # For Windows
    ```

2. **Install Python Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set Up Qdrant Vector Database**:
    ```sh
    docker pull qdrant/qdrant
    docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant --name qdrant
    ```

4. **Set Up Ollama for Meditron**:
    ```sh
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    # If using GPU:
    # docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```

5. **Pull Required Models**:
    ```sh
    docker exec -it ollama ollama pull meditron
    docker exec -it ollama ollama pull openchat
    ```

6. **Create Docker Network**:
    ```sh
    docker network create my_network
    docker network connect my_network qdrant
    docker network connect my_network ollama
    ```

7. **Build and Run the Chainlit App**:
    ```sh
    docker build -t my-chainlit-app .
    docker run --network my_network -p 8005:8005 my-chainlit-app
    ```

## Usage

Once the setup is complete, the Medical RAG QA app will be running on `http://localhost:8005`. You can interact with the application via a web interface, entering medical questions and receiving answers powered by the Meditron 7B LLM and the Qdrant Vector Database.

## Troubleshooting

- **Docker Network Issues**: Ensure all containers are connected to the `my_network`.
    ```sh
    docker network connect my_network qdrant
    docker network connect my_network ollama
    ```
- **Container Issues**: If a container fails to start, check the logs for errors.
    ```sh
    docker logs <container_name>
    ```
  
## Acknowledgments

- [Meditron 7B LLM](https://example.com)
- [Qdrant Vector Database](https://qdrant.tech/)
- [PubMedBERT](https://github.com/microsoft/BiomedNLP)
- [Chainlit](https://chainlit.io/)

---

Feel free to customize this README further to fit your specific needs and details!
