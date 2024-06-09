# Medical RAG-using-Meditron-7B-LLM
Medical RAG QA App using Meditron 7B LLM, Qdrant Vector Database, and PubMedBERT Embedding Model.

python -m venv .venv

pip install -r requiremnets.txt

docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant  --name qdrant

docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
or
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama ollama pull meditron
docker exec -it ollama ollama pull openchat



docker network create my_network

docker network connect my_network qdrant

docker network connect my_network ollama

docker build -t my-chainlit-app .

docker run --network my_network -p 8005:8005 my-chainlit-app
