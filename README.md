# To start docker
docker-compose up -d --build

# Then start mcp server
cd mcp
python3 mcp_server.py

# Run redis server
docker pull redis/redis-stack:latest
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  redis/redis-stack:latest