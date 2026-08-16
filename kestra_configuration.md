## Configuration

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

The project uses Kestra's environment-backed secret convention. Store the ROIC key as a base64-encoded value:

```dotenv
SECRET_ROIC_API_KEY=<base64-encoded-roic-api-key>
PG_CONN_STR_DOCKER=postgresql://raguser:ragpass@host.docker.internal:5432/ragdb
```

To encode a key locally:

```bash
printf %s "YOUR_ROIC_API_KEY" | base64 | tr -d '\n'
echo
```

Do not commit `.env` or share API keys. Add `.env` to `.gitignore` and restrict local permissions:


## Run the ingestion workflow

Before executing the Kestra flow, build the local ingestion image:

```bash
docker build -f Dockerfile.ingestion -t earnings-rag-ingestion:latest .
```

Start the RAG PostgreSQL database using the repository's database setup, then initialize the schema if needed.

Start Kestra:

```bash
docker compose up -d
```

Open `http://localhost:8080`, create a flow in the `earnings_rag` namespace, and paste or import `earnings_calls_ingest.yaml`.

Run the `ingest_batch` task from Kestra. A successful run logs the selected ticker and the number of inserted transcript chunks.
```
