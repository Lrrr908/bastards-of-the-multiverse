from fastapi import FastAPI

app = FastAPI(title="BotMCMS API")

@app.get("/health")
def health():
    return {"status": "ok"}

# Later, when you're home:
# Paste your real LittleCMS + colormatch routes below this.
