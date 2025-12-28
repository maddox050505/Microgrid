from fastapi import FastAPI

app = FastAPI(title="Microgrid API")

@app.get("/health")
def health():
    return {"ok": True}
