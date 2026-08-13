from fastapi import FastAPI

from blunderless import __version__

app = FastAPI(title="Blunderless", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
