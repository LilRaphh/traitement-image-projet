from fastapi import FastAPI
from src.api.routes import analysis, preprocessing, segmentation, comparison, pipeline

app = FastAPI(
    title="Deforestation Detector API",
    description="Pipeline de détection de déforestation par analyse d'images satellites.",
    version="0.1.0",
)

app.include_router(analysis.router, prefix="/analyze", tags=["Analyse initiale"])
app.include_router(preprocessing.router, prefix="/preprocess", tags=["Prétraitement"])
app.include_router(segmentation.router, prefix="/segment", tags=["Segmentation"])
app.include_router(comparison.router, prefix="/compare", tags=["Comparaison"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline complet"])


@app.get("/")
def root():
    return {"message": "Deforestation Detector API — voir /docs pour la documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)
