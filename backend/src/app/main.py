from fastapi import FastAPI
from src.app.api.v1.predict import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

    
app = FastAPI(
    title="PaySim Fraud Detection API",
    description="API for predicting fraudulent transactions using PaySim dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routes
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "🧠 PaySim Fraud Detection API is running!"}
