from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from predict import MutationPredictor
import os
import logging
from typing import List
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    filename=os.getenv("LOG_FILE", "app.log"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Could not validate API key"
        )
    return api_key

# Initialize FastAPI app
app = FastAPI(
    title="Personalized Medicine Classifier",
    description="API for classifying genetic mutations based on clinical text evidence.",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# Request model
class MutationRequest(BaseModel):
    gene: str
    variation: str
    text: str

# Initialize predictor
model_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
predictor = MutationPredictor(model_dir)

@app.post("/predict")
async def predict_mutation(
    request: MutationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict the class of a genetic mutation.
    
    Args:
        request: MutationRequest containing gene, variation, and text
        api_key: API key for authentication
        
    Returns:
        Prediction results including class probabilities
    """
    logger.info(f"Processing prediction request for gene: {request.gene}")
    try:
        result = predictor.predict(
            gene=request.gene,
            variation=request.variation,
            text=request.text
        )
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info(f"Successfully predicted class {result['predicted_class']} for gene {request.gene}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)