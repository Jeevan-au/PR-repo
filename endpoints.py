from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel

from app.services.qdrant_service import qdrant_service
from app.core.config import settings

# Create the main API router
router = APIRouter()

# Pydantic models for request/response
class CollectionCreate(BaseModel):
    name: str
    vector_size: int
    distance: str = "Cosine"

class ConnectionStatus(BaseModel):
    status: str
    connection_string: str
    error: str = None
    server_info: Dict[str, Any] = None
    collections: List[Dict[str, Any]] = []
    response_time_ms: float = None

@router.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {"message": "Welcome to RAG FastAPI Project"}

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get("/info")
async def api_info() -> Dict[str, Any]:
    """API information endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "A FastAPI application for RAG (Retrieval Augmented Generation) operations",
        "debug": settings.debug
    }

# Qdrant connection and management endpoints
@router.get("/qdrant/connection", response_model=ConnectionStatus, tags=["qdrant"])
async def check_qdrant_connection() -> Dict[str, Any]:
    """
    Check Qdrant vector database connection status
    
    Returns detailed information about:
    - Connection status
    - Server information
    - Available collections
    - Response time
    """
    try:
        result = qdrant_service.check_connection()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check Qdrant connection: {str(e)}"
        )

@router.get("/qdrant/collections", tags=["qdrant"])
async def get_qdrant_collections() -> Dict[str, Any]:
    """
    Get detailed information about all Qdrant collections
    
    Returns:
    - List of collections with metadata
    - Vector counts and configuration details
    """
    try:
        collections = qdrant_service.get_collections_info()
        return {
            "collections": collections,
            "total_collections": len(collections),
            "connection_string": qdrant_service._connection_params
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collections: {str(e)}"
        )

@router.post("/qdrant/collections", tags=["qdrant"])
async def create_qdrant_collection(collection_data: CollectionCreate) -> Dict[str, Any]:
    """
    Create a new Qdrant collection
    
    Parameters:
    - name: Collection name
    - vector_size: Size of vectors (e.g., 384 for all-MiniLM-L6-v2)
    - distance: Distance metric (Cosine, Euclid, Dot)
    """
    try:
        result = qdrant_service.create_collection(
            collection_name=collection_data.name,
            vector_size=collection_data.vector_size,
            distance=collection_data.distance
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection: {str(e)}"
        )

@router.get("/qdrant/status", tags=["qdrant"])
async def get_qdrant_status() -> Dict[str, Any]:
    """
    Get quick Qdrant status without detailed collection info (faster endpoint)
    """
    try:
        # Quick connection test
        client = qdrant_service.get_client()
        collections_response = client.get_collections()
        
        return {
            "status": "connected",
            "collections_count": len(collections_response.collections),
            "connection_string": qdrant_service._connection_params,
            "message": "Qdrant is running and accessible"
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "connection_string": qdrant_service._connection_params,
            "message": "Qdrant is not accessible"
        }
