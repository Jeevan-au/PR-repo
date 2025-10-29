import logging
from typing import Dict, List, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import time

from app.core.config import QdrantConfig, settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._connection_params = QdrantConfig.get_connection_params()
        
    def get_client(self) -> QdrantClient:
        """Get or create Qdrant client instance"""
        if self._client is None:
            try:
                self._client = QdrantClient(**self._connection_params)
                logger.info(f"Qdrant client created: {QdrantConfig.get_connection_string()}")
            except Exception as e:
                logger.error(f"Failed to create Qdrant client: {str(e)}")
                raise
        return self._client
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check Qdrant connection status and return detailed information
        
        Returns:
            Dictionary with connection status, info, and collections
        """
        result = {
            "status": "disconnected",
            "connection_string": QdrantConfig.get_connection_string(),
            "error": None,
            "server_info": None,
            "collections": [],
            "response_time_ms": None
        }
        
        try:
            start_time = time.time()
            
            # Create client and test connection
            client = self.get_client()
            
            # Get server info
            server_info = client.get_collections()
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Get collections list
            collections = []
            if hasattr(server_info, 'collections'):
                collections = [
                    {
                        "name": collection.name,
                        "vectors_count": getattr(collection, 'vectors_count', 0),
                        "points_count": getattr(collection, 'points_count', 0)
                    }
                    for collection in server_info.collections
                ]
            
            result.update({
                "status": "connected",
                "server_info": {
                    "collections_count": len(collections),
                    "total_vectors": sum(col.get('vectors_count', 0) for col in collections),
                    "total_points": sum(col.get('points_count', 0) for col in collections)
                },
                "collections": collections,
                "response_time_ms": round(response_time, 2)
            })
            
            logger.info(f"Qdrant connection successful - {len(collections)} collections found")
            
        except UnexpectedResponse as e:
            error_msg = f"Qdrant server error: {str(e)}"
            result["error"] = error_msg
            logger.error(error_msg)
            
        except ConnectionError as e:
            error_msg = f"Connection failed: {str(e)}"
            result["error"] = error_msg
            logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            result["error"] = error_msg
            logger.error(error_msg)
            
        return result
    
    def get_collections_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all collections
        
        Returns:
            List of dictionaries with collection details
        """
        try:
            client = self.get_client()
            collections_response = client.get_collections()
            
            collections_info = []
            
            for collection in collections_response.collections:
                try:
                    # Get detailed collection info
                    collection_info = client.get_collection(collection.name)
                    
                    info = {
                        "name": collection.name,
                        "vectors_count": getattr(collection_info, 'vectors_count', 0),
                        "points_count": getattr(collection_info, 'points_count', 0),
                        "config": {
                            "vector_size": getattr(collection_info.config.params.vector, 'size', None),
                            "distance": getattr(collection_info.config.params.vector, 'distance', None),
                        } if hasattr(collection_info, 'config') else None,
                        "status": getattr(collection_info, 'status', 'unknown')
                    }
                    
                    collections_info.append(info)
                    
                except Exception as e:
                    logger.warning(f"Could not get detailed info for collection {collection.name}: {str(e)}")
                    collections_info.append({
                        "name": collection.name,
                        "error": str(e)
                    })
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Failed to get collections info: {str(e)}")
            raise
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ) -> Dict[str, Any]:
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors
            distance: Distance metric (Cosine, Euclid, Dot)
        
        Returns:
            Dictionary with creation result
        """
        try:
            client = self.get_client()
            
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper())
                )
            )
            
            result = {
                "success": True,
                "collection_name": collection_name,
                "vector_size": vector_size,
                "distance": distance,
                "message": f"Collection '{collection_name}' created successfully"
            }
            
            logger.info(f"Created collection: {collection_name}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to create collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def close(self):
        """Close the Qdrant client connection"""
        if self._client:
            try:
                self._client.close()
                self._client = None
                logger.info("Qdrant client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {str(e)}")


# Global service instance
qdrant_service = QdrantService()
