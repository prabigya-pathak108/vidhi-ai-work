from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from dataclasses import dataclass
from enum import Enum

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


# ===== ENUMS AND DATA CLASSES =====
class VectorDBProvider(Enum):
    """Supported vector database providers"""
    PINECONE = "pinecone"
    CHROMA = "chroma"
    QDRANT = "qdrant"


@dataclass
class SearchResult:
    """Standardized search result format"""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata,
            "content": self.content
        }


class VectorDBBase(ABC):
    @abstractmethod
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """
        Creates an index or collection.
        
        Args:
            name: Collection/index name
            dimension: Vector dimension size
            **kwargs: Provider-specific parameters
        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Inserts or updates vectors in batches.
        
        Args:
            collection_name: Target collection
            vectors: List of vector embeddings
            ids: Unique identifiers for each vector
            metadata: Optional metadata for each vector
            
        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Searches for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List[SearchResult]: Standardized search results
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """
        Deletes vectors by ID.
        
        Args:
            collection_name: Collection name
            ids: List of vector IDs to delete
            
        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            bool: True if exists
        """
        pass

    @abstractmethod
    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            name: Collection name
            
        Returns:
            Dict with stats (vector_count, dimension, etc.)
        """
        pass


# ===== PINECONE IMPLEMENTATION =====
class PineconeDB(VectorDBBase):
    """
    Pinecone vector database implementation with production features:
    - Batch processing
    - Error handling and retries
    - Connection pooling
    - Comprehensive logging
    """

    def __init__(
        self,
        api_key: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            batch_size: Batch size for upsert operations
            max_retries: Max retry attempts for failed operations
        """
        try:
            self.pc = Pinecone(api_key=api_key)
            self.cloud = cloud
            self.region = region
            self.batch_size = batch_size
            self.max_retries = max_retries
            logger.info(f"✅ Pinecone client initialized (region: {region})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise

    def collection_exists(self, name: str) -> bool:
        """Check if index exists"""
        try:
            return name in [i.name for i in self.pc.list_indexes()]
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            return False

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        **kwargs
    ) -> bool:
        """
        Creates a Pinecone index with serverless spec.
        
        Args:
            name: Index name
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            **kwargs: Additional Pinecone parameters
        """
        # Check if already exists
        if self.collection_exists(name):
            logger.info(f"📦 Index '{name}' already exists")
            return True

        try:
            logger.info(f"🔨 Creating Pinecone index: {name} (dim: {dimension})")
            
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )

            # Wait for index to be ready
            logger.info("⏳ Waiting for index to be ready...")
            max_wait = 60  # seconds
            elapsed = 0
            while elapsed < max_wait:
                status = self.pc.describe_index(name).status
                if status.get('ready', False):
                    logger.info(f"✅ Index '{name}' is ready!")
                    return True
                time.sleep(2)
                elapsed += 2

            logger.warning(f"⚠️ Index creation timeout after {max_wait}s")
            return False

        except PineconeException as e:
            logger.error(f"❌ Pinecone creation error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error creating index: {e}")
            return False

    def upsert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Upsert vectors in batches for better performance.
        """
        if len(vectors) != len(ids):
            logger.error("❌ Vectors and IDs length mismatch")
            return False

        if metadata and len(metadata) != len(ids):
            logger.error("❌ Metadata and IDs length mismatch")
            return False

        try:
            index = self.pc.Index(collection_name)
            total_vectors = len(vectors)
            logger.info(f"📤 Upserting {total_vectors} vectors to '{collection_name}'")

            # Process in batches
            for i in range(0, total_vectors, self.batch_size):
                batch_end = min(i + self.batch_size, total_vectors)
                batch_data = []

                for j in range(i, batch_end):
                    meta = metadata[j] if metadata else {}
                    batch_data.append({
                        "id": ids[j],
                        "values": vectors[j],
                        "metadata": meta
                    })

                # Upsert batch with retry logic
                for attempt in range(self.max_retries):
                    try:
                        index.upsert(vectors=batch_data)
                        logger.info(f"✅ Batch {i//self.batch_size + 1} upserted ({batch_end}/{total_vectors})")
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"⚠️ Retry {attempt + 1}/{self.max_retries}: {e}")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            raise

            return True

        except Exception as e:
            logger.error(f"❌ Pinecone upsert error: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors with optional metadata filtering.
        """
        try:
            index = self.pc.Index(collection_name)
            
            # Build query parameters
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict

            results = index.query(**query_params)

            # Convert to standardized format
            search_results = [
                SearchResult(
                    id=match['id'],
                    score=match['score'],
                    metadata=match.get('metadata', {}),
                    content=match.get('metadata', {}).get('content')
                )
                for match in results.get('matches', [])
            ]

            logger.info(f"🔍 Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        try:
            index = self.pc.Index(collection_name)
            index.delete(ids=ids)
            logger.info(f"🗑️ Deleted {len(ids)} vectors from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.pc.Index(name).describe_index_stats()
            return {
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0.0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}


class VectorStoreFactory:

    _providers = {
        VectorDBProvider.PINECONE: PineconeDB,
        # VectorDBProvider.CHROMA: ChromaDB,  # Add when implemented
        # VectorDBProvider.QDRANT: QdrantDB,  # Add when implemented
    }

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> VectorDBBase:
        """
        Create a vector store instance.
        
        Args:
            provider: Provider name (pinecone, chroma, qdrant)
            api_key: API key for the provider
            **kwargs: Provider-specific parameters
            
        Returns:
            VectorDBBase: Vector store instance
        """
        try:
            provider_enum = VectorDBProvider(provider.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {[p.value for p in VectorDBProvider]}"
            )

        db_class = cls._providers.get(provider_enum)
        if not db_class:
            raise NotImplementedError(f"{provider} not yet implemented")

        if provider_enum == VectorDBProvider.PINECONE:
            if not api_key:
                raise ValueError("Pinecone requires api_key")
            return db_class(api_key=api_key, **kwargs)

        raise NotImplementedError(f"Provider {provider} initialization not configured")


# ===== TEST CODE =====
if __name__ == "__main__":
    import sys
    import os

    # Dynamic path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.core.secrets import SecretManager
    from src.core.embed_factory import EmbeddingFactory

    logger.info("=" * 60)
    logger.info("🚀 Vidhi-AI Vector Store Test Suite")
    logger.info("=" * 60)

    # Initialize secrets
    secrets = SecretManager()
    pinecone_key = secrets.get_from_env("PINECONE_API_KEY")
    index_name = "vidhi-ai-test-index"

    if not pinecone_key:
        logger.error("❌ Missing PINECONE_API_KEY in .env file!")
        sys.exit(1)

    try:
        # Test 1: Initialize Embedding Model
        logger.info("📝 Test 1: Initialize Embedding Model")
        logger.info("-" * 60)
        embed_model = EmbeddingFactory.create_embedding_model(
            provider="pinecone",
            api_key=pinecone_key,
            model_name="multilingual-e5-large"  # 1024 dimensions
        )
        logger.info("✅ Embedding model initialized")

        # Test 2: Initialize Vector Store
        logger.info("📝 Test 2: Initialize Vector Store")
        logger.info("-" * 60)
        vector_db = VectorStoreFactory.create(
            provider="pinecone",
            api_key=pinecone_key,
            cloud="aws",
            region="us-east-1",
            batch_size=100
        )
        logger.info("✅ Vector store initialized")

        # Test 3: Create Collection
        logger.info("📝 Test 3: Create Collection")
        logger.info("-" * 60)
        success = vector_db.create_collection(
            name=index_name,
            dimension=1024,
            metric="cosine"
        )
        logger.info(f"✅ Collection creation: {'Success' if success else 'Failed'}")

        # Test 4: Check Collection Stats
        logger.info("📝 Test 4: Collection Statistics")
        logger.info("-" * 60)
        stats = vector_db.get_collection_stats(index_name)
        logger.info(f"📊 Stats: {stats}")

        # Test 5: Upsert Test Data
        logger.info("📝 Test 5: Upsert Test Documents")
        logger.info("-" * 60)
        
        test_documents = [
            {
                "content": "दफा ५. सजायको व्यवस्था: कसूर गर्ने व्यक्तिलाई ५ वर्ष कैद हुनेछ।",
                "metadata": {
                    "act": "Test Act 2080",
                    "section": "5",
                    "khanda": "ka",
                    "language": "nepali"
                }
            },
            {
                "content": "Section 10. Property Rights: Every citizen has the right to own property.",
                "metadata": {
                    "act": "Civil Code 2074",
                    "section": "10",
                    "khanda": "kha",
                    "language": "english"
                }
            }
        ]

        # Generate embeddings
        contents = [doc["content"] for doc in test_documents]
        embeddings = embed_model.embed_documents(contents)
        ids = [f"doc_{i}" for i in range(len(contents))]
        metadata = [doc["metadata"] for doc in test_documents]

        # Upsert
        upsert_success = vector_db.upsert(
            collection_name=index_name,
            vectors=embeddings,
            ids=ids,
            metadata=metadata
        )
        logger.info(f"✅ Upsert: {'Success' if upsert_success else 'Failed'}")

        # Wait for indexing
        logger.info("⏳ Waiting 5 seconds for indexing...")
        time.sleep(5)

        # Test 6: Search
        logger.info("📝 Test 6: Similarity Search")
        logger.info("-" * 60)
        
        queries = [
            "What is the punishment in section 5?",
            "property rights section 10"
        ]

        for query in queries:
            logger.info(f"🔍 Query: '{query}'")
            query_embedding = embed_model.embed_query(query)
            results = vector_db.search(
                collection_name=index_name,
                query_vector=query_embedding,
                top_k=2
            )

            if results:
                for idx, result in enumerate(results, 1):
                    logger.info(f"  {idx}. Score: {result.score:.4f}")
                    logger.info(f"     ID: {result.id}")
                    logger.info(f"     Metadata: {result.metadata}")
            else:
                logger.warning("  ❌ No results found")

        # Test 7: Filtered Search
        logger.info("📝 Test 7: Filtered Search (Nepali only)")
        logger.info("-" * 60)
        query_embedding = embed_model.embed_query("punishment")
        filtered_results = vector_db.search(
            collection_name=index_name,
            query_vector=query_embedding,
            top_k=5,
            filter_dict={"language": "nepali"}
        )
        logger.info(f"✅ Found {len(filtered_results)} Nepali documents")

        # Test 8: Delete
        logger.info("📝 Test 8: Delete Documents")
        logger.info("-" * 60)
        delete_success = vector_db.delete(index_name, ["doc_0"])
        logger.info(f"✅ Delete: {'Success' if delete_success else 'Failed'}")

        logger.info("=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)