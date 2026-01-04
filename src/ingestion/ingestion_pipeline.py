"""
Complete Data Ingestion Pipeline: Structured Chunks → Vector Database
Features: File-level deduplication, UUID-based tracking, chunk deletion on update
"""
import os
import re
import json
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a parsed chunk with metadata"""
    chunk_id: str  # UUID for this chunk
    file_uuid: str  # UUID of parent file (for deletion)
    act: str
    act_english: str
    published_date: str
    is_amendment: bool
    chapter: Optional[str]
    chapter_title: Optional[str]
    section: str
    section_title: str
    text: str
    source_file: str


class ChunkParser:
    """Parse structured markdown files into chunks"""
    
    CHUNK_START = re.compile(r'^--------START OF CHUNK (\d+)--------$')
    CHUNK_END = re.compile(r'^--------END OF CHUNK \d+--------$')
    
    @staticmethod
    def parse_file(file_path: str, file_uuid: str) -> List[Chunk]:
        """
        Parse a structured markdown file into chunks
        
        Args:
            file_path: Path to structured .md file
            file_uuid: UUID assigned to this file
            
        Returns:
            List of Chunk objects
        """
        logger.info(f"📄 Parsing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Find chunk start
            match = ChunkParser.CHUNK_START.match(line)
            if match:
                chunk_num = match.group(1)
                chunk_data = ChunkParser._parse_chunk_block(lines, i)
                
                if chunk_data:
                    # Create unique chunk ID (UUID)
                    chunk_id = str(uuid.uuid4())
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        file_uuid=file_uuid,
                        source_file=os.path.basename(file_path),
                        **chunk_data
                    )
                    chunks.append(chunk)
            
            i += 1
        
        logger.info(f"✅ Parsed {len(chunks)} chunks from {os.path.basename(file_path)}")
        return chunks
    
    @staticmethod
    def _parse_chunk_block(lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a single chunk block"""
        metadata = {}
        hierarchy = {}
        text_lines = []
        
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            
            # End of chunk
            if ChunkParser.CHUNK_END.match(line.strip()):
                break
            
            # Parse metadata section
            if line.strip().startswith('metadata:'):
                i += 1
                while i < len(lines) and lines[i].startswith('  '):
                    key_val = lines[i].strip().split(':', 1)
                    if len(key_val) == 2:
                        key = key_val[0].strip()
                        val = key_val[1].strip()
                        # Convert boolean strings
                        if val == 'True':
                            val = True
                        elif val == 'False':
                            val = False
                        metadata[key] = val
                    i += 1
                continue
            
            # Parse hierarchy section
            if line.strip().startswith('hierarchy:'):
                i += 1
                while i < len(lines) and lines[i].startswith('  '):
                    key_val = lines[i].strip().split(':', 1)
                    if len(key_val) == 2:
                        key = key_val[0].strip()
                        val = key_val[1].strip()
                        hierarchy[key] = val
                    i += 1
                continue
            
            # Parse text section
            if line.strip().startswith('text: |'):
                i += 1
                while i < len(lines):
                    text_line = lines[i]
                    if ChunkParser.CHUNK_END.match(text_line.strip()):
                        break
                    # Remove leading indentation (2 spaces)
                    if text_line.startswith('  '):
                        text_lines.append(text_line[2:])
                    else:
                        text_lines.append(text_line)
                    i += 1
                break
            
            i += 1
        
        if not metadata or not hierarchy or not text_lines:
            return None
        
        return {
            'act': metadata.get('act'),
            'act_english': metadata.get('act_english'),
            'published_date': metadata.get('published_date'),
            'is_amendment': metadata.get('is_amendment', False),
            'chapter': hierarchy.get('chapter'),
            'chapter_title': hierarchy.get('chapter_title'),
            'section': hierarchy.get('section'),
            'section_title': hierarchy.get('section_title'),
            'text': '\n'.join(text_lines).strip()
        }


class FileTracker:
    """Track uploaded files using MD5 hash and UUID"""
    
    def __init__(self, tracker_path: str):
        self.tracker_path = tracker_path
        self.data = self._load_tracker()
        logger.info(f"📊 File Tracker loaded: {len(self.data.get('files', {}))} files")
    
    def _load_tracker(self) -> Dict:
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load tracker: {e}")
        
        return {"files": {}, "last_updated": None}
    
    def _save_tracker(self):
        self.data["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
        with open(self.tracker_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get file info if exists, None if not uploaded yet
        
        Returns:
            {
                "file_uuid": "abc-123-...",
                "file_hash": "md5hash...",
                "file_name": "file.md",
                "chunk_count": 30,
                "uploaded_at": "2026-01-05T14:30:00"
            }
        """
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        # Check if this file exists with same hash
        if file_name in self.data["files"]:
            stored_info = self.data["files"][file_name]
            if stored_info["file_hash"] == file_hash:
                return stored_info
        
        return None
    
    def is_file_changed(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if file has changed
        
        Returns:
            (has_changed, old_uuid)
            - (False, uuid) if file unchanged (already uploaded)
            - (True, uuid) if file changed (need to delete old chunks)
            - (True, None) if file is new
        """
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        if file_name not in self.data["files"]:
            # New file
            logger.debug(f"New file detected: {file_name}")
            return (True, None)
        
        stored_info = self.data["files"][file_name]
        if stored_info["file_hash"] == file_hash:
            # File unchanged
            logger.debug(f"File unchanged: {file_name}")
            return (False, stored_info["file_uuid"])
        else:
            # File changed
            logger.debug(f"File changed: {file_name}")
            return (True, stored_info["file_uuid"])
    
    def register_file(self, file_path: str, file_uuid: str, chunk_count: int):
        """Register a file as uploaded"""
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        self.data["files"][file_name] = {
            "file_uuid": file_uuid,
            "file_hash": file_hash,
            "file_name": file_name,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "uploaded_at": datetime.now().isoformat()
        }
        self._save_tracker()
        logger.debug(f"Registered file: {file_name} (UUID: {file_uuid})")
    
    def get_statistics(self) -> Dict:
        return {
            "total_files": len(self.data.get("files", {})),
            "last_updated": self.data.get("last_updated")
        }


class IngestionPipeline:
    """Main ingestion pipeline: Chunks → Embeddings → Vector DB"""
    
    def __init__(
        self,
        embed_model,
        vector_db,
        index_name: str,
        structured_dir: str,
        tracker_path: str,
        batch_size: int = 100
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            embed_model: Embedding model instance
            vector_db: Vector database instance
            index_name: Name of the vector index
            structured_dir: Directory with structured chunk files
            tracker_path: Path to upload tracker
            batch_size: Batch size for uploads
        """
        self.embed_model = embed_model
        self.vector_db = vector_db
        self.index_name = index_name
        self.structured_dir = structured_dir
        self.tracker = FileTracker(tracker_path)
        self.batch_size = batch_size
        
        logger.info("✅ IngestionPipeline initialized")
        logger.info(f"📂 Structured directory: {structured_dir}")
        logger.info(f"📊 Index: {index_name}")
        logger.info(f"📦 Batch size: {batch_size}")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single structured file
        
        Args:
            file_path: Path to structured markdown file
            
        Returns:
            Processing statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing: {os.path.basename(file_path)}")
        logger.info(f"{'='*60}")
        
        try:
            # Check if file has changed
            has_changed, old_uuid = self.tracker.is_file_changed(file_path)
            
            if not has_changed:
                logger.info(f"⏭️ File unchanged, skipping (UUID: {old_uuid})")
                return {
                    "status": "skipped",
                    "reason": "unchanged"
                }
            
            # File is new or changed
            if old_uuid:
                logger.info(f"🔄 File changed, deleting old chunks (UUID: {old_uuid})")
                self._delete_chunks_by_file_uuid(old_uuid)
            else:
                logger.info(f"🆕 New file detected")
            
            # Generate new UUID for this file
            file_uuid = str(uuid.uuid4())
            logger.info(f"🆔 Assigned UUID: {file_uuid}")
            
            # Parse chunks
            chunks = ChunkParser.parse_file(file_path, file_uuid)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks found in file")
                return {
                    "status": "failed",
                    "error": "No chunks found"
                }
            
            # Generate embeddings
            logger.info("🔢 Generating embeddings...")
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embed_model.embed_documents(texts)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Prepare for upload
            ids = [chunk.chunk_id for chunk in chunks]
            metadata = [self._create_metadata(chunk) for chunk in chunks]
            
            # Upload in batches
            logger.info(f"📤 Uploading to vector database (batch_size={self.batch_size})...")
            
            uploaded_count = 0
            for i in range(0, len(chunks), self.batch_size):
                batch_end = min(i + self.batch_size, len(chunks))
                
                batch_ids = ids[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_metadata = metadata[i:batch_end]
                
                success = self.vector_db.upsert(
                    collection_name=self.index_name,
                    vectors=batch_embeddings,
                    ids=batch_ids,
                    metadata=batch_metadata
                )
                
                if success:
                    uploaded_count += len(batch_ids)
                    logger.info(f"✅ Batch {i//self.batch_size + 1}: Uploaded {len(batch_ids)} chunks ({uploaded_count}/{len(chunks)} total)")
                else:
                    logger.error(f"❌ Batch {i//self.batch_size + 1} failed")
                    return {
                        "status": "failed",
                        "error": "Batch upload failed"
                    }
            
            # Register file in tracker
            self.tracker.register_file(file_path, file_uuid, len(chunks))
            logger.info(f"✅ File registered: {len(chunks)} chunks uploaded")
            
            return {
                "status": "success",
                "file_uuid": file_uuid,
                "chunks": len(chunks),
                "uploaded": uploaded_count,
                "was_update": old_uuid is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _delete_chunks_by_file_uuid(self, file_uuid: str):
        """Delete all chunks belonging to a file UUID"""
        try:
            # Use metadata filter to delete chunks with this file_uuid
            deleted = self.vector_db.delete(
                collection_name=self.index_name,
                filter={"file_uuid": file_uuid}
            )
            logger.info(f"🗑️ Deleted old chunks for UUID: {file_uuid}")
        except Exception as e:
            logger.error(f"⚠️ Failed to delete old chunks: {e}")
            # Continue anyway - will just have duplicates which can be cleaned later
    
    def _create_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Create metadata for vector database"""
        metadata = {
            "file_uuid": chunk.file_uuid,  # IMPORTANT: for deletion
            "content": chunk.text,
            "act": chunk.act,
            "act_english": chunk.act_english,
            "published_date": chunk.published_date,
            "is_amendment": chunk.is_amendment,
            "section": chunk.section,
            "section_title": chunk.section_title,
            "source_file": chunk.source_file
        }
        
        # Add optional fields
        if chunk.chapter:
            metadata["chapter"] = chunk.chapter
        if chunk.chapter_title:
            metadata["chapter_title"] = chunk.chapter_title
        
        return metadata
    
    def process_directory(self) -> Dict[str, Any]:
        """
        Process all structured files in directory
        
        Returns:
            Overall statistics
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 Starting Vector Database Ingestion Pipeline")
        logger.info("="*70)
        
        # Get all structured markdown files
        md_files = list(Path(self.structured_dir).glob("*_chunks.md"))
        logger.info(f"📁 Found {len(md_files)} structured files")
        
        if not md_files:
            logger.warning("⚠️ No structured files found")
            return {"processed": 0, "total_chunks": 0, "uploaded": 0, "failed": 0}
        
        stats = {
            "processed": 0,
            "skipped": 0,
            "updated": 0,
            "failed": 0,
            "total_chunks": 0
        }
        
        for i, file_path in enumerate(md_files, 1):
            logger.info(f"\n[{i}/{len(md_files)}] {file_path.name}")
            
            result = self.process_file(str(file_path))
            
            if result["status"] == "success":
                if result.get("was_update"):
                    stats["updated"] += 1
                else:
                    stats["processed"] += 1
                stats["total_chunks"] += result["chunks"]
            elif result["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("📊 Ingestion Summary")
        logger.info("="*70)
        logger.info(f"🆕 New files: {stats['processed']}")
        logger.info(f"🔄 Updated files: {stats['updated']}")
        logger.info(f"⏭️  Skipped: {stats['skipped']} (unchanged)")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"📦 Total Chunks: {stats['total_chunks']}")
        
        # Tracker stats
        tracker_stats = self.tracker.get_statistics()
        logger.info(f"📊 Database: {tracker_stats['total_files']} files tracked")
        logger.info("="*70)
        
        return stats


# CLI Entry Point
if __name__ == "__main__":
    import sys
    
    # Dynamic path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.core.secrets import SecretManager
    from src.core.embed_factory import EmbeddingFactory
    from src.core.vector_factory import VectorStoreFactory
    
    # Load configuration
    secrets = SecretManager()
    
    # Get configuration
    structured_dir = secrets.get_from_env("STRUCTURED_DATA_PATH", default="data/structured")
    pinecone_key = secrets.get_from_env("PINECONE_API_KEY")
    index_name = secrets.get_from_env("PINECONE_INDEX_NAME", default="vidhi-ai-legal-index")
    
    # Resolve paths
    structured_dir = os.path.join(project_root, structured_dir)
    tracker_path = os.path.join(project_root, "data", "vector_upload_tracker.json")
    
    # Validate
    if not pinecone_key:
        logger.error("❌ PINECONE_API_KEY not found in .env")
        sys.exit(1)
    
    if not os.path.exists(structured_dir):
        logger.error(f"❌ Structured directory not found: {structured_dir}")
        logger.info("💡 Please run postprocessor first")
        sys.exit(1)
    
    try:
        # Initialize embedding model
        logger.info("📦 Loading embedding model...")
        embed_model = EmbeddingFactory.create_embedding_model(
            provider="pinecone",
            api_key=pinecone_key,
            model_name="multilingual-e5-large"
        )
        
        # Initialize vector database
        logger.info("📦 Connecting to vector database...")
        vector_db = VectorStoreFactory.create(
            provider="pinecone",
            api_key=pinecone_key,
            cloud="aws",
            region="us-east-1"
        )
        
        # Check if index exists, create if needed
        if not vector_db.collection_exists(index_name):
            logger.info(f"🔨 Creating index: {index_name}")
            vector_db.create_collection(
                name=index_name,
                dimension=1024,  # multilingual-e5-large dimension
                metric="cosine"
            )
        
        # Initialize pipeline
        pipeline = IngestionPipeline(
            embed_model=embed_model,
            vector_db=vector_db,
            index_name=index_name,
            structured_dir=structured_dir,
            tracker_path=tracker_path,
            batch_size=100
        )
        
        # Process all files
        stats = pipeline.process_directory()
        
        # Exit with appropriate code
        sys.exit(0 if stats["failed"] == 0 else 1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
