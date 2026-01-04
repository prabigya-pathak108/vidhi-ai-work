"""
Production-ready Legal Document Postprocessor
Features: Metadata extraction, text normalization, hierarchical parsing, chunking
"""
import os
import re
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Document-level metadata"""
    act_name_nepali: str
    act_name_english: str
    published_date: str
    is_amendment: bool
    amendment_info: Optional[str]
    act_number: Optional[str]
    source_file: str
    extracted_at: str


@dataclass
class ChunkHierarchy:
    """Hierarchical structure for a chunk"""
    act: str
    act_english: str
    part: Optional[str]
    part_title: Optional[str]
    chapter: Optional[str]
    chapter_title: Optional[str]
    section: Optional[str]
    section_title: Optional[str]


class MetadataExtractor:
    """Extract metadata from legal documents using LLM"""
    
    def __init__(self, llm):
        """
        Initialize metadata extractor
        
        Args:
            llm: Language model instance (from LLMFactory)
        """
        self.llm = llm
        logger.info("✅ MetadataExtractor initialized")
    
    def extract(self, text_sample: str, filename: str) -> DocumentMetadata:
        """
        Extract metadata from document text
        
        Args:
            text_sample: First ~2000 chars of document
            filename: Source filename
            
        Returns:
            DocumentMetadata object
        """
        logger.info(f"📋 Extracting metadata from: {filename}")
        
        prompt = f"""You are a Nepali legal document analyzer. Extract metadata from this legal act.

Document text (first 2000 chars):
{text_sample}

Extract the following information in JSON format:
{{
    "act_name_nepali": "Full Nepali name of the act",
    "act_name_english": "English translation of the act name",
    "published_date": "Publication date in format YYYY.MM.DD (e.g., 2074.6.27)",
    "is_amendment": true/false,
    "amendment_info": "If amendment, mention what it amends, else null",
    "act_number": "Act number (e.g., 'ऐन नं. १९') if mentioned, else null"
}}

Rules:
- Extract exact names as they appear
- For dates, use the Nepali calendar format if available (BS)
- If amendment, look for "संशोधन" keywords
- Output ONLY valid JSON, no explanations

JSON output:"""

        try:
            response = self.llm.invoke(prompt)
            
            # Clean JSON from response
            json_text = response.content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Parse JSON
            metadata_dict = json.loads(json_text)
            
            # Create metadata object
            metadata = DocumentMetadata(
                act_name_nepali=metadata_dict.get("act_name_nepali", "Unknown"),
                act_name_english=metadata_dict.get("act_name_english", "Unknown"),
                published_date=metadata_dict.get("published_date", "Unknown"),
                is_amendment=metadata_dict.get("is_amendment", False),
                amendment_info=metadata_dict.get("amendment_info"),
                act_number=metadata_dict.get("act_number"),
                source_file=filename,
                extracted_at=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Metadata extracted: {metadata.act_name_english}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            # Return default metadata
            return DocumentMetadata(
                act_name_nepali=Path(filename).stem,
                act_name_english=Path(filename).stem,
                published_date="Unknown",
                is_amendment=False,
                amendment_info=None,
                act_number=None,
                source_file=filename,
                extracted_at=datetime.now().isoformat()
            )


class TextNormalizer:
    """Normalize and clean OCR text"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text by removing artifacts and standardizing format
        
        Args:
            text: Raw text from parser
            
        Returns:
            Normalized text
        """
        logger.info("🧹 Normalizing text...")
        
        # Remove HTML-like tags
        text = re.sub(r'<ins>|</ins>|<sup>|</sup>', '', text)
        
        # Remove special markers
        text = re.sub(r'[◉✕❋⊗◇]', '', text)
        
        # Remove excessive underscores (used for emphasis)
        text = re.sub(r'_{2,}', '', text)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{4,}', '\n\n', text)  # Max 2 line breaks
        
        # Fix broken Nepali parentheses
        text = re.sub(r'\(\s+([१-९०]+)\s+\)', r'(\1)', text)
        text = re.sub(r'\(\s+([क-ह])\s+\)', r'(\1)', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text


class HierarchicalParser:
    """Parse document hierarchy (Act -> Part -> Chapter -> Section)"""
    
    # Regex patterns for hierarchy detection
    PATTERNS = {
        'act_title': re.compile(r'^#\s+(.+ऐन.+)$', re.MULTILINE),
        'part': re.compile(r'^##\s+भाग[–-]?\s*([१-९०]+|\S+)?(.*)$', re.MULTILINE | re.UNICODE),
        'chapter': re.compile(r'^##\s+परिच्छेद[–-]?\s*([१-९०]+|\d+)?(.*)$', re.MULTILINE | re.UNICODE),
        'section': re.compile(r'^(\d+)\.\s+(.+?):', re.MULTILINE),
        'subsection': re.compile(r'^\(([१-९०]+)\)', re.MULTILINE),
        'clause': re.compile(r'^\(([क-ह])\)', re.MULTILINE)
    }
    
    def __init__(self, metadata: DocumentMetadata):
        """
        Initialize parser with document metadata
        
        Args:
            metadata: Document metadata
        """
        self.metadata = metadata
        self.current_part = None
        self.current_part_title = None
        self.current_chapter = None
        self.current_chapter_title = None
        logger.info("🔍 HierarchicalParser initialized")
    
    def parse_sections(self, text: str) -> List[Tuple[ChunkHierarchy, str]]:
        """
        Parse text into sections with hierarchy
        
        Args:
            text: Normalized document text
            
        Returns:
            List of (hierarchy, section_text) tuples
        """
        logger.info("📄 Parsing document structure...")
        
        chunks = []
        lines = text.split('\n')
        
        current_section_num = None
        current_section_title = None
        section_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                if section_content:
                    section_content.append('')
                i += 1
                continue
            
            # Check for Part
            part_match = self.PATTERNS['part'].match(line)
            if part_match:
                # Save previous section
                if current_section_num and section_content:
                    chunk = self._create_chunk(
                        current_section_num,
                        current_section_title,
                        '\n'.join(section_content)
                    )
                    chunks.append(chunk)
                    section_content = []
                
                self.current_part = part_match.group(1) or "N/A"
                self.current_part_title = part_match.group(2).strip() if part_match.group(2) else None
                logger.info(f"  📂 Part: {self.current_part}")
                i += 1
                continue
            
            # Check for Chapter
            chapter_match = self.PATTERNS['chapter'].match(line)
            if chapter_match:
                # Save previous section
                if current_section_num and section_content:
                    chunk = self._create_chunk(
                        current_section_num,
                        current_section_title,
                        '\n'.join(section_content)
                    )
                    chunks.append(chunk)
                    section_content = []
                
                self.current_chapter = chapter_match.group(1) or "N/A"
                self.current_chapter_title = chapter_match.group(2).strip() if chapter_match.group(2) else None
                logger.info(f"  📖 Chapter: परिच्छेद–{self.current_chapter} ({self.current_chapter_title})")
                i += 1
                continue
            
            # Check for Section
            section_match = self.PATTERNS['section'].match(line)
            if section_match:
                # Save previous section
                if current_section_num and section_content:
                    chunk = self._create_chunk(
                        current_section_num,
                        current_section_title,
                        '\n'.join(section_content)
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_section_num = section_match.group(1)
                current_section_title = section_match.group(2).strip()
                section_content = [line]
                logger.info(f"    📝 Section {current_section_num}: {current_section_title[:50]}...")
                i += 1
                continue
            
            # Add to current section content
            if current_section_num:
                section_content.append(line)
            
            i += 1
        
        # Save last section
        if current_section_num and section_content:
            chunk = self._create_chunk(
                current_section_num,
                current_section_title,
                '\n'.join(section_content)
            )
            chunks.append(chunk)
        
        logger.info(f"✅ Parsed {len(chunks)} sections")
        return chunks
    
    def _create_chunk(
        self,
        section_num: str,
        section_title: str,
        content: str
    ) -> Tuple[ChunkHierarchy, str]:
        """Create a chunk with hierarchy"""
        hierarchy = ChunkHierarchy(
            act=self.metadata.act_name_nepali,
            act_english=self.metadata.act_name_english,
            part=self.current_part,
            part_title=self.current_part_title,
            chapter=self.current_chapter,
            chapter_title=self.current_chapter_title,
            section=section_num,
            section_title=section_title
        )
        
        return (hierarchy, content)


class PostProcessingTracker:
    """Track processed files to avoid duplicate processing"""
    
    def __init__(self, tracker_path: str):
        """
        Initialize tracker
        
        Args:
            tracker_path: Path to tracking JSON file
        """
        self.tracker_path = tracker_path
        self.data = self._load_tracker()
        logger.info(f"📊 PostProcessing Tracker loaded: {len(self.data.get('processed', {}))} files")
    
    def _load_tracker(self) -> Dict:
        """Load existing tracker data"""
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load tracker: {e}")
        
        return {"processed": {}, "failed": {}}
    
    def _save_tracker(self):
        """Save tracker to disk"""
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
        with open(self.tracker_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_processed(self, file_path: str) -> bool:
        """Check if file has been processed"""
        file_hash = self._get_file_hash(file_path)
        return file_hash in self.data["processed"]
    
    def mark_processed(
        self,
        file_path: str,
        output_path: str,
        metadata_path: str,
        chunks_count: int
    ):
        """Mark file as successfully processed"""
        file_hash = self._get_file_hash(file_path)
        
        self.data["processed"][file_hash] = {
            "file_name": os.path.basename(file_path),
            "source_path": file_path,
            "output_path": output_path,
            "metadata_path": metadata_path,
            "chunks_count": chunks_count,
            "processed_at": datetime.now().isoformat()
        }
        
        self._save_tracker()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark file as failed"""
        file_hash = self._get_file_hash(file_path)
        
        self.data["failed"][file_hash] = {
            "file_name": os.path.basename(file_path),
            "error": str(error),
            "failed_at": datetime.now().isoformat()
        }
        
        self._save_tracker()


class DocumentPostProcessor:
    """Main postprocessor class"""
    
    def __init__(
        self,
        llm,
        processed_dir: str,
        structured_dir: str,
        metadata_dir: str,
        tracker_path: str
    ):
        """
        Initialize postprocessor
        
        Args:
            llm: Language model for metadata extraction
            processed_dir: Input directory (parsed markdown files)
            structured_dir: Output directory (chunked files)
            metadata_dir: Directory for metadata JSON files
            tracker_path: Path to processing tracker
        """
        self.llm = llm
        self.processed_dir = processed_dir
        self.structured_dir = structured_dir
        self.metadata_dir = metadata_dir
        self.tracker = PostProcessingTracker(tracker_path)
        
        # Create directories
        os.makedirs(structured_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        logger.info("✅ DocumentPostProcessor initialized")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file
        
        Args:
            file_path: Path to processed markdown file
            
        Returns:
            Processing statistics
        """
        filename = os.path.basename(file_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing: {filename}")
        logger.info(f"{'='*60}")
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Step 1: Extract metadata (from first 2000 chars)
            logger.info("📋 Step 1: Extracting metadata...")
            text_sample = raw_text[:2000]
            metadata_extractor = MetadataExtractor(self.llm)
            metadata = metadata_extractor.extract(text_sample, filename)
            
            # Save metadata to JSON
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
            metadata_path = os.path.join(self.metadata_dir, f"{file_hash}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Metadata saved: {metadata_path}")
            
            # Step 2: Normalize text
            logger.info("🧹 Step 2: Normalizing text...")
            normalized_text = TextNormalizer.normalize(raw_text)
            
            # Step 3: Parse hierarchy and create chunks
            logger.info("🔍 Step 3: Parsing hierarchy...")
            parser = HierarchicalParser(metadata)
            chunks = parser.parse_sections(normalized_text)
            
            # Step 4: Write chunks to file
            logger.info("💾 Step 4: Writing chunks...")
            output_filename = Path(filename).stem + "_chunks.md"
            output_path = os.path.join(self.structured_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write file header
                f.write(f"# FILE METADATA\n")
                f.write(f"# Source: {filename}\n")
                f.write(f"# Act: {metadata.act_name_english}\n")
                f.write(f"# Total Chunks: {len(chunks)}\n")
                f.write(f"# Processed: {datetime.now().isoformat()}\n\n")
                
                # Write each chunk
                for i, (hierarchy, content) in enumerate(chunks, 1):
                    f.write(f"--------START OF CHUNK {i}--------\n")
                    f.write(f"metadata:\n")
                    f.write(f"  act: {hierarchy.act}\n")
                    f.write(f"  act_english: {hierarchy.act_english}\n")
                    f.write(f"  published_date: {metadata.published_date}\n")
                    f.write(f"  is_amendment: {metadata.is_amendment}\n")
                    
                    f.write(f"hierarchy:\n")
                    if hierarchy.part:
                        f.write(f"  part: {hierarchy.part}\n")
                        if hierarchy.part_title:
                            f.write(f"  part_title: {hierarchy.part_title}\n")
                    if hierarchy.chapter:
                        f.write(f"  chapter: परिच्छेद–{hierarchy.chapter}\n")
                        if hierarchy.chapter_title:
                            f.write(f"  chapter_title: {hierarchy.chapter_title}\n")
                    f.write(f"  section: {hierarchy.section}\n")
                    f.write(f"  section_title: {hierarchy.section_title}\n")
                    
                    f.write(f"text: |\n")
                    # Indent the content
                    for line in content.split('\n'):
                        f.write(f"  {line}\n")
                    f.write(f"--------END OF CHUNK {i}--------\n\n")
            
            logger.info(f"✅ Created {len(chunks)} chunks: {output_path}")
            
            # Mark as processed
            self.tracker.mark_processed(file_path, output_path, metadata_path, len(chunks))
            
            return {
                "status": "success",
                "chunks": len(chunks),
                "output": output_path,
                "metadata": metadata_path
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}", exc_info=True)
            self.tracker.mark_failed(file_path, str(e))
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def process_directory(self) -> Dict[str, Any]:
        """
        Process all files in the processed directory
        
        Returns:
            Processing statistics
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting Batch Postprocessing")
        logger.info("="*60)
        
        # Get all markdown files
        md_files = list(Path(self.processed_dir).glob("*.md"))
        logger.info(f"📁 Found {len(md_files)} markdown files")
        
        if not md_files:
            logger.warning("⚠️ No files found to process")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}
        
        for i, file_path in enumerate(md_files, 1):
            logger.info(f"\n[{i}/{len(md_files)}] {file_path.name}")
            
            # Check if already processed
            if self.tracker.is_processed(str(file_path)):
                logger.info(f"⏭️ Skipping (already processed)")
                stats["skipped"] += 1
                continue
            
            # Process file
            result = self.process_file(str(file_path))
            
            if result["status"] == "success":
                stats["processed"] += 1
                stats["total_chunks"] += result["chunks"]
            else:
                stats["failed"] += 1
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 Processing Summary")
        logger.info("="*60)
        logger.info(f"✅ Processed: {stats['processed']} files")
        logger.info(f"📦 Total Chunks: {stats['total_chunks']}")
        logger.info(f"⏭️ Skipped: {stats['skipped']} files")
        logger.info(f"❌ Failed: {stats['failed']} files")
        logger.info("="*60)
        
        return stats


if __name__ == "__main__":
    import sys
    
    # Dynamic path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.core.secrets import SecretManager
    from src.core.llm_factory import LLMFactory
    
    # Load configuration
    secrets = SecretManager()
    
    # Get paths from .env
    processed_dir = secrets.get_from_env("PROCESSED_DATA_PATH", default="data/processed")
    structured_dir = secrets.get_from_env("STRUCTURED_DATA_PATH", default="data/structured")
    
    # Resolve paths
    processed_dir = os.path.join(project_root, processed_dir)
    structured_dir = os.path.join(project_root, structured_dir)
    metadata_dir = os.path.join(project_root, "data", "metadata")
    tracker_path = os.path.join(project_root, "data", "postprocessing_tracker.json")
    
    # Initialize LLM for metadata extraction (using Groq for speed)
    try:
        groq_key = secrets.get_from_env("GROQ_API_KEY")
        if not groq_key:
            logger.error("❌ GROQ_API_KEY not found in .env")
            sys.exit(1)
        
        llm = LLMFactory.create_llm(
            provider="groq",
            model_name="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0.1
        )
        
        # Initialize postprocessor
        postprocessor = DocumentPostProcessor(
            llm=llm,
            processed_dir=processed_dir,
            structured_dir=structured_dir,
            metadata_dir=metadata_dir,
            tracker_path=tracker_path
        )
        
        # Process all files
        stats = postprocessor.process_directory()
        
        # Exit with appropriate code
        sys.exit(0 if stats["failed"] == 0 else 1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
