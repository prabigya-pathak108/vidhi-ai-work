"""
Production-ready PDF Parser Factory with multiple parser implementations
Supports: LlamaParse, Tesseract OCR
Features: Deduplication, logging, error handling
"""
import os
import re
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParserBase(ABC):
    """Abstract base class for all parsers"""
    
    @abstractmethod
    def parse(self, pdf_path: str, output_path: str) -> str:
        """
        Parse a PDF file and save to output path
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path to save parsed output
            
        Returns:
            str: Path to the saved output file
        """
        pass
    
    @abstractmethod
    def get_parser_name(self) -> str:
        """Return the name of the parser"""
        pass


class LlamaParser(ParserBase):
    """LlamaParse implementation for high-quality parsing"""
    
    def __init__(self, api_key: str):
        """
        Initialize LlamaParse
        
        Args:
            api_key: LlamaParse API key
        """
        if not api_key:
            raise ValueError("LlamaParse API key is required")
        
        self.api_key = api_key
        logger.info("✅ LlamaParser initialized")
    
    def parse(self, pdf_path: str, output_path: str) -> str:
        """Parse PDF using LlamaParse"""
        from llama_cloud_services import LlamaParse

        
        logger.info(f"📄 Parsing with LlamaParse: {os.path.basename(pdf_path)}")
        
        try:
            parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                tier="agentic_plus",
                version="latest",
                high_res_ocr=True,
                adaptive_long_table=True,
                outlined_table_extraction=True,
                output_tables_as_HTML=True,
                max_pages=0,
                skip_diagonal_text=True,
                hide_headers=True,
                hide_footers=True,
                precise_bounding_box=True,
            )
            
            # Parse the document
            documents = parser.load_data(pdf_path)
            
            # Combine all pages
            full_text = "\n\n".join([doc.text for doc in documents])
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"✅ Successfully parsed {len(documents)} pages")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ LlamaParse failed: {e}")
            raise
    
    def get_parser_name(self) -> str:
        return "llama"


class TesseractParser(ParserBase):
    """Tesseract OCR implementation for Nepali text extraction"""
    
    def __init__(self):
        """Initialize Tesseract parser"""
        try:
            import pytesseract
            import cv2
            from pdf2image import convert_from_path
            logger.info("✅ TesseractParser initialized")
        except ImportError as e:
            logger.error("❌ Missing dependencies. Install: pip install pytesseract opencv-python pdf2image")
            raise ImportError(f"Tesseract dependencies not installed: {e}")
    
    def _preprocess_image(self, pil_img):
        """Apply preprocessing for better OCR results"""
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        open_cv_image = np.array(pil_img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Upscale for better OCR
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Binarization using Otsu's method
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def parse(self, pdf_path: str, output_path: str) -> str:
        """Parse PDF using Tesseract OCR"""
        import pytesseract
        from pdf2image import convert_from_path
        
        logger.info(f"📄 Parsing with Tesseract OCR: {os.path.basename(pdf_path)}")
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            logger.info(f"📑 Converted PDF to {len(pages)} images")
            
            full_text = []
            
            for i, page in enumerate(pages, 1):
                logger.info(f"🔍 Processing page {i}/{len(pages)}...")
                
                # Preprocess image
                processed_img = self._preprocess_image(page)
                
                # Tesseract configuration for Nepali
                custom_config = r'--oem 3 --psm 3 -l nep+eng'
                
                # Extract text
                text = pytesseract.image_to_string(processed_img, config=custom_config)
                full_text.append(f"--- Page {i} ---\n\n{text}")
            
            # Combine all pages
            combined_text = "\n\n".join(full_text)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            logger.info(f"✅ Successfully extracted text from {len(pages)} pages")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Tesseract parsing failed: {e}")
            raise
    
    def get_parser_name(self) -> str:
        return "tesseract"


class ProcessingTracker:
    """Track processed files to avoid duplicate processing"""
    
    def __init__(self, tracker_path: str):
        """
        Initialize processing tracker
        
        Args:
            tracker_path: Path to the tracking JSON file
        """
        self.tracker_path = tracker_path
        self.data = self._load_tracker()
        logger.info(f"📊 Tracker loaded: {len(self.data.get('processed', {}))} files tracked")
    
    def _load_tracker(self) -> Dict:
        """Load existing tracker data"""
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load tracker, creating new: {e}")
        
        return {
            "processed": {},
            "failed": {},
            "last_updated": None
        }
    
    def _save_tracker(self):
        """Save tracker data to disk"""
        self.data["last_updated"] = datetime.now().isoformat()
        
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
    
    def mark_processed(self, file_path: str, output_path: str, parser_name: str):
        """Mark file as successfully processed"""
        file_hash = self._get_file_hash(file_path)
        
        self.data["processed"][file_hash] = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "output_path": output_path,
            "parser": parser_name,
            "processed_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        
        self._save_tracker()
        logger.info(f"✅ Marked as processed: {os.path.basename(file_path)}")
    
    def mark_failed(self, file_path: str, error: str, parser_name: str):
        """Mark file as failed"""
        file_hash = self._get_file_hash(file_path)
        
        self.data["failed"][file_hash] = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "parser": parser_name,
            "error": str(error),
            "failed_at": datetime.now().isoformat()
        }
        
        self._save_tracker()
        logger.warning(f"⚠️ Marked as failed: {os.path.basename(file_path)}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            "total_processed": len(self.data["processed"]),
            "total_failed": len(self.data["failed"]),
            "last_updated": self.data.get("last_updated")
        }


class ParserFactory:
    """Factory for creating and managing parsers"""
    
    @staticmethod
    def create_parser(parser_type: str, **kwargs) -> ParserBase:
        """
        Create a parser instance
        
        Args:
            parser_type: Type of parser ('llama' or 'tesseract')
            **kwargs: Parser-specific arguments
            
        Returns:
            ParserBase: Parser instance
        """
        parser_type = parser_type.lower().strip()
        
        if parser_type == "llama":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("LlamaParse requires 'api_key' parameter")
            return LlamaParser(api_key=api_key)
        
        elif parser_type == "tesseract":
            return TesseractParser()
        
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}. Use 'llama' or 'tesseract'")
    
    @staticmethod
    def process_directory(
        raw_dir: str,
        processed_dir: str,
        parser_type: str,
        tracker_path: str,
        parser_kwargs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process all PDF files in a directory
        
        Args:
            raw_dir: Directory containing raw PDF files
            processed_dir: Directory to save processed files
            parser_type: Type of parser to use
            tracker_path: Path to processing tracker file
            parser_kwargs: Additional arguments for parser
            
        Returns:
            Dict with processing statistics
        """
        if parser_kwargs is None:
            parser_kwargs = {}
        
        logger.info("="*60)
        logger.info("🚀 Starting Batch PDF Processing")
        logger.info("="*60)
        
        # Initialize parser
        parser = ParserFactory.create_parser(parser_type, **parser_kwargs)
        logger.info(f"📦 Using parser: {parser.get_parser_name()}")
        
        # Initialize tracker
        tracker = ProcessingTracker(tracker_path)
        
        # Create output directory
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get all PDF files
        pdf_files = list(Path(raw_dir).glob("*.pdf"))
        logger.info(f"📁 Found {len(pdf_files)} PDF files in {raw_dir}")
        
        if not pdf_files:
            logger.warning("⚠️ No PDF files found to process")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        # Process files
        stats = {"processed": 0, "skipped": 0, "failed": 0}
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            # Check if already processed
            if tracker.is_processed(str(pdf_path)):
                logger.info(f"⏭️ Skipping (already processed): {pdf_path.name}")
                stats["skipped"] += 1
                continue
            
            try:
                # Generate output filename
                output_filename = pdf_path.stem + ".md"
                output_path = os.path.join(processed_dir, output_filename)
                
                # Parse the file
                parser.parse(str(pdf_path), output_path)
                
                # Mark as processed
                tracker.mark_processed(
                    str(pdf_path),
                    output_path,
                    parser.get_parser_name()
                )
                
                stats["processed"] += 1
                logger.info(f"✅ Successfully processed: {pdf_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path.name}: {e}")
                tracker.mark_failed(str(pdf_path), str(e), parser.get_parser_name())
                stats["failed"] += 1
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 Processing Summary")
        logger.info("="*60)
        logger.info(f"✅ Processed: {stats['processed']}")
        logger.info(f"⏭️ Skipped: {stats['skipped']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"📈 Success Rate: {stats['processed']/(stats['processed']+stats['failed'])*100:.1f}%" if (stats['processed']+stats['failed']) > 0 else "N/A")
        logger.info("="*60)
        
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
    
    # Load configuration
    secrets = SecretManager()
    
    # Get configuration from .env
    parser_type = secrets.get_from_env("PARSER_TYPE", default="llama")
    raw_dir = secrets.get_from_env("RAW_DATA_PATH", default="data/raw")
    processed_dir = secrets.get_from_env("PROCESSED_DATA_PATH", default="data/processed")
    
    # Resolve paths
    raw_dir = os.path.join(project_root, raw_dir)
    processed_dir = os.path.join(project_root, processed_dir)
    tracker_path = os.path.join(project_root, "data", "processing_tracker.json")
    
    # Get parser-specific configuration
    parser_kwargs = {}
    if parser_type == "llama":
        api_key = secrets.get_from_env("LLAMA_PARSE_API_KEY")
        if not api_key:
            logger.error("❌ LLAMA_PARSE_API_KEY not found in .env")
            sys.exit(1)
        parser_kwargs["api_key"] = api_key
    
    # Process all files
    try:
        stats = ParserFactory.process_directory(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            parser_type=parser_type,
            tracker_path=tracker_path,
            parser_kwargs=parser_kwargs
        )
        
        # Exit with appropriate code
        sys.exit(0 if stats["failed"] == 0 else 1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

