"""
Complete End-to-End Data Collection & Processing Pipeline
Stages: Scraping → Collection → Parsing → Postprocessing

Usage:
    python -m src.ingestion.data_collection_processing_pipeline --all
    python -m src.ingestion.data_collection_processing_pipeline --scrape
    python -m src.ingestion.data_collection_processing_pipeline --parse
    python -m src.ingestion.data_collection_processing_pipeline --postprocess
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Dynamic path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.scraper import VidhiScraper
from src.ingestion.parser_factory import ParserFactory
from src.ingestion.postprocessor import DocumentPostProcessor
from src.core.secrets import SecretManager
from src.core.llm_factory import LLMFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'pipeline.log'))
    ]
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Complete data collection and processing pipeline"""
    
    def __init__(self):
        """Initialize pipeline with configuration from .env"""
        self.secrets = SecretManager()
        
        # Get paths
        self.raw_dir = os.path.join(project_root, self.secrets.get_from_env("RAW_DATA_PATH", default="data/raw"))
        self.processed_dir = os.path.join(project_root, self.secrets.get_from_env("PROCESSED_DATA_PATH", default="data/processed"))
        self.structured_dir = os.path.join(project_root, self.secrets.get_from_env("STRUCTURED_DATA_PATH", default="data/structured"))
        self.metadata_dir = os.path.join(project_root, "data", "metadata")
        
        # Tracker paths
        self.processing_tracker = os.path.join(project_root, "data", "processing_tracker.json")
        self.postprocessing_tracker = os.path.join(project_root, "data", "postprocessing_tracker.json")
        
        # Parser configuration
        self.parser_type = self.secrets.get_from_env("PARSER_TYPE", default="llama")
        
        logger.info("✅ Pipeline initialized")
        logger.info(f"📁 Raw: {self.raw_dir}")
        logger.info(f"📁 Processed: {self.processed_dir}")
        logger.info(f"📁 Structured: {self.structured_dir}")
        logger.info(f"🔧 Parser: {self.parser_type}")
    
    def stage_1_scrape_and_collect(self) -> Dict[str, Any]:
        """
        Stage 1: Scrape lawcommission.gov.np and download PDFs
        
        Returns:
            Statistics dict
        """
        logger.info("\n" + "="*70)
        logger.info("🌐 STAGE 1: WEB SCRAPING & COLLECTION")
        logger.info("="*70)
        
        try:
            scraper = VidhiScraper()
            
            # Get initial count
            initial_count = len(list(Path(self.raw_dir).glob("*.pdf")))
            logger.info(f"📊 Initial PDFs: {initial_count}")
            
            # Run scraper
            scraper.scrape_all_volumes()
            
            # Get final count
            final_count = len(list(Path(self.raw_dir).glob("*.pdf")))
            new_downloads = final_count - initial_count
            
            logger.info(f"✅ Stage 1 Complete!")
            logger.info(f"📦 Total PDFs: {final_count}")
            logger.info(f"⬇️  New Downloads: {new_downloads}")
            
            return {
                "status": "success",
                "initial": initial_count,
                "final": final_count,
                "new": new_downloads
            }
            
        except Exception as e:
            logger.error(f"❌ Stage 1 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def stage_2_parse_pdfs(self) -> Dict[str, Any]:
        """
        Stage 2: Parse PDFs to structured markdown
        
        Returns:
            Statistics dict
        """
        logger.info("\n" + "="*70)
        logger.info("📄 STAGE 2: PDF PARSING")
        logger.info("="*70)
        
        try:
            # Get parser-specific configuration
            parser_kwargs = {}
            if self.parser_type == "llama":
                api_key = self.secrets.get_from_env("LLAMA_PARSE_API_KEY")
                if not api_key:
                    raise ValueError("LLAMA_PARSE_API_KEY not found in .env")
                parser_kwargs["api_key"] = api_key
            
            # Run parser
            stats = ParserFactory.process_directory(
                raw_dir=self.raw_dir,
                processed_dir=self.processed_dir,
                parser_type=self.parser_type,
                tracker_path=self.processing_tracker,
                parser_kwargs=parser_kwargs
            )
            
            logger.info(f"✅ Stage 2 Complete!")
            logger.info(f"📝 Processed: {stats['processed']} files")
            logger.info(f"⏭️  Skipped: {stats['skipped']} files")
            logger.info(f"❌ Failed: {stats['failed']} files")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Stage 2 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def stage_3_postprocess(self) -> Dict[str, Any]:
        """
        Stage 3: Postprocess markdown to structured chunks
        
        Returns:
            Statistics dict
        """
        logger.info("\n" + "="*70)
        logger.info("🔄 STAGE 3: POSTPROCESSING & CHUNKING")
        logger.info("="*70)
        
        try:
            # Initialize LLM for metadata extraction
            groq_key = self.secrets.get_from_env("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            
            llm = LLMFactory.create_llm(
                provider="groq",
                model_name="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            
            # Initialize postprocessor
            postprocessor = DocumentPostProcessor(
                llm=llm,
                processed_dir=self.processed_dir,
                structured_dir=self.structured_dir,
                metadata_dir=self.metadata_dir,
                tracker_path=self.postprocessing_tracker
            )
            
            # Process all files
            stats = postprocessor.process_directory()
            
            logger.info(f"✅ Stage 3 Complete!")
            logger.info(f"📦 Processed: {stats['processed']} files")
            logger.info(f"🔢 Total Chunks: {stats['total_chunks']}")
            logger.info(f"⏭️  Skipped: {stats['skipped']} files")
            logger.info(f"❌ Failed: {stats['failed']} files")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Stage 3 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run all stages sequentially
        
        Returns:
            Complete statistics
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 VIDHI AI - COMPLETE DATA PIPELINE")
        logger.info("="*70)
        logger.info(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        results = {}
        
        # Stage 1: Scrape & Collect
        results["stage_1_scraping"] = self.stage_1_scrape_and_collect()
        
        if results["stage_1_scraping"].get("status") == "failed":
            logger.error("❌ Pipeline stopped at Stage 1")
            return results
        
        # Stage 2: Parse PDFs
        results["stage_2_parsing"] = self.stage_2_parse_pdfs()
        
        if results["stage_2_parsing"].get("failed", 0) > 0:
            logger.warning("⚠️ Stage 2 had failures, but continuing...")
        
        # Stage 3: Postprocess
        results["stage_3_postprocessing"] = self.stage_3_postprocess()
        
        # Calculate totals
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("✨ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"⏰ Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary
        logger.info("\n📊 SUMMARY:")
        logger.info(f"  Stage 1 - Downloaded: {results['stage_1_scraping'].get('new', 0)} new PDFs")
        logger.info(f"  Stage 2 - Parsed: {results['stage_2_parsing'].get('processed', 0)} PDFs")
        logger.info(f"  Stage 3 - Created: {results['stage_3_postprocessing'].get('total_chunks', 0)} chunks")
        logger.info("="*70)
        
        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Vidhi AI Data Collection & Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (all stages)
  python -m src.ingestion.data_collection_processing_pipeline --all
  
  # Run individual stages
  python -m src.ingestion.data_collection_processing_pipeline --scrape
  python -m src.ingestion.data_collection_processing_pipeline --parse
  python -m src.ingestion.data_collection_processing_pipeline --postprocess
  
  # Run parsing + postprocessing (skip scraping)
  python -m src.ingestion.data_collection_processing_pipeline --parse --postprocess
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all stages (scraping → parsing → postprocessing)'
    )
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Run Stage 1: Web scraping and PDF collection'
    )
    parser.add_argument(
        '--parse',
        action='store_true',
        help='Run Stage 2: PDF parsing to markdown'
    )
    parser.add_argument(
        '--postprocess',
        action='store_true',
        help='Run Stage 3: Postprocessing and chunking'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not (args.all or args.scrape or args.parse or args.postprocess):
        parser.print_help()
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    try:
        # Run complete pipeline
        if args.all:
            results = pipeline.run_full_pipeline()
            sys.exit(0 if all(r.get("status") != "failed" for r in results.values()) else 1)
        
        # Run individual stages
        if args.scrape:
            result = pipeline.stage_1_scrape_and_collect()
            if result.get("status") == "failed":
                sys.exit(1)
        
        if args.parse:
            result = pipeline.stage_2_parse_pdfs()
            if result.get("failed", 0) > 0:
                logger.warning("⚠️ Some files failed to parse")
        
        if args.postprocess:
            result = pipeline.stage_3_postprocess()
            if result.get("failed", 0) > 0:
                logger.warning("⚠️ Some files failed to postprocess")
        
        logger.info("\n✅ Pipeline execution completed!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
