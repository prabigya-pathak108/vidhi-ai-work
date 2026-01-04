import requests
import hashlib
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
import sys
import io


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.storage_factory import StorageFactory
from src.core.secrets import SecretManager
import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict
import re

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LawCollector:
    def __init__(self):
        self.secrets = SecretManager()
        
        self.storage_type = self.secrets.get_from_env("STORAGE_TYPE", default="local")
        self.raw_path = self.secrets.get_from_env("RAW_DATA_PATH", default="data/raw")
        
        self.storage = StorageFactory.get_storage(self.storage_type, path=self.raw_path)
        self.manifest_path = os.path.join(os.path.dirname(self.raw_path), "manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"urls": {}, "last_run": None}
        return {"urls": {}, "last_run": None}

    def _save_manifest(self):
        self.manifest["last_run"] = datetime.now().isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=4, ensure_ascii=False)

    def download_law(self, item: Dict[str, str]):
        """
        item keys: 'url', 'title', 'published_date', 'category', 'volume_name'
        """
        # 1. Safely extract values at the very start
        url = item.get('url')
        title = item.get('title', 'Unknown_Title')
        
        if not url:
            logger.error("Skipping item: No URL found")
            return
            
        # 2. URL-based de-duplication
        if url in self.manifest["urls"]:
            return

        try:
            logger.info(f"📥 Downloading: {title}")
            
            # Using a custom User-Agent to avoid being blocked by CDN
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Vidhi-AI/1.0'}
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            # 3. Clean Filename
            # Remove illegal characters for both Windows and Linux
            # Keeping Nepali characters as they are valid in modern filesystems
            clean_title = re.sub(r'[\\/*?:"<>|]', '', title).strip()
            
            # If title is empty after cleaning, use a timestamp
            if not clean_title:
                clean_title = f"law_doc_{datetime.now().strftime('%H%M%S')}"
            
            filename = f"{clean_title}.pdf" if not clean_title.lower().endswith('.pdf') else clean_title

            # 4. Save via Storage Factory
            local_full_path = self.storage.save(response.content, filename)

            # 5. Update Manifest
            self.manifest["urls"][url] = {
                **item,
                "local_path": local_full_path,
                "downloaded_at": datetime.now().isoformat(),
                "status": "downloaded"
            }
            self._save_manifest()
            logger.info(f"✅ Saved: {filename}")

        except Exception as e:
            # We use 'url' here instead of 'title' because if 'title' failed to assign, 
            # we want to know WHICH url failed.
            logger.error(f"❌ Failed to process {url}: {str(e)}")

if __name__ == "__main__":
    # Smoke test for collector
    collector = LawCollector()
    print(f"Collector initialized. Storage: {collector.raw_path}")