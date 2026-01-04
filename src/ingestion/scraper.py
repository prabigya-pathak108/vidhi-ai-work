import requests
from bs4 import BeautifulSoup

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.ingestion.collector import LawCollector
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re
import logging
from datetime import datetime
from src.ingestion.collector import LawCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VidhiScraper:
    def __init__(self):
        self.collector = LawCollector()
        self.base_url = "https://lawcommission.gov.np"
        self.volumes_list_url = f"{self.base_url}/pages/list-volume-act/"
        
        # Mapping from your logic
        self.volume_categories = {
            1762: "Constitutional Body and Good Governance",
            1763: "Courts and Administration of Justice", 
            1764: "Federal Parliament, Political Parties and Elections",
            1765: "Internal Administration",
            1766: "Security Administration",
            1767: "Revenue and Financial Administration",
            1768: "Currency, Banking, Insurance, Financial Institutions and Securities",
            1769: "Industry, Commerce and Supply",
            1783: "Tourism, Labor and Transport",
            1784: "Communication, Science and Technology",
            1785: "Planning, Development and Construction",
            1786: "Food, Agriculture, Cooperative and Land Administration",
            1787: "Nature, Environment and Water Resources",
            1788: "Foreign Affairs, Education and Sports",
            1789: "Health",
            1790: "Women, Children, Social Welfare and Culture",
            1791: "Local Development"
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Vidhi-AI/1.0'
        })

    def get_volume_links(self):
        """Extracts volume category links from the main table."""
        try:
            resp = self.session.get(self.volumes_list_url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            volume_links = {}
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        link = cols[1].find('a', class_='in-cell-link')
                        if link:
                            cat_id = link['href'].split('/')[-2]
                            volume_links[cat_id] = link['href']
            return volume_links
        except Exception as e:
            logger.error(f"Failed to scrape volume list: {e}")
            return {}

    def scrape_all_volumes(self):
        volume_links = self.get_volume_links()
        logger.info(f"Found {len(volume_links)} volume categories. Starting collection...")

        for cat_id, category_url in volume_links.items():
            vol_name = self.volume_categories.get(int(cat_id), 'Unknown')
            logger.info(f"--- Processing Volume {cat_id}: {vol_name} ---")
            
            page_num = 1
            while True:
                if category_url.startswith('http'):
                    target_url = category_url
                else:
                    target_url = urljoin(self.base_url, category_url)
                
                # Construct the paginated URL correctly
                category_page_url = f"{target_url}?page={page_num}"
                
                try:
                    resp = self.session.get(category_page_url)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    
                    table = soup.find('table', class_='table')
                    if not table: break
                    
                    rows = table.find('tbody').find_all('tr')
                    if not rows: break
                    
                    found_on_page = 0
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) < 5: continue
                        
                        title = cols[1].get_text(strip=True)
                        pub_date = cols[2].get_text(strip=True)
                        
                        pdf_link = cols[3].find('a')
                        if not pdf_link or not pdf_link.get('href'): continue
                        
                        pdf_url = urljoin(self.base_url, pdf_link['href'])
                        
                        # Prepare data for the collector
                        law_item = {
                            "url": pdf_url,
                            "title": title, # e.g., "नागरिकता ऐन, २०६३"
                            "published_date": pub_date,
                            "volume_name": vol_name,
                            "volume_id": cat_id,
                            "serial": cols[0].get_text(strip=True),
                            "category": vol_name  # Changed from "Volume-Act" to the actual category name
                        }
                                                
                        # Use collector to check manifest and download
                        self.collector.download_law(law_item)
                        found_on_page += 1

                    if found_on_page == 0: break
                    
                    # Pagination Check
                    next_btn = soup.find('a', class_='next__pagination')
                    if not next_btn: break
                    
                    page_num += 1
                    time.sleep(1) # Polite scraping
                    
                except Exception as e:
                    logger.error(f"Error on volume {cat_id} page {page_num}: {e}")
                    break

if __name__ == "__main__":
    scraper = VidhiScraper()
    scraper.scrape_all_volumes()