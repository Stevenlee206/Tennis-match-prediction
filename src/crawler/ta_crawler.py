import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import time

class BaseCrawler(ABC):
    """
    Abstract Base Class for Crawlers.
    """
    def __init__(self, start_year: int, end_year: int, save_dir: str):
        self.start_year = start_year
        self.end_year = end_year
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    @abstractmethod
    def crawl(self) -> pd.DataFrame:
        """
        Main crawling logic. Must return a pandas DataFrame.
        """
        pass
        
    def save(self, df: pd.DataFrame, filename: str):
        """
        Saves the crawled data to CSV.
        """
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")

class TACrawler(BaseCrawler):
    """
    Tennis Abstract Crawler to fetch match data exactly matching the format of Jeff Sackmann's ATP matches.
    Note: Tennis Abstract's backend database is maintained open-source by its creator, Jeff Sackmann, 
    at https://github.com/JeffSackmann/tennis_atp. 
    Fetching the raw CSVs directly from this repo is the official, most robust way to 'crawl' TennisAbstract data, 
    ensuring 100% column compatibility without fragile HTML DOM parsing.
    """
    def __init__(self, start_year: int, end_year: int, save_dir: str = "data/raw_data"):
        super().__init__(start_year, end_year, save_dir)
        self.base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"

    def crawl(self) -> pd.DataFrame:
        """
        Fetches the exact match data year by year from the official TennisAbstract raw data repository.
        """
        all_dfs = []
        for year in range(self.start_year, self.end_year + 1):
            url = self.base_url.format(year)
            print(f"Fetching data for year {year} from TennisAbstract backend...")
            try:
                # Fetching directly into pandas
                df_year = pd.read_csv(url)
                if not df_year.empty:
                    self.save(df_year, f"atp_matches_{year}.csv")
                    all_dfs.append(df_year)
                    print(f"  -> Successfully fetched {len(df_year)} matches for {year}.")
            except Exception as e:
                print(f"  -> No data or error for year {year} (Error: {e})")
                
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    crawler = TACrawler(start_year=2025, end_year=2026)
    df = crawler.crawl()
    print(f"Summary: Crawled {len(df)} real matches successfully.")

