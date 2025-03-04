import requests
import time
import logging
import sqlite3
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
from urllib.parse import urljoin

nltk.download('punkt_tab')
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file
DB_FILE = 'inria_jobs.db'
# The main URL for INRIA job offers
SCRAPE_URL = "https://jobs.inria.fr/public/classic/en/offres"

def init_db():
    """Initialize the database and create the jobs table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            job_id TEXT UNIQUE,
            title TEXT,
            location TEXT,
            team TEXT,
            deadline TEXT,
            summary TEXT,
            link TEXT,
            keywords TEXT,
            supervisor TEXT,
            funding TEXT,
            is_phd BOOLEAN,
            last_updated TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

def extract_keywords(text, num_keywords=5):
    """Extract important keywords from job description."""
    try:
        # Download NLTK resources if not already available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[A-Za-z][A-Za-z-]{2,}\b', text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([word for word, freq in sorted_words[:num_keywords]])
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return ""

def create_summary(text, max_length=200):
    """Create a brief summary of the job description."""
    try:
        # Get the first 1-2 sentences if available
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:2])
        
        # Truncate if still too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary
    except Exception as e:
        logger.warning(f"Error creating summary: {e}")
        return text[:max_length] + "..." if len(text) > max_length else text


def is_phd_position(title, description):
    title_lower = title.strip().lower()
    return title_lower.startswith("phd position f/m") or title_lower.startswith("doctorant f/h")

def fetch_jobs():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(SCRAPE_URL, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"Fetched main page with status {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"HTTP error occurred: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    jobs = []
    
    # Find job cards
    job_cards = soup.select("div.job-card")
    if not job_cards:
        logger.warning("No job cards found on the main page.")
        return jobs
    
    logger.info(f"Found {len(job_cards)} job cards on the main page.")
    
    for idx, card in enumerate(job_cards, start=1):
        try:
            # 1) Title
            title_elem = card.find("h2")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            
            # 2) Parse location, team, deadline from <ul>
            ul = card.find("ul", class_="list-unstyled infos-liste-offre-inria")
            location = "Unknown"
            team = ""
            deadline = ""
            if ul:
                li_tags = ul.find_all("li")
                for li in li_tags:
                    li_text = li.get_text(separator=" ", strip=True)
                    if ":" in li_text:
                        label, value = li_text.split(":", 1)
                        label = label.strip().lower()
                        value = value.strip()
                        if "town/city" in label:
                            location = value
                        elif "inria team" in label:
                            team = value
                        elif "deadline" in label:
                            deadline = value
            
            # 3) Extract a link from the card, if it exists
            link_elem = card.find("a", href=True)
            if link_elem:
                raw_link = link_elem["href"]  # e.g. "/public/classic/en/offres/2025-08547"
                # Make sure it's an absolute URL
                if not raw_link.startswith("http"):
                    # Join with the base site URL
                    job_link = urljoin("https://jobs.inria.fr", raw_link)
            else:
                # If there's no <a>, you could guess the link from some known ID
                # or just skip. For example:
                job_link = f"https://jobs.inria.fr/public/classic/en/offres/job-{idx}"
            
            # 4) Derive a job_id from the link (the last part of the URL)
            #    e.g. "2025-08547"
            job_id = job_link.rsplit("/", 1)[-1]  # everything after last '/'
            
            # 5) Check if it's PhD
            is_phd = is_phd_position(title, "")
            if not is_phd:
                continue
            
            # 6) Build the job data
            job_data = {
                'job_id': job_id,
                'title': title,
                'location': location,
                'team': team,
                'deadline': deadline,
                'summary': "",
                'link': job_link,
                'keywords': "",
                'supervisor': "",
                'funding': "",
                'is_phd': True,
                'last_updated': datetime.now().isoformat()
            }

            details = fetch_job_details(job_link)
            job_data['supervisor'] = details.get('supervisor', '')
            job_data['funding'] = details.get('funding', '')
            job_data['keywords'] = details.get('keywords', '')

            jobs.append(job_data)
        except Exception as ex:
            logger.warning(f"Failed to parse job card {idx}: {ex}")
        time.sleep(0.5)
    
    logger.info(f"Collected {len(jobs)} PhD positions from the main page.")
    return jobs

def fetch_job_details(url):
    """Fetch detailed information from a job's detail page using text extraction."""
    details = {}
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        detail_soup = BeautifulSoup(response.text, 'html.parser')
        logger.debug(f"Fetched details page with status {response.status_code}")
        
        # Try to extract the main description from known selectors.
        description_div = None
        for selector in [
            'div.content-offre', 
            'div.job-description', 
            'div#offer-description',
            'div.content',
            'article.job-detail'
        ]:
            description_div = detail_soup.select_one(selector)
            if description_div:
                break
        
        if description_div:
            full_text = description_div.get_text(separator=" ", strip=True)
        else:
            # Fallback: use the entire page text if no specific container is found.
            full_text = detail_soup.get_text(separator=" ", strip=True)
        
        details['full_description'] = full_text
        
        # Extract PhD Supervisor info: text after "PhD Supervisor :"
        supervisor_pattern = re.compile(r"PhD Supervisor\s*:\s*([^\n]+)", re.IGNORECASE)
        supervisor_match = supervisor_pattern.search(full_text)
        if supervisor_match:
            details['supervisor'] = supervisor_match.group(1).strip()
        
        # Extract funding: text after "remuneration :" and before "euros" or "€"
        funding_pattern = re.compile(r"Remuneration\s*:\s*([\d\s,\.]+)\s+(euros|€)", re.IGNORECASE)
        funding_match = funding_pattern.search(full_text)
        if funding_match:
            details['funding'] = funding_match.group(1).strip()

        
        # Extract keywords: text after "Theme/Domain :"
        keywords_pattern = re.compile(r"Theme/Domain\s*:\s*(.+?)(?=\s{2,}|$)", re.IGNORECASE)
        keywords_match = keywords_pattern.search(full_text)
        if keywords_match:
            details['keywords'] = keywords_match.group(1).strip()
        
        # Determine if this is a PhD position using the title (if available) and the full text
        title_element = detail_soup.select_one('h1') or detail_soup.select_one('h2.offer-title')
        title = title_element.get_text(strip=True) if title_element else ""
        details['is_phd'] = is_phd_position(title, full_text)
        
        # Be polite: sleep a little before returning
        time.sleep(0.5)
        return details
    except Exception as e:
        logger.error(f"Error fetching job details: {e}")
        return details


def reset_database_file():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        logger.info("Existing database file removed.")

def update_database(jobs):
    """Insert new jobs or update existing ones in the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    new_jobs = 0
    updated_jobs = 0
    
    for job in jobs:
        try:
            # Check if job exists
            c.execute("SELECT id FROM jobs WHERE job_id = ?", (job['job_id'],))
            result = c.fetchone()
            
            if result is None:
                # New job
                c.execute('''
                    INSERT INTO jobs (
                        job_id, title, location, team, deadline, 
                        summary, link, keywords, supervisor, funding, is_phd, last_updated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job['job_id'], job['title'], job['location'], job.get('team', ''), 
                    job.get('deadline', ''), job['summary'], job['link'], job.get('keywords', ''),
                    job.get('supervisor', ''), job.get('funding', ''), job.get('is_phd', True), job['last_updated']
                ))
                new_jobs += 1
            else:
                # Update existing job
                c.execute('''
                    UPDATE jobs
                    SET title = ?, location = ?, team = ?, deadline = ?, 
                        summary = ?, link = ?, keywords = ?, supervisor = ?, funding = ?, 
                        is_phd = ?, last_updated = ?
                    WHERE job_id = ?
                ''', (
                    job['title'], job['location'], job.get('team', ''),
                    job.get('deadline', ''), job['summary'], job['link'], job.get('keywords', ''),
                    job.get('supervisor', ''), job.get('funding', ''), job.get('is_phd', True), 
                    job['last_updated'], job['job_id']
                ))
                updated_jobs += 1
        except Exception as e:
            logger.error(f"Error updating database for job {job.get('job_id', 'unknown')}: {e}")
    
    conn.commit()
    conn.close()
    logger.info(f"{new_jobs} new jobs added, {updated_jobs} jobs updated.")
    return new_jobs

def export_to_excel():


    """Export the current job data from the database to an Excel file with a timestamped filename."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM jobs WHERE is_phd = 1", conn)
    conn.close()
    
    if df.empty:
        logger.warning("No PhD positions found in database. Export aborted.")
        return None
    
    # Format the dataframe for better Excel display
    if 'last_updated' in df.columns:
        try:
            df['last_updated'] = pd.to_datetime(df['last_updated']).dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass
    
    # Drop the is_phd column for the export
    if 'is_phd' in df.columns:
        df = df.drop(columns=['is_phd'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inria_phd_positions_{timestamp}.xlsx"
    
    # Create a worksheet with formatting
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='INRIA PhD Positions', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['INRIA PhD Positions']
        
        # Add some formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with the defined format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column widths
        worksheet.set_column('B:B', 40)  # Title column
        worksheet.set_column('C:C', 15)  # Location
        worksheet.set_column('D:D', 15)  # Team
        worksheet.set_column('F:F', 12)  # Deadline
        worksheet.set_column('G:G', 60)  # Summary
        worksheet.set_column('H:H', 40)  # Link
        worksheet.set_column('I:I', 30)  # Keywords
        worksheet.set_column('J:J', 25)  # Supervisor
        worksheet.set_column('K:K', 25)  # Funding
    
    logger.info(f"Exported data to {filename}")
    return filename

def main():
    # Reset the database file
    reset_database_file()
    # Initialize DB if necessary
    init_db()
    
    # Fetch and process jobs
    jobs = fetch_jobs()
    if jobs:
        update_database(jobs)
        export_file = export_to_excel()
        if export_file:
            logger.info(f"Process complete. Data exported to {export_file}")
        else:
            logger.warning("Export failed - no data was written to Excel.")
    else:
        logger.warning("No jobs fetched on this run. Check the HTML dump to inspect the page structure.")

if __name__ == "__main__":
    main()