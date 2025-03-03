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
            posted_date TEXT,
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


def fetch_job_details(url):
    """Fetch detailed information from a job's detail page."""
    details = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        detail_soup = BeautifulSoup(response.text, 'html.parser')
        logger.debug(f"Fetched details page with status {response.status_code}")
        
        # Dump the first detail page to debug
        # if 'debug_dump.html' not in os.listdir():
        #    with open("debug_dump.html", "w", encoding="utf-8") as f:
        #        f.write(response.text)
        
        # Extract the full description - try different possible selectors
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
            full_text = description_div.get_text(strip=True, separator=" ")
            details['full_description'] = full_text
            details['keywords'] = extract_keywords(full_text)
            
            # Try to find supervisor info from the description
            supervisor_pattern = re.compile(r'(supervisor|adviser|director|contact):?\s*([^\.]+?)(?=\.|$)', re.IGNORECASE)
            supervisor_match = supervisor_pattern.search(full_text)
            if supervisor_match:
                details['supervisor'] = supervisor_match.group(2).strip()
            
            # Try to find funding info
            funding_pattern = re.compile(r'(funding|grant|scholarship):?\s*([^\.]+?)(?=\.|$)', re.IGNORECASE)
            funding_match = funding_pattern.search(full_text)
            if funding_match:
                details['funding'] = funding_match.group(2).strip()
        
        # Check if this is a PhD position
        title_element = detail_soup.select_one('h1') or detail_soup.select_one('h2.offer-title')
        title = title_element.get_text(strip=True) if title_element else ""
        details['is_phd'] = is_phd_position(title, details.get('full_description', ''))
                
        # Extract application deadline if available - try various patterns
        deadline_patterns = [
            (re.compile(r'deadline', re.IGNORECASE), 'Application deadline:'),
            (re.compile(r'apply before', re.IGNORECASE), 'Apply before:'),
            (re.compile(r'apply by', re.IGNORECASE), 'Apply by:')
        ]
        
        for pattern, replace_text in deadline_patterns:
            deadline_elem = detail_soup.find(string=pattern)
            if deadline_elem:
                parent = deadline_elem.parent
                if parent:
                    deadline_text = parent.get_text()
                    details['deadline'] = deadline_text.replace(replace_text, '').strip()
                    break
        
        # Extract research team info if available
        team_patterns = [
            (re.compile(r'research team', re.IGNORECASE), 'Research team:'),
            (re.compile(r'team:', re.IGNORECASE), 'Team:'),
            (re.compile(r'laboratory', re.IGNORECASE), 'Laboratory:')
        ]
        
        for pattern, replace_text in team_patterns:
            team_elem = detail_soup.find(string=pattern)
            if team_elem:
                parent = team_elem.parent
                if parent:
                    team_text = parent.get_text()
                    details['team'] = team_text.replace(replace_text, '').strip()
                    break
            
        time.sleep(1)  # Be polite with the server
        return details
    except Exception as e:
        logger.error(f"Error fetching job details: {e}")
        return details

def fetch_jobs():
    """Fetch job offers from the INRIA page and parse them."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(SCRAPE_URL, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"Fetched main page with status {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"HTTP error occurred: {e}")
        return []
    
    # Save the HTML for inspection
    with open("inria_main_page.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    logger.info("Saved main page HTML for inspection")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    jobs = []
    
    # Try multiple potential selectors for job listings
    potential_selectors = [
        "div.list-offers div.offer",
        "table.offer-table tr.offer-item",
        "ul.job-list li",
        "div.job-listing",
        "article.job-offer"
    ]
    
    job_elements = []
    used_selector = ""
    
    for selector in potential_selectors:
        job_elements = soup.select(selector)
        if job_elements:
            used_selector = selector
            logger.info(f"Found job elements with selector: {selector}")
            break
    
    if not job_elements:
        logger.warning("Could not find job listings with any selector. Attempting to find links directly.")
        # Fallback: look for any links that might contain job offers
        job_links = soup.select("a[href*='offre']")
        if job_links:
            logger.info(f"Found {len(job_links)} potential job links")
            
            for link in job_links:
                try:
                    job_url = link['href']
                    if not job_url.startswith('http'):
                        job_url = f"https://jobs.inria.fr{job_url}" if job_url.startswith('/') else f"https://jobs.inria.fr/{job_url}"
                    
                    job_id = re.search(r'offre/(\w+)', job_url)
                    job_id = job_id.group(1) if job_id else f"job-{len(jobs)}"
                    
                    title = link.get_text(strip=True)
                    if not title:
                        title = f"Job {job_id}"
                    
                    # Get the details page
                    details = fetch_job_details(job_url)
                    
                    # Only include PhD positions
                    if not details.get('is_phd', False) and not is_phd_position(title, ''):
                        continue
                    
                    # Create a summary of the job description
                    summary = create_summary(details.get('full_description', ''))
                    
                    jobs.append({
                        'job_id': job_id,
                        'title': title,
                        'location': "Unknown",  # We'll need to extract this from details
                        'team': details.get('team', ''),
                        'posted_date': "Unknown",
                        'deadline': details.get('deadline', ''),
                        'summary': summary,
                        'link': job_url,
                        'keywords': details.get('keywords', ''),
                        'supervisor': details.get('supervisor', ''),
                        'funding': details.get('funding', ''),
                        'is_phd': True,
                        'last_updated': datetime.now().isoformat()
                    })
                    
                    # Be polite with the server
                    time.sleep(2)
                except Exception as ex:
                    logger.warning(f"Failed to process job link: {ex}")
            
            logger.info(f"Extracted {len(jobs)} PhD positions from links")
            return jobs
    
    # Process the job elements if found with a selector
    for job_elem in job_elements:
        try:
            # Extract data based on the selector that worked
            if used_selector == "div.list-offers div.offer":
                # Process offers from div.list-offers
                title_elem = job_elem.select_one("h3.title") or job_elem.select_one("div.title")
                link_elem = job_elem.select_one("a")
                location_elem = job_elem.select_one("div.location") or job_elem.select_one("span.location")
                date_elem = job_elem.select_one("div.date") or job_elem.select_one("span.date")
                
            elif used_selector == "table.offer-table tr.offer-item":
                # Process offers from table.offer-table
                title_elem = job_elem.select_one("td.offer-title a")
                link_elem = title_elem
                location_elem = job_elem.select_one("td.offer-location")
                date_elem = job_elem.select_one("td.offer-date")
                
            else:
                # Generic approach for other selectors
                title_elem = job_elem.select_one("h3") or job_elem.select_one("h4") or job_elem.select_one("div.title")
                link_elem = job_elem.select_one("a")
                location_elem = job_elem.select_one("div.location") or job_elem.select_one("span.location")
                date_elem = job_elem.select_one("div.date") or job_elem.select_one("span.date")
            
            # Skip if we don't have title or link
            if not title_elem or not link_elem:
                continue
                
            title = title_elem.get_text(strip=True)
            
            # Get link
            link = link_elem['href']
            if not link.startswith('http'):
                link = f"https://jobs.inria.fr{link}" if link.startswith('/') else f"https://jobs.inria.fr/{link}"
            
            # Extract job_id from the link or element id
            job_id = job_elem.get('id', '')
            if not job_id:
                job_id_match = re.search(r'offre/(\w+)', link)
                job_id = job_id_match.group(1) if job_id_match else f"job-{len(jobs)}"
            job_id = job_id.replace('offer-', '')
            
            # Get location and date
            location = location_elem.get_text(strip=True) if location_elem else "Unknown"
            posted_date = date_elem.get_text(strip=True) if date_elem else "Unknown"
            
            logger.info(f"Processing job: {title}")
            
            # Get additional details from the job's detail page
            details = fetch_job_details(link)
            
            # Only include PhD positions
            if not details.get('is_phd', False) and not is_phd_position(title, details.get('full_description', '')):
                logger.debug(f"Skipping non-PhD position: {title}")
                continue
            
            # Create a summary of the job description
            summary = create_summary(details.get('full_description', ''))
            
            jobs.append({
                'job_id': job_id,
                'title': title,
                'location': location,
                'team': details.get('team', ''),
                'posted_date': posted_date,
                'deadline': details.get('deadline', ''),
                'summary': summary,
                'link': link,
                'keywords': details.get('keywords', ''),
                'supervisor': details.get('supervisor', ''),
                'funding': details.get('funding', ''),
                'is_phd': True,
                'last_updated': datetime.now().isoformat()
            })
            
            # Be polite with the server
            time.sleep(2)
        except Exception as ex:
            logger.warning(f"Failed to parse a job offer: {ex}")
    
    logger.info(f"Fetched {len(jobs)} PhD positions")
    return jobs

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
                        job_id, title, location, team, posted_date, deadline, 
                        summary, link, keywords, supervisor, funding, is_phd, last_updated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job['job_id'], job['title'], job['location'], job.get('team', ''), job['posted_date'], 
                    job.get('deadline', ''), job['summary'], job['link'], job.get('keywords', ''),
                    job.get('supervisor', ''), job.get('funding', ''), job.get('is_phd', True), job['last_updated']
                ))
                new_jobs += 1
            else:
                # Update existing job
                c.execute('''
                    UPDATE jobs
                    SET title = ?, location = ?, team = ?, posted_date = ?, deadline = ?, 
                        summary = ?, link = ?, keywords = ?, supervisor = ?, funding = ?, 
                        is_phd = ?, last_updated = ?
                    WHERE job_id = ?
                ''', (
                    job['title'], job['location'], job.get('team', ''), job['posted_date'], 
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
        worksheet.set_column('E:E', 12)  # Posted date
        worksheet.set_column('F:F', 12)  # Deadline
        worksheet.set_column('G:G', 60)  # Summary
        worksheet.set_column('H:H', 40)  # Link
        worksheet.set_column('I:I', 30)  # Keywords
        worksheet.set_column('J:J', 25)  # Supervisor
        worksheet.set_column('K:K', 25)  # Funding
    
    logger.info(f"Exported data to {filename}")
    return filename

def main():
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