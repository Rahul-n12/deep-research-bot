import streamlit as st
import requests
import os
import re
import PyPDF2
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List
from google.generativeai import configure, GenerativeModel
import json
import logging
import time
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from Kaggle Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPERDEV_KEY = os.getenv("SERPERDEV_KEY")


# Check if keys are loaded properly
if not GEMINI_API_KEY or not SERPERDEV_KEY:
    raise ValueError("API Keys could not be loaded. Check the .env file.")

# Set Gemini API key
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("gemini-2.0-flash-001")

# Sci-Hub base URL
SCI_HUB_URL = "https://sci-hub.se/"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# User-Agent list for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)
def search_serperdev_with_retry(url, params, max_retries=5, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            headers = {"X-API-KEY": SERPERDEV_KEY, "Content-Type": "application/json"}
            data = json.dumps(params)
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retries += 1
                time.sleep(retry_delay * retries)  # Exponential backoff
            else:
                raise
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: {e}")
            return None
    st.error("Max retries exceeded for SerperDev request.")
    return None

    
# Define AI Agents
class ManagerAgent:
    def __init__(self):
        self.gemini_model = GenerativeModel("gemini-2.0-flash-001")
        self.review_writing_training()

    def review_writing_training(self):
        training_data = self.gather_review_training_data()
        prompt = f"""
        You are an expert review paper writer.
        Learn from the following training data to write a well-structured review paper:
        {training_data}
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            st.write("Manager Agent Training Completed")
        except Exception as e:
            st.error(f"Error training Manager Agent: {e}")
            logging.error(f"Error training Manager Agent: {e}")

    def gather_review_training_data(self):
        training_data = ""
        search_engines = ["google", "youtube"]  # Added "youtube"
        for engine in search_engines:
            results = self.search_review_writing_resources(engine)
            for result in results:
                training_data += result + "\n"
        return training_data

    def search_review_writing_resources(self, search_engine):
        if search_engine == "google":
            return self.search_google_review_resources()
        elif search_engine == "youtube":
            return self.search_youtube_review_resources()  # Call YouTube search
        else:
            return

    def search_google_review_resources(self):
        results = []
        url = "https://google.serper.dev/search"
        params = {
            "q": "how to write a good review paper",
            "gl": "us",
            "hl": "en",
            "num": 3,
        }
        data = search_serperdev_with_retry(url, params)
        if data and data.get("organic"):
            for item in data["organic"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                results.append(f"Source: {title}\nSummary: {snippet}\n")
        return results

    def search_youtube_review_resources(self):
        results = []
        url = "https://google.serper.dev/videos"
        params = {
            "q": "how to write a good review paper",
            "gl": "us",
            "hl": "en",
            "num": 2,
        }
        data = search_serperdev_with_retry(url, params)
        if data and data.get("videos"):
            for item in data["videos"]:
                title = item.get("title", "")
                snippet = item.get("description", "")
                results.append(f"Source: YouTube - {title}\nSummary: {snippet}\n")
        return results

    
    def oversee_research(self, topic: str):
        
        st.write(f"Starting deep research on: {topic}")
        
        research_agent = ResearchAgent()
        papers = research_agent.search_papers(topic)
        
    
        doi_extractor = DOIExtractorAgent()
        doi_list = [doi_extractor.extract_doi(paper.get('link', '') + ' ' + paper.get('snippet', '')) for paper in papers]
        doi_list = list(filter(None, doi_list))  # Remove empty DOIs
    
        downloader = PaperDownloaderAgent()
        pdf_files = []
        downloaded_count = 0
        max_papers = 100
        paper_index = 0
    
        while downloaded_count < max_papers and paper_index < len(papers):
            
            paper = papers[paper_index]
            doi = doi_list[paper_index] if paper_index < len(doi_list) else None
            pdf_path = downloader.download_paper(paper, doi)
            if pdf_path:
                pdf_files.append(pdf_path)
                downloaded_count += 1
            
            paper_index += 1
    
        pdf_files = [pdf for pdf in pdf_files if pdf and os.path.exists(pdf)]  # Ensure valid paths
    
        if not pdf_files:
            logging.info("No valid papers were downloaded. Please try again with a different topic.")
            return
    
        reference_extractor = ReferenceExtractorAgent()
        summarizer = SummarizationAgent()
        summaries = []
        references = []
        processed_papers = []
    
        def process_paper(pdf, depth=0):
            nonlocal downloaded_count
            if downloaded_count >= max_papers:
                return
            if pdf in processed_papers:
                return
            processed_papers.append(pdf)
            
            
            
            extracted_refs = reference_extractor.extract_references(pdf)
            
            # Merge broken references into complete ones
            merged_refs = []
            current_ref = ""
    
            for line in extracted_refs:
                line = line.strip()
                if re.match(r"^\d+\.", line):  # If the line starts with a number, it's a new reference
                    if current_ref:  
                        merged_refs.append(current_ref)  # Save the previous reference
                    current_ref = line  # Start a new reference
                else:
                    current_ref += " " + line  # Append to the current reference
    
            if current_ref:
                merged_refs.append(current_ref)  # Add last reference
    
            # Remove duplicates and clean up references
            cleaned_refs = list(set([re.sub(r'\s+', ' ', ref).strip() for ref in merged_refs if len(ref.strip()) > 10]))
            references.extend(cleaned_refs)
    
            summary = summarizer.summarize_paper(pdf)
            summaries.append(summary)
    
            logging.info(f"**Manager Agent:** Paper Summary: {summary}")
            logging.info(f"**Manager Agent:** Extracted References: {cleaned_refs}")

            
            # Check again before going deeper
            if downloaded_count >= max_papers:
                logging.info(f"Reached max paper limit ({max_papers}). Stopping further reference searches.")
                return  # Stop all further processing
    
            if depth < 3:
                for ref in cleaned_refs:
                    if downloaded_count >= max_papers:
                        break
                    
                    
                    logging.info(f"**Manager Agent:** Searching for papers related to reference: {ref}")
                    
                    search_results = research_agent.search_papers(query=f"{topic} {ref}", max_results=1)
                    
                    if search_results:
                        ref_paper = search_results[0]
                        ref_doi = doi_extractor.extract_doi(ref_paper.get('link', '') + ' ' + ref_paper.get('snippet', ''))
                        if ref_doi:
                            ref_pdf = downloader.download_paper(ref_paper, ref_doi)
                            if ref_pdf:
                                downloaded_count += 1
                                if downloaded_count < max_papers:
                                    process_paper(ref_pdf, depth + 1)
                                    
                                
                                
    
        for pdf in pdf_files:
            process_paper(pdf)
    
        review_writer = ReviewWriterAgent()
        review_paper = review_writer.generate_review(summaries, references)
    
        st.write(f"**Manager Agent:** Generated Review Paper:")
        st.text_area("Generated Review Paper:", review_paper, height=500)

        return review_paper


class ResearchAgent:
    def search_papers(self, query: str, max_results=5) -> List[dict]:
        logging.info(f"Searching for papers on: {query}")
        url = "https://google.serper.dev/search"
        params = {
            "q": f"{query} scholar", #added scholar to the query.
            "gl": "us",
            "hl": "en",
            "num": max_results,
        }
        data = search_serperdev_with_retry(url, params) #using the retry function from the previous response.

        papers = []
        if data and data.get("organic"):
            for result in data["organic"]:
                papers.append({
                    "title": result.get("title", ""),
                    "doi": result.get("link", ""),
                    "url": result.get("link", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })

        if not papers:
            logging.warning(f"No results found for '{query}' on Google Scholar via SerperDev.")
        return papers

class DOIExtractorAgent:
    def extract_doi(self, text: str) -> str:
        doi_patterns = [
            re.compile(r"10.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE),
            re.compile(r"doi.org/(10.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE),
            re.compile(r"dx.doi.org/(10.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE),
            re.compile(r"https?://[^\']*(10.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
        ]
        
        for pattern in doi_patterns:
            match = pattern.search(text)
            if match:
                extracted_doi = match.group(1) if 'doi.org/' in match.group(0) else match.group(0)
                if extracted_doi.lower().startswith("10."):
                    return extracted_doi
        
        return ""

class PaperDownloaderAgent:
    def download_paper(self, paper: dict, doi: str) -> str:
        title = paper.get('title', 'paper')
        if doi:
            pdf_path = self.download_from_scihub(doi, title)
            if pdf_path:
                return pdf_path

        link = paper.get('link', '')
        if link:
            pdf_path = self.attempt_download(link, title, paper)
            if pdf_path:
                return pdf_path

        logging.warning(f"Failed to download paper: {title}. Counting towards max_papers and trying next.")
        return None  # Ensures failed attempts count towards max_papers


    def download_from_scihub(self, doi: str, title: str) -> str:
        headers = {
            "User-Agent": get_random_user_agent()
        }
        sci_hub_url = f"{SCI_HUB_URL}{doi}"
        try:
            response = requests.get(sci_hub_url, timeout=15, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_tag = soup.find('iframe') or soup.find('a', href=re.compile(r"\.pdf$"))
            if pdf_tag:
                pdf_url = urljoin(SCI_HUB_URL, pdf_tag.get('src') or pdf_tag.get('href'))
                if pdf_url and pdf_url.endswith(".pdf"):
                    return self.save_pdf(pdf_url, title)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.warning(f"403 Forbidden error for {sci_hub_url}. Sci-Hub may be having issues.")
                
            else:
                logging.info(f"Error downloading from Sci-Hub: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error downloading from Sci-Hub: {e}")
        return ""
    

    def attempt_download(self, link: str, title: str, paper: dict) -> str:
        headers = {
            "User-Agent": get_random_user_agent()
        }

        try:
            response = requests.get(link, stream=True, timeout=15, headers=headers) #Increased timeout
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            if "application/pdf" in content_type:
                return self.save_pdf(response, title)

            elif "text/html" in content_type:
                return self.scrape_pdf_from_html(response.content, link, title, paper)

            else:
                
                return ""

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.warning(f"403 Forbidden error for {link}. Website may be blocking access.")
                
            else:
                logging.error(f"Error downloading {link}: {e}")
            return ""

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {link}: {e}")
            return ""

    def save_pdf(self, source, title: str) -> str:
        os.makedirs("papers", exist_ok=True)
        pdf_path = f"papers/{title.replace('/', '_')}.pdf"
        try:
            if isinstance(source, requests.Response):
                with open(pdf_path, "wb") as f:
                    for chunk in source.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                pdf_response = requests.get(source, timeout=15)
                pdf_response.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(pdf_response.content)
            return pdf_path
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return ""

    def scrape_pdf_from_html(self, html_content: bytes, base_url: str, title: str, paper: dict) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        pdf_links = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                pdf_links.append(urljoin(base_url, href))

        for link in soup.find_all("a", href=True):
            if "pdf" in link.text.lower() or "download" in link.text.lower():
                href = link["href"]
                pdf_links.append(urljoin(base_url, href))

        if "resources" in paper:
            for resource in paper["resources"]:
                if resource.get("link", "").lower().endswith(".pdf"):
                    pdf_links.append(resource.get("link"))

        for pdf_link in pdf_links:
            try:
                response = requests.get(pdf_link, stream=True, timeout=15)
                response.raise_for_status()
                if "application/pdf" in response.headers.get("Content-Type", ""):
                    return self.save_pdf(response, title)
            except requests.exceptions.RequestException:
                continue

        
        return ""
  
class ReferenceExtractorAgent:
    def extract_references(self, pdf_path: str) -> List[str]:
        """
        Extracts only valid references from a research paper PDF.
        Removes large text blocks and keeps only properly formatted references.
        """
        references = []
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            return references

        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

            # Find the "References" or "Bibliography" section
            ref_section = re.search(r"(References|Bibliography)(.*)", full_text, re.DOTALL)
            if ref_section:
                raw_refs = ref_section.group(2).strip().split("\n")

                # Only keep lines that appear to be valid references (contain citation elements)
                valid_references = []
                current_ref = ""

                for line in raw_refs:
                    line = line.strip()

                    # Check if line starts with a number (common format for references)
                    if re.match(r"^\d+\.\s", line) or re.match(r"^\[\d+\]", line):
                        if current_ref:
                            valid_references.append(current_ref.strip())  # Save previous reference
                        current_ref = line  # Start new reference
                    else:
                        current_ref += " " + line  # Continue merging broken references

                if current_ref:
                    valid_references.append(current_ref.strip())  # Add last reference

                # Apply final cleanup to remove URLs-only and short lines
                references = [
                    ref for ref in valid_references 
                    if len(ref) > 10 and not re.fullmatch(r"https?://\S+", ref)
                ]

        except Exception as e:
            logging.errorr(f"Error extracting references from {pdf_path}: {e}")

        return references [:20]


class SummarizationAgent:
    def summarize_paper(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        try:
            prompt = f"""
            Summarize the following research paper, focusing on the key findings, methodologies, and implications. 
            Provide a concise and accurate summary that captures the essence of the paper in a clear and understandable manner:
            {text}
            """
            response = gemini_model.generate_content(prompt)
            return response.text if response else "Summarization failed."
        except Exception as e:
            st.error(f"Error in Gemini API summarization: {e}")
            return ""

class ReviewWriterAgent:
    def generate_review(self, summary_list: List[str], references: List[str]) -> str:
        combined_text = "\n\n".join(summary_list)

        try:
            # Construct prompt for Gemini API
            prompt = f"""
            Compose a well-structured academic review paper based on the following research summaries. 
            Ensure the review includes an introduction, a critical analysis of the key findings, 
            a discussion of the implications and limitations of the research, and a conclusion that synthesizes the main points. 
            Maintain an objective and academic tone throughout the review:
            
            {combined_text}
            """
            response = gemini_model.generate_content(prompt)
            review_paper = response.text if response else "Review generation failed."

            # **Fix: Format References Properly**
            review_paper += "\n\n## References\n"

            if references:
                cleaned_refs = self.clean_references(references)

                # **Remove extra numbers but keep sequential numbering**
                formatted_refs = []
                seen_refs = set()

                for ref in cleaned_refs:
                    # Remove extra numbers that appear before the main content
                    ref = re.sub(r'^\d+\.\s*\d+\.\s*', '', ref)  # Removes "X. Y."
                    ref = re.sub(r'^\d+\.\s*', '', ref)  # Removes "X."
                    ref = re.sub(r'\s+', ' ', ref).strip()  # Clean extra spaces
                    
                    if ref and ref not in seen_refs:  # Avoid duplicates
                        formatted_refs.append(ref)
                        seen_refs.add(ref)

                # Add formatted references with correct numbering
                review_paper += "\n".join(f"{i+1}. {ref}" for i, ref in enumerate(formatted_refs))
            else:
                review_paper += "No references extracted."

            return review_paper

        except Exception as e:
            st.error(f"Error in Gemini API review generation: {e}")
            return ""

    def clean_references(self, reference_list):
        """
        Cleans extracted references to ensure proper formatting without duplicate numbering or line breaks.
        """
        cleaned_references = []
        seen_references = set()

        for ref in reference_list:
            ref = re.sub(r'\s+', ' ', ref).strip()  # Remove extra whitespace

            # Remove duplicates
            if ref and ref not in seen_references:
                cleaned_references.append(ref)
                seen_references.add(ref)

        return cleaned_references




# Streamlit interface
st.title("📚🤖 AI-Powered Research Paper Review Generator")
st.markdown("### Automate research paper review generation with AI!")

# User Input Section
st.markdown("#### ✍️ Enter your own topic or choose a research topic below:")

# Predefined topics for dropdown selection
default_topics = [
    "Artificial Intelligence in Healthcare",
    "Machine Learning for Climate Change",
    "Blockchain Applications in Finance",
    "AI-Powered Cybersecurity",
    "Natural Language Processing in Education"
]

# Manual entry input
custom_topic = st.text_input("📝 Enter your research topic:")

# Dropdown selection
selected_topic = st.selectbox("🎯 Or select a topic from the list:", ["-- Select a topic --"] + default_topics)

# Determine final topic
topic = custom_topic.strip() if custom_topic.strip() else (selected_topic if selected_topic != "-- Select a topic --" else None)

# Button to start research
if st.button("🚀 Start Research"):
    if not topic:
        st.warning("⚠️ Please enter or select a research topic before proceeding!")
    else:
        manager = ManagerAgent()
        
        # Display a spinner while research is in progress
        with st.spinner(f"🔄 Researching: **{topic}**... Please wait."):
            review_paper = manager.oversee_research(topic)

        if review_paper:
            st.success("✅ Research completed! Check the generated review paper below.")

    # Convert review paper to bytes for download
    review_text_bytes = review_paper.encode("utf-8")

    # Offer a download button for plain text file
    st.download_button(
        label="📥 Download Review Paper as .txt",
        data=review_text_bytes,
        file_name=f"{topic.replace(' ', '_')}_Review_Paper.txt",
        mime="text/plain",
    )

    st.markdown("---")

