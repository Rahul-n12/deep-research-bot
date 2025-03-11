🤖 Deep Research Bot 📚
An AI-powered research assistant for automated literature review generation.
🚀 Overview
Deep Research Bot is an AI-driven system that automates the research process by:
•	Searching for academic papers using Google Scholar via SerperDev API.
•	Downloading research papers using DOI links via Sci-Hub.
•	Extracting references and performing recursive research up to 3 levels deep.
•	Summarizing research papers using Gemini AI.
•	Generating a structured review paper with key insights from multiple research papers.
•	Providing a user-friendly Streamlit interface for ease of use.
•	Allowing users to download the final review paper in Word (.docx) format.
________________________________________
🎯 Features
✅ Automated DOI extraction and paper retrieval 📄
✅ Recursive reference-based research (Up to 3 levels deep) 🔄
✅ AI-powered summarization using Gemini AI 🤖
✅ Google Scholar integration via SerperDev API 🔍
✅ User-friendly Streamlit interface 🖥️
✅ Downloadable Review Paper in .docx format 📂
✅ Seamless deployment on Google Cloud Platform (GCP) ☁️
________________________________________
🛠️ Installation & Setup
1️. Clone the Repository
git clone https://github.com/Rahul-n12/deep-research-bot.git
cd deep-research-bot
2️. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
pip install -r requirements.txt
3️. Set Up API Keys
Create a .env file in the project directory and add your API keys:
GEMINI_API_KEY=your_gemini_api_key
SERPERDEV_KEY=your_serperdev_api_key
4️. Run the Application
Launch the Streamlit interface:
streamlit run bot.py
________________________________________
🏗️ Project Structure
📂 deep-research-bot
│── 📜 bot.py              # Main script (AI agents, Streamlit UI)
│── 📜 requirements.txt    # Python dependencies
│── 📜 Dockerfile          # For cloud deployment
│── 📜 .env                # API keys (not included in Git)
│── 📜 README.md           # Project documentation
│── 📂 papers/             # Downloaded research papers
│── 📂 output/             # Generated review papers
________________________________________
🎮 How It Works
1️. User Input:
•	Users enter a research topic manually or select from a dropdown list.
2. Search & Download:
•	The bot retrieves relevant papers using Google Scholar and downloads them using Sci-Hub.
3️. Recursive Research:
•	Extracts references from downloaded papers and follows citations up to 3 levels deep.
4️. Summarization:
•	AI-powered text processing extracts key insights from papers.
5️. Review Generation:
•	A structured review paper is created based on extracted insights.
6️. Output & Download:
•	Users can view and download the final review paper in .docx format.
________________________________________
📸 Sample Interaction
Example Scenario:
User Input: "Artificial Intelligence in healthcare"
Bot Actions:
✅ Fetches research papers via Google Scholar
✅ Extracts DOIs and downloads up to 100 papers
✅ Scrapes and processes PDFs to extract references
✅ Expands research recursively up to 3 levels deep
✅ Uses AI summarization for key insights
✅ Compiles a structured review paper
✅ Provides a downloadable .docx file
________________________________________
🚧 Limitations
❌ API Rate Limits – SerperDev and Gemini APIs have restrictions.
❌ Dependency on Sci-Hub – Service availability is not guaranteed.
❌ Processing Time – Recursive reference analysis may slow down execution.
❌ Web Scraping Risks – Websites may change, affecting scraping.
________________________________________
🔮 Future Scope
🔹 Integration with IEEE Xplore & PubMed for broader research coverage.
🔹 Advanced AI Summarization Models (GPT-4, BERT) for better text analysis.
🔹 Graph-Based Research Mapping for visualizing research connections.
🔹 Multi-User Collaboration for shared academic research.
🔹 Citation Management System to format bibliographies.
________________________________________
📝 License
This project is licensed under the MIT License.
________________________________________
⭐ Acknowledgments
Special thanks to:
•	Google Scholar & SerperDev API for academic paper retrieval.
•	Sci-Hub for free access to research papers.
•	Gemini AI for summarization and review generation.
•	Streamlit for providing an interactive UI framework.

