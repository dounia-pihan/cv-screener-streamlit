# 📄 CV Sorting & Scoring App (Streamlit) *(personnal project)*
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Altair](https://img.shields.io/badge/Altair-Visualization-F05C3C?logo=altair&logoColor=white)](https://altair-viz.github.io/)
[![pdfplumber](https://img.shields.io/badge/pdfplumber-PDF%20Parsing-yellow?logo=adobeacrobatreader&logoColor=000)](https://github.com/jsvine/pdfplumber)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


An intelligent Streamlit web application that automatically analyzes and ranks PDF resumes based on customizable HR criteria.  
Fully built with **Python**, designed for recruiters, data analysts, and HR professionals.

---

## Project Description

This application was designed to help **recruiters and HR professionals** quickly identify the most relevant resumes based on their specific needs.  
It analyzes **PDF resumes** by extracting and normalizing their content (using `pdfplumber`), then applies a **weighted scoring system** based on configurable criteria such as mandatory, optional, and redhibitory keywords, as well as experience level, contract type, languages, and more.

The user can then:
- Upload multiple résumés simultaneously  
- Adjust criteria and preferences in real time  
- Obtain an automatic ranking with interactive charts  
- Export complete results (CSV, XLSX, ZIP)

--- 

## ⚙️ How It Works

### Global Workflow

The application follows a simple and transparent process — from resume upload to the final ranking.  
Here are the main steps:

1. **File Import**
   - The user can upload multiple PDF resumes or select an entire folder.
   - All files are automatically stored in a temporary directory.

2. **Text Extraction**
   - Each PDF’s content is converted into raw text using the `pdfplumber` library.
   - A cleaning and normalization phase removes accents, special characters, and extra spaces to ensure consistent text comparison.

3. **Criteria Configuration**
   - The user defines preferences through an intuitive interface:
     - **Mandatory**, **optional**, or **redhibitory** keywords  
     - Filters for **contract type**, **work time**, **education**, **city**, **languages**, or **technologies**
     - A **minimum score threshold** and an optional **redhibitory category** (if any rule in this category is not met → the résumé is automatically rejected)

4. **Analysis & Scoring**
   - Each resume is analyzed: keyword occurrences and contextual elements (education, contract, experience, etc.) are detected and compared with preferences.
   - The app then applies a **weighted scoring system** (see next section).

5. **Results & Visualization**
   - The final dashboard displays:
     - A ranked table of analyzed resumes (with total score and reasoning)
     - A **pie chart** showing *Accepted vs Rejected*
     - A **histogram** of score distribution
     - A **“Top Profiles”** section highlighting the 6 best candidates

6. **Export**
   - All results can be exported as **CSV**, **XLSX**, or a **ZIP** file containing the retained resumes.

---

### Scoring System

Each resume goes through a **multi-criteria evaluation engine**, balancing mandatory, bonus, and elimination rules.

| **Type of Rule** | **Description** | **Impact on Final Score** |
|------------------|------------------|----------------------------|
| **Mandatory keywords** | Must appear in the text. | Absence → automatic rejection |
| **Redhibitory keywords** | Missing these words leads to elimination. | Absence → elimination regardless of score |
| **Optional keywords** | Give extra points if present. | +1 point per match (up to 10 pts) |
| **Contract type** | Matches the recruiter's preferences. | +1 if match / -1 if mismatch |
| **Work time** | Matches the recruiter's preferences. | +1 if match / -1 if mismatch |
| **Experience** | Years of experience ≥ minimum required. | +1 if threshold met |
| **Education** | Matches preferred levels. | +1 if match / 0 if neutral / -1 if mismatch |
| **City / Location** | Bonus for preferred locations. | +1 if match |
| **Languages / Skills** | Bonus depending on the number detected. | +1 to +5 depending on occurrences |

> **Redhibitory Category Feature**  
> The user can select a single category (e.g., *education*, *city*, *experience*, *contract type*, etc.) as **redhibitory**.  
> If a resume fails to meet the criteria in this category, it is automatically excluded, even if its overall score is high.

--- 

## Interface Overview

Below is an overview of the **Streamlit application interface**, showcasing its clean and intuitive design:

- The **left panel** allows users to configure all filtering and scoring parameters.  
- The **right panel** displays the results, charts, and ranked CVs in real time.  
- Each section is visually separated to ensure readability and a smooth user experience.

> 🧡 The layout and color palette were customized through a dedicated CSS theme and `config.toml` file for a modern look.

![Interface Overview](assets/Screenshot_1.png)

---

## Local Installation

```bash
git clone https://github.com/dounia-pihan/cv-screener-streamlit.git
cd cv-screener-streamlit
pip install -r requirements.txt
streamlit run app.py
```
Then open your browser at http://localhost:8501 

--- 

## Tech Stack

- **Python 3.10+**
- **Streamlit** → Web app framework for interactive dashboards  
- **pdfplumber** → Text extraction from PDF résumés  
- **pandas** → Data manipulation and tabular output  
- **Altair** → Interactive data visualization (pie charts, histograms)  
- **xlsxwriter / openpyxl** → Excel export support  
- **zipfile / io** → In-memory ZIP generation for selected CVs  
- **Custom CSS + config.toml** → Visual theming and layout styling


