# ContractSage

## Problem Statement

Legal professionals and businesses often spend countless hours reviewing lengthy legal documents. This process is not only time-consuming but also prone to human error, potentially overlooking critical clauses or signs of fraud. ContractSage aims to address these challenges by leveraging AI to automate the analysis of legal contracts, enabling efficient document review, accurate clause extraction, and fraud detection.

## Solution Overview

ContractSage is an AI-powered legal document analyzer designed to:
- **Summarize Contracts:** Automatically extract key clauses, obligations, deadlines, and penalties.
- **Detect Fraud:** Utilize anomaly detection techniques to flag inconsistent or risky clauses that may indicate fraudulent intent.
- **Extract Entities:** Employ Named Entity Recognition (NER) to identify and extract crucial details such as names, dates, monetary values, and legal terms.
- **Multi-Language Support:** Analyze documents in English and other languages, broadening its global applicability.

### Key Components

#### Frontend (React + Vite)
- **User Interface:** A modern, responsive UI built using React with Vite, featuring:
  - A clean landing page with login and dashboard navigation.
  - A FileUpload component for users to submit legal documents.
  - A SummaryDisplay component to show AI-generated document summaries.
- **Authentication:** Basic login functionality to secure access.
- **API Integration:** Uses Axios/Fetch to communicate with the backend endpoints.

#### Backend (FastAPI)
- **API Server:** Developed with FastAPI and served using Uvicorn, providing RESTful endpoints for:
  - User authentication (login and registration).
  - Document upload and processing.
- **Database Integration:** Uses SQLAlchemy with SQLite (for development) or PostgreSQL (for production) to manage user data and document records.
- **AI Integration:** Integrates AI modules (NLP models, fraud detection logic) to process legal documents and generate summaries.

## Setup & Usage Instructions

### Prerequisites
- **Node.js** (v14 or later) for the frontend.
- **Python** (v3.8 or later) and **pip** for the backend.
- **Git** for version control.

### Frontend Setup

1. **Navigate to the Frontend Directory:**
   ```bash
   cd contract-sage-frontend
2. **Install Dependencies:**
   npm install
3. **Start the Development Server:**
   npm run dev

   Visit the URL provided http://localhost:5173/.

### Backend Setup

1. **Navigate to the Backend Directory:**
   ```bash
   cd contract-sage-backend
2. **Create and Activate a Virtual Environment:**
   python3 -m venv venv
   source venv/bin/activate
3. **Install Required Python Packages:**
   pip install -r requirements.txt
4. **Run the FastAPI Server:**
   uvicorn app.main:app --reload

   Your API server will run at http://127.0.0.1:8000.