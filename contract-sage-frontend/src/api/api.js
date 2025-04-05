import axios from 'axios';

// Adjust baseURL depending on your backend URL
const API = axios.create({
  baseURL: 'http://localhost:8000', // replace with actual server URL if deployed
});

// ========== AUTH ROUTES ==========

// Register user
export const registerUser = (userData) => API.post("/api/auth/register", userData);

// Login user
export const loginUser = (credentials) => API.post("/api/auth/login", credentials);


// ========== FILE UPLOAD + SUMMARIZATION ==========

// Upload a document and summarize it
export const uploadAndSummarizeFile = (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return API.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  };
  


// ========== TEXT SUMMARIZATION ==========

// Summarize raw text
export const summarizeText = (text) =>
  API.post('/summarize/text', { text });


// ========== FRAUD DETECTION ==========

// Detect fraud in uploaded file
export const detectFraudInFile = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return API.post('/detect/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Detect fraud in raw text
export const detectFraudInText = (text) =>
  API.post('/detect/text', { text });

