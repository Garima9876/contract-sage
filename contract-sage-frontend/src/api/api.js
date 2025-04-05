import axios from 'axios';

// Adjust baseURL depending on your backend URL
const API = axios.create({
  baseURL: 'http://localhost:8000', // replace with actual server URL if deployed
});

// ========== AUTH ROUTES ==========

// Register user with just email
export const registerUser = (email) => API.post("/api/auth/register", { email });

// Activate account using token and new password
export const activateUser = (token, password) => 
  API.post("/api/auth/activate", null, {
    params: { token, password }
  });

// Login user with email and password
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

