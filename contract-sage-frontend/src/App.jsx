// src/App.jsx
import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  Navigate,
} from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import AuthPage from "./pages/AuthPage"; // a layout component for auth routes
import Dashboard from "./pages/Dashboard";
import ProtectedRoute from "./components/ProtectedRoute";
import Login from "./components/Login";
import Register from "./components/Register";

function App() {
  const [token, setToken] = useState(localStorage.getItem("token") || "");

  // Persist token changes in localStorage
  useEffect(() => {
    if (token) {
      localStorage.setItem("token", token);
    } else {
      localStorage.removeItem("token");
    }
  }, [token]);

  return (
    <>
      <ToastContainer />
      <Router>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" />} />
          <Route path="/dashboard" element={
            <ProtectedRoute token={token}>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/auth" element={<AuthPage />}>
            <Route
              path="login"
              element={<Login onLogin={(tok) => setToken(tok)} />}
            />
            <Route path="register" element={<Register />} />
          </Route>
        </Routes>
      </Router>
    </>
  );
}

export default App;
