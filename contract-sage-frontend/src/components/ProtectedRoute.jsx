// src/components/ProtectedRoute.jsx
import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ token, children }) => {
  if (!token) {
    return <Navigate to="/auth/login" />;
  }
  return children;
};

export default ProtectedRoute;
