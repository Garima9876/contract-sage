// src/pages/Dashboard.jsx
import React, { useState, useEffect } from "react";
import FileUpload from "../components/FileUpload";
import SummaryDisplay from "../components/SummaryDisplay";
import { getAllDocuments } from '../api/api';

const Dashboard = () => {
  const [docs, setDocs] = useState([]);
  const [summary, setSummary] = useState([]);

  useEffect(() => {
    getAllDocuments().then((res) => {
      setDocs(res.data);
    });
  }, []);

  return (
    <div className="w-full min-h-screen bg-gray-50 flex flex-col items-center p-6">
      <header className="w-full max-w-4xl text-center py-6">
        <h1 className="text-3xl font-bold text-sky-700 mb-2">
          ContractSage Dashboard
        </h1>
      </header>
      <main className="w-full max-w-4xl space-y-6">
        <div className="bg-white shadow-md rounded-2xl p-6">
          <FileUpload onSummary={(data) => setSummary(data)} />
        </div>
        <div className="bg-white shadow-md rounded-2xl p-6">
          <SummaryDisplay summary={summary} documents={docs} />
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
