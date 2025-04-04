import React from "react";
import FileUpload from "./components/FileUpload";
import SummaryDisplay from "./components/SummaryDisplay";

function App() {
  return (
    <div className="w-full min-h-screen bg-gray-50 flex flex-col items-center p-6">
      <header className="w-full max-w-4xl text-center py-6">
        <h1 className="text-3xl font-bold text-green-700 mb-2">
          ContractSage
        </h1>
        <p className="text-gray-600 text-lg">
          AI-Powered Legal Document Analyzer & Fraud Detection
        </p>
      </header>

      <main className="w-full max-w-4xl space-y-6">
        <div className="bg-white shadow-md rounded-2xl p-6">
          <FileUpload />
        </div>

        <div className="bg-white shadow-md rounded-2xl p-6">
          <SummaryDisplay />
        </div>
      </main>
    </div>
  );
}

export default App;
