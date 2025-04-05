import React from 'react';

const SummaryDisplay = ({ summary }) => {
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold text-sky-700 border-b pb-2">
        📑 Document Summary
      </h2>
      <p className="text-gray-800 leading-relaxed">
        {summary || 'No summary available yet. Please upload a document.'}
      </p>
    </div>
  );
};

export default SummaryDisplay;
