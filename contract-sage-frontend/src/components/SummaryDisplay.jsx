import React from 'react';

const SummaryDisplay = ({ summary, documents = [] }) => {
  return (
    <div className="space-y-8">
      {/* Summary Section */}
      {/* <div>
        <h2 className="text-2xl font-semibold text-sky-700 border-b pb-2">
          📑 Document Summary
        </h2>
        <p className="text-gray-800 leading-relaxed">
          {summary || 'No summary available yet. Please upload a document.'}
        </p>
      </div> */}

      {/* Segmentation Display */}
      {summary?.segmentation && (
        <div>
          <h2 className="text-2xl font-semibold text-sky-700 border-b pb-2">
            🧩 LLM Segmentation
          </h2>

          <div className="space-y-2 mt-4">
            {summary.segmentation.segments.map((segment, idx) => (
              <div key={idx} className="p-4 border border-gray-200 rounded-md shadow-sm bg-white">
                <p className="text-gray-800">
                  <span className="font-semibold text-sky-600">{segment.segment}:</span> {segment.sentence}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-medium text-gray-700 mb-2">📊 Segment Counts</h3>
            <ul className="list-disc list-inside text-gray-700">
              {Object.entries(summary.segmentation.segment_counts).map(([label, count]) => (
                <li key={label}>
                  <span className="font-semibold">{label}</span>: {count}
                </li>
              ))}
              <li className="mt-2 font-medium text-gray-800">
                Total Segments: {summary.segmentation.total_segments}
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Documents Table Section */}
      <div>
        <h2 className="text-2xl font-semibold text-sky-700 border-b pb-2">
          📂 Submitted Documents
        </h2>
        {documents.length === 0 ? (
          <p className="text-gray-600">No documents found.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border border-gray-300">
            <table className="min-w-full divide-y divide-gray-300">
              <thead className="bg-gray-100 text-left">
                <tr>
                  <th className="px-4 py-2 font-medium text-gray-700">ID</th>
                  <th className="px-4 py-2 font-medium text-gray-700">Filename</th>
                  <th className="px-4 py-2 font-medium text-gray-700">Summary</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {documents.map((doc) => (
                  <tr key={doc.id}>
                    <td className="px-4 py-2">{doc.id}</td>
                    <td className="px-4 py-2">{doc.filename}</td>
                    <td className="px-4 py-2 text-gray-700">
                      {doc.summary.length > 100
                        ? doc.summary.slice(0, 100) + '...'
                        : doc.summary}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default SummaryDisplay;