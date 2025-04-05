import React, { useState } from 'react';
import { uploadAndSummarizeFile } from '../api/api';
import { toast } from "react-toastify";

const FileUpload = ({ onSummary }) => {
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);

    try {
      const response = await uploadAndSummarizeFile(file);
      toast.success("File processed successfully!");
      onSummary(response.data.summary);
    } catch (error) {
      toast.error(
        error.response?.data?.detail || "Error processing the file."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      <label className="w-full text-center cursor-pointer bg-sky-600 hover:bg-sky-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300">
        Upload Legal Document
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
          className="hidden"
        />
      </label>
      {fileName && (
        <p className="text-gray-700 font-medium">
          📄 Selected file: <span className="text-sky-700">{fileName}</span>
        </p>
      )}
      {loading && <p className="text-sm text-gray-500">Processing file...</p>}
    </div>
  );
};

export default FileUpload;
