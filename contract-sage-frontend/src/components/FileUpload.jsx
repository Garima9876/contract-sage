import React, { useState } from 'react';

const FileUpload = ({ onFileSelect }) => {
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      onFileSelect(file);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      <label className="w-full text-center cursor-pointer bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300">
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
          📄 Selected file: <span className="text-green-700">{fileName}</span>
        </p>
      )}
    </div>
  );
};

export default FileUpload;
