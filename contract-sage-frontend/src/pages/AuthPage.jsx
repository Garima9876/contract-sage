import React from "react";
import { Outlet } from "react-router-dom";
import logoBg from "../assets/logo-bg.png";

const AuthPage = () => {
  return (
    <div className="flex h-screen">
      {/* Left side - background image */}
      <div
        className="hidden md:flex md:w-1/2 bg-cover bg-center relative items-center justify-center"
        style={{
          backgroundImage: `linear-gradient(180.18deg, rgba(73, 77, 99, 0) 0.17%, #191B2F 90.29%), url(${logoBg})`,
        }}
      >
        <div className="absolute bottom-0 p-8 md:p-4 text-center text-white">
          <h1 className="extra-large-text font-heading italic font-extrabold mb-2">
            CONTRACT SAGE
          </h1>
          <p className="font-semibold mb-2 p-5 md:text-lg lg:text-3xl">
            AI-Powered Legal Document Analyzer & Fraud Detection
          </p>
        </div>
      </div>

      {/* Right side - form */}
      <div className="flex w-full md:w-1/2 items-center justify-center bg-gray-50">
        <div className="w-5/8 margin-responsive" style={{ maxWidth: "550px" }}>
          <Outlet /> {/* Render the child routes here */}
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
