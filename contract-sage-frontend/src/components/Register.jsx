// src/components/Register.jsx
import React, { useState } from "react";
import TextInput from "../shared/TextInput";
import PrimaryButton from "../shared/PrimaryButton";
import { registerUser } from "../api/api";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const Register = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      await registerUser(email);
      toast.success("Activation link sent to your email!");
      navigate("/auth/login"); // Navigate to activation page
    } catch (err) {
      console.error("Registration error:", err);
      toast.error(err.response?.data?.detail || "Registration failed.");
    }
  };

  return (
    <div
      className="bg-white rounded-[20px] padding-responsive"
      style={{ boxShadow: "0px 9px 34px 0px #0000001A" }}
    >
      <div className="flex items-center mt-4">
        <span className="pr-2 mb-1 text-[var(--color-dark-grey)] font-heading large-text font-semibold">
          Welcome to Contract Sage!
        </span>
      </div>
      <div className="text-grey small-text mb-8">
        Please enter your email to register
      </div>
      <form onSubmit={handleRegister} className="flex flex-col gap-4">
        <TextInput
          id="email"
          type="email"
          label="Email"
          placeholder="Enter Email"
          className="mb-5"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <PrimaryButton type="submit" label="Send Activation Link" className="w-full" />
      </form>
      <p className="mt-5">
        Already have an account?{" "}
        <button
          onClick={() => navigate("/auth/login")}
          className="text-blue underline"
        >
          Login here
        </button>
      </p>
    </div>
  );
};

export default Register;
