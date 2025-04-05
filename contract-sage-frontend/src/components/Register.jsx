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
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }
    try {
      await registerUser({ email, password });
      toast.success("Registration successful! Please log in.");
      navigate("/auth/login");
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
        Please register to create your account
      </div>
      <form onSubmit={handleRegister} className="flex flex-col gap-4">
        <TextInput
          id="email"
          type="email"
          label="Email"
          placeholder="Enter Email"
          className="mb-2"
          value={email} // Bind state to input value
          onChange={(e) => setEmail(e.target.value)} // Update email state
        />
        <TextInput
          id="password"
          type="password"
          label="Password"
          placeholder="Enter Password"
          className="mb-2"
          value={password} // Bind state to input value
          onChange={(e) => setPassword(e.target.value)} // Update password state
        />
        <TextInput
          id="confirm-password"
          type="password"
          label="Confirm Password"
          placeholder="Confirm Password"
          className="mb-5"
          value={confirmPassword} // Bind state to input value
          onChange={(e) => setConfirmPassword(e.target.value)} // Update confirm password state
        />
        <PrimaryButton
          type="submit"
          label="Create Account"
          className="w-full"
        />
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
