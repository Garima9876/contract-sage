// src/components/Login.jsx
import React, { useState } from "react";
import TextInput from "../shared/TextInput";
import PrimaryButton from "../shared/PrimaryButton";
import { loginUser } from "../api/api";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const Login = ({ onLogin }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await loginUser({ email, password });
      toast.success("Login successful!");
      onLogin(response.data.access_token);
      navigate("/dashboard");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Error connecting to server.");
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
        Please login to your account
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <TextInput
          id="email"
          type="email"
          label="Email"
          placeholder="Enter Email"
          className="mb-5"
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
        <PrimaryButton type="submit" label="Login" className="w-full" />

        <div className="mt-10 mb-4 text-center medium-text">
          <span className="text-grey">Don't have an account?</span>{" "}
          <button
            type="button"
            className="font-medium"
            onClick={() => navigate("/auth/register")}
          >
            Create Account
          </button>
        </div>
      </form>
    </div>
  );
};

export default Login;
