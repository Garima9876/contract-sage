// src/components/Activate.jsx
import React, { useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import TextInput from "../shared/TextInput";
import PrimaryButton from "../shared/PrimaryButton";
import { activateUser } from "../api/api";
import { toast } from "react-toastify";

const Activate = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const navigate = useNavigate();

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleActivate = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }
    try {
      await activateUser(token, password);
      toast.success("Account activated! You can now log in.");
      navigate("/auth/login");
    } catch (err) {
      console.error("Activation error:", err);
      toast.error(err.response?.data?.detail || "Activation failed.");
    }
  };

  return (
    <div
      className="bg-white rounded-[20px] padding-responsive"
      style={{ boxShadow: "0px 9px 34px 0px #0000001A" }}
    >
      <div className="flex items-center mt-4">
        <span className="pr-2 mb-1 text-[var(--color-dark-grey)] font-heading large-text font-semibold">
          Activate Your Account
        </span>
      </div>
      <div className="text-grey small-text mb-8">
        Set a password to activate your account
      </div>
      <form onSubmit={handleActivate} className="flex flex-col gap-4">
        <TextInput
          id="password"
          type="password"
          label="Password"
          placeholder="Enter Password"
          className="mb-2"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <TextInput
          id="confirm-password"
          type="password"
          label="Confirm Password"
          placeholder="Confirm Password"
          className="mb-5"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
        <PrimaryButton type="submit" label="Activate Account" className="w-full" />
      </form>
        <p className="mt-5 text-center">
        Back to{" "}
        <button
          onClick={() => navigate("/auth/login")}
          className="text-sky underline"
        >
          Login
        </button>
      </p>
    </div>
  );
};

export default Activate;
