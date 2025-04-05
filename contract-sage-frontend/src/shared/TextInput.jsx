import React, { useState } from "react";
import { useTheme } from "@mui/material/styles";
import { TextField, InputAdornment } from "@mui/material";
import { EyeIcon, EyeSlashIcon } from "@heroicons/react/24/outline";

const TextInput = ({
  id,
  type,
  label,
  className,
  value,
  error,
  helperText,
  required,
  onChange,
  readOnly
}) => {
  const theme = useTheme();
  const [showPassword, setShowPassword] = useState(false); // state to toggle password visibility

  const togglePasswordVisibility = () => {
    setShowPassword((prevShowPassword) => !prevShowPassword);
  };

  return (
    <div className={`${className} relative`}>
      <label
        className="block text-[var(--color-dark-grey)] mb-1 small-text"
        htmlFor={id}
      >
        {label} {required && <span style={{ color: "#FF1F00" }}>*</span>}
      </label>
      <TextField
        type={showPassword ? "text" : type}
        fullWidth
        sx={{
          width: "100%",
          backgroundColor:
            theme.palette.mode === "dark" ? "transparent" : "white",
          borderRadius: "8px !important",
          "& .MuiOutlinedInput-root": {
            "& fieldset": {
              border: "1px solid #5F6177",
              borderRadius: "8px !important",
            },
            "&.Mui-focused fieldset": {
              border: "1px solid #5F6177",
            },
          },
        }}
        id={id}
        placeholder={`Enter ${label}`}
        value={value}
        onChange={onChange}
        size="small"
        error={error}
        helperText={helperText}
        required={required}
        InputProps={{
          readOnly: readOnly,
          endAdornment:
            type === "password" ? (
              <InputAdornment position="end" onClick={togglePasswordVisibility}>
                {showPassword ? (
                  <EyeIcon className="h-5 w-5 text-gray-500 cursor-pointer" />
                ) : (
                  <EyeSlashIcon className="h-5 w-5 text-gray-500 cursor-pointer" />
                )}
              </InputAdornment>
            ) : null,
        }}
      />
    </div>
  );
};

export default TextInput;
