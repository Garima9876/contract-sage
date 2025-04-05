import React from "react";
import Button from '@mui/material/Button';

const PrimaryButton = ({
  type,
  label,
  className,
  onClick,
  disabled,
}) => {
  return (
    <Button
      type={type}
      onClick={onClick}
      variant="contained"
      disabled={disabled}
      className={`${className}`}
      sx={{
        borderRadius: "8px",
        textTransform: "none",
        fontWeight: "600",
        fontSize:{
          md: "14px",
          xxl: "16px",
          // xxl: "18px",
        },
        paddingY: "10px"
      }}
    >
      {label}
    </Button>
  );
};

export default PrimaryButton;
