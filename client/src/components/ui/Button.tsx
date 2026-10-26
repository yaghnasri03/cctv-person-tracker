import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "primary" | "outline";
}

export function Button({
  children,
  variant = "primary",
  className = "",
  ...props
}: ButtonProps) {
  const base =
    "px-4 py-2 rounded-lg font-medium transition-all duration-200 cursor-pointer";

  const styles =
    variant === "primary"
      ? "bg-blue-600 hover:bg-blue-700 text-white"
      : "border border-blue-600 text-blue-600 hover:bg-blue-50";

  return (
    <button
      type="button"
      className={`${base} ${styles} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}