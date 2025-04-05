/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      screens: {
        xxs: "340px",
        xs: "480px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
      },
      fontFamily: {
        body: ["Roboto", "sans-serif"],
        heading: ["Poppins", "sans-serif"],
      },
      fontSize: {
        s: "15px",
      },
    },
  },
  plugins: [],
}

