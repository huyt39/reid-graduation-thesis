import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: "#0f1117",
        panel: "#1a1d27",
        border: "#2a2d3a",
        accent: "#3b82f6",
        good: "#22c55e",
        mid: "#eab308",
        bad: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
