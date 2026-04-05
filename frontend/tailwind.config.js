/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
        mono: ["DM Mono", "ui-monospace", "monospace"],
      },
      colors: {
        bg:       "#0f0f0f",
        surface:  "#161618",
        surface2: "#1e1e21",
        card:     "#1e1e21",
        border:   "#2a2a2d",
        border2:  "#3f3f46",
        muted:    "#6b6b7a",
        gold:     "#d4a853",
        adaptive: "#d4a853",
        static:   "#6b6b7a",
        market:   "#60a5fa",
        success:  "#4ade80",
        warning:  "#fb923c",
        danger:   "#f87171",
      },
    },
  },
  plugins: [],
};
