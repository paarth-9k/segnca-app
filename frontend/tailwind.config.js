/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0e13",
        panel: "#0f151c",
        panel2: "#141b24",
        border: "#1f2a37",
        text: "#e6edf3",
        muted: "#7d8a9a",
        nca: "#e0524f",
        "nca-dim": "#7a3230",
        unet: "#3f7fd6",
        "unet-dim": "#28405f",
        phosphor: "#5eead4",
      },
      fontFamily: {
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [],
};
