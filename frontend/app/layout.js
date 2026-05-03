import "./globals.css";

export const metadata = {
  title: "VetAI — Cattle health assistant",
  description: "Checklist-based cattle disease and pregnancy staging helper (API-backed).",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
