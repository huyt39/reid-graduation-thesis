import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReID Dashboard",
  description: "Person re-identification live monitoring",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
