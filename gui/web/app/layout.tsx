import type { Metadata } from "next";
import { Urbanist, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// UI: Urbanist — a geometric sans whose round, even shapes stay legible at the
// small sizes a node card needs. Data + the readout: IBM Plex Mono, whose
// tabular figures read like a measurement display.
const display = Urbanist({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "iNOB — trials-to-detect journey",
  description:
    "Place a nerve source on the anatomy, run the forward model, and read how " +
    "many averaged trials it takes to detect it.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable}`}>
      {/* Grammarly and friends inject attributes into <body> before React
          hydrates, which React reports as a hydration mismatch. The markup we
          render is identical either way, so ignore attribute drift here. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
