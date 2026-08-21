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
    <html lang="en" className={`${display.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        {/* Resolve the theme before the first paint: ?theme=light|dark wins
            (handy for a screenshot or a print preview), then a stored choice,
            then the OS. Doing this in an effect would show one frame of the
            wrong palette on every load. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var q=new URLSearchParams(location.search).get("theme");var t=(q==="light"||q==="dark")?q:localStorage.getItem("inob-theme");if(t!=="light"&&t!=="dark"){t=window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";}document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme="dark";}})();`,
          }}
        />
      </head>
      {/* Grammarly and friends inject attributes into <body> before React
          hydrates, which React reports as a hydration mismatch. The markup we
          render is identical either way, so ignore attribute drift here. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
