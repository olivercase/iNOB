import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "iNOB — Vagus Nerve Forward Model",
  description:
    "Configure sources, array, conductivities and meshes, simulate, and estimate trials-to-detect.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
