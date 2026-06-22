import type { Metadata } from "next";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";
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
    <html lang="en" className="bp6-dark">
      <body className="bp6-dark">{children}</body>
    </html>
  );
}
