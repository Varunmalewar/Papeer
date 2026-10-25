import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { NebulaBackground } from "@/components/nebula-background";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Papeer",
  description: "Research Paper Assistant",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} dark`} suppressHydrationWarning>
      {/* suppressHydrationWarning: browser extensions (e.g. Brave) inject
          attributes (bis_*) into body before React hydrates */}
      <body
        suppressHydrationWarning
        className="min-h-dvh font-sans text-slate-200 antialiased"
      >
        <NebulaBackground />
        {children}
      </body>
    </html>
  );
}
