import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PADOC Trace Viewer",
  description:
    "Focus + context exploration for traces that are too large to open all at once.",
  openGraph: {
    title: "PADOC Trace Viewer",
    description: "Focus one tree. Keep the whole trace in context.",
    images: [
      {
        url: "/og-padoc-trace-viewer.png",
        width: 1729,
        height: 910,
        alt: "PADOC focus and context trace visualization",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
