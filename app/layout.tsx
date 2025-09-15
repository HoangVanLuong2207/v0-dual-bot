import type React from "react"
import type { Metadata } from "next"
import { GeistMono } from "geist/font/mono"
import { Be_Vietnam_Pro } from "next/font/google"
import "./globals.css"

export const metadata: Metadata = {
  title: {
    default: "ORS Bot",
    template: "%s | ORS Bot",
  },
  description: "ORS Bot — Trợ lý AI",
  generator: "v0.dev",
}

const beVietnam = Be_Vietnam_Pro({
  subsets: ["vietnamese", "latin"],
  weight: ["400", "500", "600", "700"], // thêm các weight cần thiết
  variable: "--font-sans",
  display: "swap",
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="vi">
      <body className={`${beVietnam.variable} ${GeistMono.variable} antialiased`}>{children}</body>
    </html>
  )
}
