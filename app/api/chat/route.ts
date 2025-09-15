import { type NextRequest, NextResponse } from "next/server"

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if (!OPENROUTER_API_KEY) {
  console.warn("OPENROUTER_API_KEY is not set")
}

export async function POST(request: NextRequest) {
  try {
    const { messages, model, stream = false, files = [] } = await request.json()

    if (!OPENROUTER_API_KEY) {
      return NextResponse.json({ error: "OpenRouter API key not configured" }, { status: 500 })
    }

    // Prepare the request body
    const requestBody: any = {
      model,
      messages,
      stream,
      temperature: 0.7,
      max_tokens: 2000,
    }

    if (files && files.length > 0) {
      // Transform files for OpenRouter API
      const transformedFiles = files.map((file: any) => {
        // Check if data is a URL (for PDF URLs) or base64 data
        const isUrl = typeof file.data === "string" && file.data.startsWith("http")

        return {
          filename: file.name,
          file_data: isUrl ? file.data : `data:${file.type};base64,${file.data}`,
        }
      })

      // Add files to the last user message instead of separate files parameter
      if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1]
        if (lastMessage.role === "user") {
          // Convert content to array format if it's a string
          if (typeof lastMessage.content === "string") {
            lastMessage.content = [
              {
                type: "text",
                text: lastMessage.content,
              },
            ]
          }

          // Add file entries to the content array
          transformedFiles.forEach((file: any) => {
            lastMessage.content.push({
              type: "file",
              file: file,
            })
          })
        }
      }

      requestBody.messages = messages
    }

    const hasPDFFiles = files.some(
      (file: any) => file.type === "application/pdf" || file.name?.toLowerCase().endsWith(".pdf"),
    )

    if (hasPDFFiles || files.length > 0) {
      requestBody.plugins = [
        {
          id: "file-parser",
          pdf: {
            engine: "pdf-text", // Free option for well-structured PDFs
            // engine: 'mistral-ocr' // $2 per 1,000 pages for scanned PDFs
          },
        },
      ]
    }

    console.log("Sending request to OpenRouter:", {
      model,
      messagesCount: messages.length,
      filesCount: files.length,
      hasPDFFiles,
      plugins: requestBody.plugins,
      lastMessageContent: messages[messages.length - 1]?.content,
    })

    const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000",
        "X-Title": "ORS Bot",
      },
      body: JSON.stringify(requestBody),
    })

    if (!response.ok) {
      const errorData = await response.text()
      console.error("OpenRouter API error:", errorData)
      return NextResponse.json({ error: "Failed to get response from AI model" }, { status: response.status })
    }

    if (stream) {
      // Return streaming response
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      })
    } else {
      // Return regular JSON response
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
