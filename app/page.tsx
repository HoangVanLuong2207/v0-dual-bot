"use client"

import type React from "react"
import { Lightbulb } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from "react"
import { v4 as uuidv4 } from "uuid"
import {
  MessageSquare,
  Plus,
  Send,
  Trash2,
  Search,
  Bot,
  User,
  SplitSquareVertical,
  History,
  Download,
  Building2,
  Paperclip,
  X,
  Sparkles,
  BookMarked,
  FlaskConical,
  Folder,
  MoreHorizontal,
  Menu,
  FileText,
  ImageIcon,
  Table,
  AlertCircle,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronRight,
  Zap,
} from "lucide-react"
import { Document, Packer, Paragraph, TextRun, HeadingLevel } from "docx"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { cn } from "@/lib/utils"
import { isImageFile, SUPPORTED_FILE_TYPES, MAX_FILE_SIZE, type FileContent, processFile } from "@/lib/file-utils" // Declare the processFile variable

type Role = "user" | "assistant"
type Model = "chatgpt" | "gemini"
type Workflow = "single" | "chatgpt-to-gemini" | "perplexity-to-gemini" | "perplexity-chatgpt-gemini"

type ChatMessage = {
  id: string
  role: Role
  model?: Model
  content: string
  createdAt: number
}

type ChatTurn = {
  id: string
  user: ChatMessage
  chatgpt: ChatMessage
  gemini: ChatMessage
  workflow?: Workflow
  workflowSteps?: {
    step1: any
    step2: any
  }
  searchResults?: any
}

type ChatSession = {
  id: string
  title: string
  createdAt: number
  categoryId: string
  turns: ChatTurn[]
}

type Category = {
  id: string
  name: string
  icon: "book" | "flask" | "folder" | "more"
}

const SESSIONS_STORAGE = "dual-ai-chat-sessions-v2"
const CATEGORIES_STORAGE = "dual-ai-chat-categories-v1"

// Layout constants
const HEADER_H = 56
const INPUT_H = 120

// Model mappings for direct APIs
const MODEL_MAPPING = {
  gemini: "gemini-2.5-flash-lite",
}

// --- Utilities ---
function formatTimestamp(d = new Date()) {
  const pad = (n: number) => n.toString().padStart(2, "0")
  const yyyy = d.getFullYear()
  const mm = pad(d.getMonth() + 1)
  const dd = pad(d.getDate())
  const hh = pad(d.getHours())
  const min = pad(d.getMinutes())
  const ss = pad(d.getSeconds())
  return `${yyyy}${mm}${dd}-${hh}${min}${ss}`
}

function slugify(s: string) {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)+/g, "")
}

function makeMarkdown(modelLabel: string, text: string) {
  const when = new Date().toLocaleString("vi-VN")
  return `# ${modelLabel}

> Tạo lúc: ${when}

${text}
`
}

function downloadTextFile(filename: string, text: string, mime: string) {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function formatBytes(bytes: number) {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}

// --- API call function ---
async function callDirectAPI(
  model: Model,
  messages: any[],
  stream = false,
  files: FileContent[] = [],
  workflow: Workflow = "single",
  deepSearch = false,
) {
  console.log("[v0] callDirectAPI called with:", {
    model,
    messagesCount: messages.length,
    filesCount: files.length,
    workflow,
  })

  // For ChatGPT to Gemini workflow, send all files as native files to API
  // and don't add file content to text message (ChatGPT should only see text question)
  let nativeFiles: any[] = []
  let processedContent: FileContent[] = []

  if (workflow === "chatgpt-to-gemini") {
    // For ChatGPT to Gemini workflow: send all files as native files
    nativeFiles = files
      .map(
        (fc) =>
          fc.fileData || {
            name: fc.fileName,
            type: fc.type || "unknown",
            data: fc.content,
          },
      )
      .filter(Boolean)
    processedContent = [] // Don't process files locally for this workflow
  } else {
    // For single workflow: use original logic
    const locallyProcessedFiles = files.filter((fc) => fc.type === "text" || fc.type === "table")
    nativeFiles = files
      .filter((fc) => fc.type === "pdf-native" || fc.type === "office-native")
      .map((fc) => fc.fileData)
      .filter(Boolean)
    processedContent = files.filter((fc) => fc.type !== "pdf-native" && fc.type !== "office-native")
  }

  console.log("[v0] File processing breakdown:", {
    workflow,
    nativeFilesCount: nativeFiles.length,
    processedContentCount: processedContent.length,
  })

  // Build user content with text and images
  const userContent: any[] = []
  const textContent = messages[messages.length - 1]?.content || ""

  if (workflow === "single") {
    // Add processed file content to text
    processedContent.forEach((fileContent) => {
      if (fileContent.type === "text" || fileContent.type === "table") {
        // textContent += `\n\n--- File Content ---\n${fileContent.content}`;
      } else if (fileContent.type === "image") {
        userContent.push({
          type: "image_url",
          image_url: {
            url: fileContent.content,
          },
        })
      }
    })
  } else {
    // For ChatGPT to Gemini workflow, only add images to userContent
    processedContent.forEach((fileContent) => {
      if (fileContent.type === "image") {
        userContent.push({
          type: "image_url",
          image_url: {
            url: fileContent.content,
          },
        })
      }
    })
  }

  if (textContent.trim()) {
    userContent.push({
      type: "text",
      text: textContent,
    })
  }

  const requestBody = {
    messages: [
      ...messages.slice(0, -1),
      {
        role: "user",
        content: userContent.length === 1 && userContent[0].type === "text" ? userContent[0].text : userContent,
      },
    ],
    model: MODEL_MAPPING[model],
    stream,
    files: nativeFiles,
    workflow,
    deepSearch,
  }

  console.log("[v0] Sending request to API:", {
    endpoint: "/api/direct-chat",
    model: MODEL_MAPPING[model],
    workflow,
    hasFiles: nativeFiles.length > 0,
    textOnlyForChatGPT: workflow === "chatgpt-to-gemini",
  })

  try {
    const response = await fetch("/api/direct-chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    })

    if (!response.ok) {
      const errorData = await response.json()
      console.error("[v0] API error response:", errorData)
      throw new Error(errorData.error || `HTTP ${response.status}`)
    }

    const data = await response.json()
    console.log("[v0] API response received:", {
      workflow: data.workflow,
      hasStep1: !!data.step1,
      hasStep2: !!data.step2,
      responsePreview: data.choices?.[0]?.message?.content?.substring(0, 100) + "...",
    })

    return data
  } catch (error) {
    console.error("[v0] API call failed:", error)
    throw error
  }
}

// --- Categories (folders) state ---
function useCategories() {
  const [categories, setCategories] = useState<Category[]>([])
  const [activeCategoryId, setActiveCategoryId] = useState<string>("")

  useEffect(() => {
    try {
      const raw = localStorage.getItem(CATEGORIES_STORAGE)
      if (raw) {
        const parsed = JSON.parse(raw) as {
          categories: Category[]
          activeId: string
        }
        setCategories(parsed.categories)
        setActiveCategoryId(parsed.activeId || parsed.categories[0]?.id || "")
      } else {
        const defaults: Category[] = [
          { id: "quan-ly-du-an", name: "Quản lý dự án", icon: "book" },
          {
            id: "nghien-cuu-giai-phap",
            name: "Nghiên cứu giải pháp",
            icon: "flask",
          },
        ]
        setCategories(defaults)
        setActiveCategoryId(defaults[0].id)
        localStorage.setItem(CATEGORIES_STORAGE, JSON.stringify({ categories: defaults, activeId: defaults[0].id }))
      }
    } catch {
      const defaults: Category[] = [
        { id: "quan-ly-du-an", name: "Quản lý dự án", icon: "book" },
        {
          id: "nghien-cuu-giai-phap",
          name: "Nghiên cứu giải pháp",
          icon: "flask",
        },
      ]
      setCategories(defaults)
      setActiveCategoryId(defaults[0].id)
    }
  }, [])

  useEffect(() => {
    if (!categories.length) return
    try {
      localStorage.setItem(CATEGORIES_STORAGE, JSON.stringify({ categories, activeId: activeCategoryId }))
    } catch {}
  }, [categories, activeCategoryId])

  function createCategory(name: string) {
    const base = slugify(name) || "chu-de-moi"
    let id = base
    let i = 1
    const existing = new Set(categories.map((c) => c.id))
    while (existing.has(id)) {
      id = `${base}-${i++}`
    }
    const cat: Category = { id, name, icon: "folder" }
    setCategories((prev) => [cat, ...prev])
    setActiveCategoryId(cat.id)
  }

  return {
    categories,
    activeCategoryId,
    setActiveCategoryId,
    createCategory,
  }
}

// --- Sessions state ---
function useChatSessions(defaultCategoryId: string) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)

  // Load
  useEffect(() => {
    try {
      const raw = localStorage.getItem(SESSIONS_STORAGE)
      if (raw) {
        const parsed = JSON.parse(raw) as ChatSession[]
        setSessions(parsed)
        setActiveId(parsed[0]?.id ?? null)
      } else {
        const initial: ChatSession = {
          id: uuidv4(),
          title: "Cuộc trò chuyện mới",
          createdAt: Date.now(),
          categoryId: defaultCategoryId,
          turns: [],
        }
        setSessions([initial])
        setActiveId(initial.id)
      }
    } catch {
      const initial: ChatSession = {
        id: uuidv4(),
        title: "Cuộc trò chuyện mới",
        createdAt: Date.now(),
        categoryId: defaultCategoryId,
        turns: [],
      }
      setSessions([initial])
      setActiveId(initial.id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist
  useEffect(() => {
    try {
      localStorage.setItem(SESSIONS_STORAGE, JSON.stringify(sessions))
    } catch {}
  }, [sessions])

  useEffect(() => {
    if (!activeId && sessions.length > 0) {
      setActiveId(sessions[0].id)
    }
  }, [activeId, sessions])

  const active = useMemo(() => sessions.find((s) => s.id === activeId) ?? null, [sessions, activeId])

  function newSession(categoryId: string) {
    const s: ChatSession = {
      id: uuidv4(),
      title: "Cuộc trò chuyện mới",
      createdAt: Date.now(),
      categoryId,
      turns: [],
    }
    setSessions((prev) => [s, ...prev])
    setActiveId(s.id)
  }

  function deleteSession(id: string) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id)
      if (next.length === 0) {
        const s: ChatSession = {
          id: uuidv4(),
          title: "Cuộc trò chuyện mới",
          createdAt: Date.now(),
          categoryId: defaultCategoryId,
          turns: [],
        }
        // đặt active ngay khi tạo mới
        setActiveId(s.id)
        return [s]
      }
      // nếu đang active và bị xoá, chuyển sang phần tử đầu tiên còn lại
      setActiveId((curr) => (curr === id ? next[0].id : curr))
      return next
    })
  }

  function renameSession(id: string, title: string) {
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title } : s)))
  }

  function setActive(id: string) {
    setActiveId(id)
  }

  function addTurn(sessionId: string, turn: ChatTurn) {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s
        const isFirst = s.turns.length === 0
        const newTitle = isFirst ? (turn.user.content || "Cuộc trò chuyện").slice(0, 40) : s.title
        return { ...s, title: newTitle, turns: [...s.turns, turn] }
      }),
    )
  }

  function clearActiveSession(sessionId: string) {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, turns: [] } : s)))
  }

  return {
    sessions,
    active,
    activeId,
    setActive,
    newSession,
    deleteSession,
    renameSession,
    addTurn,
    clearActiveSession,
  }
}

// --- UI Helpers ---
function MessageBubble({ role, content }: { role: Role; content: string }) {
  const isUser = role === "user"
  return (
    <div
      className={cn(
        "flex items-start gap-3 px-1 sm:px-0", // ensure same left/right padding as cards on mobile
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {!isUser && (
        <div className="mt-1 rounded-full bg-emerald-100 p-2 text-emerald-700">
          <Bot className="h-4 w-4" />
          <span className="sr-only">{"Assistant"}</span>
        </div>
      )}
      <div
        className={cn(
          // harmonize width so it aligns visually with response cards on mobile
          "max-w-[88%] sm:max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
          isUser ? "bg-emerald-600 text-white" : "bg-muted text-foreground",
        )}
      >
        <div className="whitespace-pre-wrap">{content}</div>
      </div>
      {isUser && (
        <div className="mt-1 rounded-full bg-emerald-100 p-2 text-emerald-700">
          <User className="h-4 w-4" />
          <span className="sr-only">{"User"}</span>
        </div>
      )}
    </div>
  )
}

// Convert Gemini response to HTML
const convertToHtml = (text: string) => {
  const escapeHtml = (value: string) =>
    value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')

  const urlRegex = /(https?:\/\/[^\s]+)/g

  const escaped = escapeHtml(text)

  return escaped.replace(
    urlRegex,
    (url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`,
  )
}

function DualAnswer({
  chatgpt,
  gemini,
  workflowSteps,
  searchResults,
}: {
  chatgpt: string
  gemini: string
  workflowSteps?: any
  searchResults?: any
}) {
  const [showWorkflow, setShowWorkflow] = useState(false)

  function handleDownload(modelLabel: string, body: string, ext: "md" | "txt") {
    const ts = formatTimestamp()
    const name = `${slugify(modelLabel)}-${ts}.${ext}`
    const content = ext === "md" ? makeMarkdown(modelLabel, body) : body
    const mime = ext === "md" ? "text/markdown;charset=utf-8" : "text/plain;charset=utf-8"
    downloadTextFile(name, content, mime)
  }

  function extractTextLines(body: string): string[] {
    // 1. Nếu có HTML, parse và lấy text
    const parser = new DOMParser()
    const doc = parser.parseFromString(body, "text/html")
    let text = doc.body.textContent || ""

    // 2. Tách theo \n hoặc ngắt dòng
    // - Replace nhiều khoảng trắng thừa
    text = text.replace(/\r/g, "") // loại bỏ carriage return
    const lines = text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)

    return lines
  }

  async function handleDownloadDocx(modelLabel: string, body: string) {
    const ts = formatTimestamp()
    const name = `${slugify(modelLabel)}-${ts}.docx`

    const lines = extractTextLines(body) // lấy text

    const doc = new Document({
      sections: [
        {
          children: [
            new Paragraph({
              text: modelLabel,
              heading: HeadingLevel.HEADING_1,
            }),
            ...lines.map((line) => new Paragraph({ children: [new TextRun(line)] })),
          ],
        },
      ],
    })

    const blob = await Packer.toBlob(doc)
    downloadBlob(name, blob)
  }

  // async function handleDownloadDocx(modelLabel: string, body: string) {
  //   const ts = formatTimestamp();
  //   const name = `${slugify(modelLabel)}-${ts}.docx`;

  //   const doc = new Document({
  //     sections: [
  //       {
  //         children: [
  //           new Paragraph({
  //             text: modelLabel,
  //             heading: HeadingLevel.HEADING_1,
  //           }),
  //           ...body.split("\n").map(
  //             (line) =>
  //               new Paragraph({
  //                 children: [new TextRun(line)],
  //               })
  //           ),
  //         ],
  //       },
  //     ],
  //   });

  //   const blob = await Packer.toBlob(doc);
  //   downloadBlob(name, blob);
  // }

  return (
    <div className="grid grid-cols-1 gap-3 sm:gap-4">
      {workflowSteps && workflowSteps.length > 0 && (
        <div className="mx-1 sm:mx-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowWorkflow(!showWorkflow)}
            className="mb-2 h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
          >
            {showWorkflow ? <EyeOff className="mr-1 h-3 w-3" /> : <Eye className="mr-1 h-3 w-3" />}
            {showWorkflow ? "Ẩn luồng xử lý" : "Xem luồng xử lý"}
            {showWorkflow ? <ChevronDown className="ml-1 h-3 w-3" /> : <ChevronRight className="ml-1 h-3 w-3" />}
          </Button>

          {showWorkflow && (
            <div className="mb-4 rounded-lg border border-dashed border-gray-300 bg-gray-50/50 p-4">
              <div className="space-y-4">
                {/* Step 1: Original Question */}
                <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-xs font-medium text-blue-700">
                    1
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-gray-900">Câu hỏi gốc</h4>
                    <p className="mt-1 text-sm text-gray-500">{workflowSteps[0]?.input?.user ?? ""}</p>
                  </div>
                </div>

                {/* Arrow */}
                <div className="ml-3 flex items-center">
                  <ChevronDown className="h-4 w-4 text-gray-400" />
                </div>

                {/* Step 2: ChatGPT generates prompt */}
                <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-orange-100 text-xs font-medium text-orange-700">
                    2
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-gray-900">ChatGPT sinh prompt</h4>
                    <p className="mt-1 text-sm text-gray-500">{workflowSteps[0]?.output ?? ""}</p>
                  </div>
                </div>

                {/* Arrow */}
                {/* <div className="ml-3 flex items-center">
                  <ChevronDown className="h-4 w-4 text-gray-400" />
                </div> */}

                {/* Step 3: Gemini result */}
                {/* <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 text-xs font-medium text-emerald-700">
                    3
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-gray-900">
                      Kết quả từ Gemini
                    </h4>
                    <p className="mt-1 text-sm text-gray-500">
                      Hiển thị phản hồi chi tiết từ Gemini dựa trên prompt đã
                      được tối ưu hóa...
                    </p>
                  </div>
                </div> */}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Gemini - Only showing this card now */}
      <Card className="border-emerald-200 mx-1 sm:mx-0">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <SplitSquareVertical className="h-4 w-4 text-emerald-600" />
            Chatbot
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="prose prose-sm max-w-none whitespace-pre-wrap leading-relaxed">
            <div dangerouslySetInnerHTML={{ __html: convertToHtml(gemini) }} />
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={() => handleDownloadDocx("Chatbot", gemini)}>
              <Download className="mr-2 h-4 w-4" />
              {"Xuất dữ liệu"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

type Attached = {
  id: string
  file: File
  content?: FileContent
  isProcessing?: boolean
}

function FileChip({
  attachment,
  onRemove,
}: {
  attachment: Attached
  onRemove: () => void
}) {
  const getFileTypeIcon = (content?: FileContent) => {
    if (!content) return <Paperclip className="h-3.5 w-3.5" />

    switch (content.type) {
      case "image":
        return <ImageIcon className="h-3.5 w-3.5" />
      case "table":
        return <Table className="h-3.5 w-3.5" />
      case "text":
        return <FileText className="h-3.5 w-3.5" />
      case "error":
        return <AlertCircle className="h-3.5 w-3.5 text-red-500" />
      default:
        return <Paperclip className="h-3.5 w-3.5" />
    }
  }

  const getStatusColor = () => {
    if (attachment.isProcessing) return "bg-blue-50 border-blue-200"
    if (attachment.content?.type === "error") return "bg-red-50 border-red-200"
    return "bg-green-50 border-green-200"
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs",
        getStatusColor(),
        attachment.isProcessing && "animate-pulse",
      )}
      title={`${attachment.file.name} • ${formatBytes(attachment.file.size)}${
        attachment.content?.metadata ? ` • ${JSON.stringify(attachment.content.metadata)}` : ""
      }`}
    >
      {attachment.isProcessing ? (
        <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-emerald-600 border-t-transparent" />
      ) : isImageFile(attachment.file) && attachment.content?.type === "image" ? (
        <img
          src={URL.createObjectURL(attachment.file) || "/placeholder.svg"}
          alt={attachment.file.name}
          className="h-6 w-6 rounded object-cover"
        />
      ) : (
        getFileTypeIcon(attachment.content)
      )}
      <span className="max-w-[120px] truncate sm:max-w-[160px]">{attachment.file.name}</span>
      {attachment.content?.type === "error" && <span className="text-red-500 text-xs">⚠️</span>}
      {attachment.content && attachment.content.type !== "error" && !attachment.isProcessing && (
        <span className="text-green-500 text-xs">✓</span>
      )}
      <button onClick={onRemove} className="ml-1 rounded p-0.5 hover:bg-white" aria-label="Xoá tệp">
        <X className="h-3.5 w-3.5" />
      </button>
    </span>
  )
}

export default function Page() {
  // Categories
  const { categories, activeCategoryId, setActiveCategoryId, createCategory } = useCategories()

  // Sessions
  const {
    sessions,
    active,
    activeId,
    setActive,
    newSession,
    deleteSession,
    renameSession,
    addTurn,
    clearActiveSession,
  } = useChatSessions(activeCategoryId || "quan-ly-du-an")

  // Input state
  const [deepSearch, setDeepSearch] = useState(false)
  const [input, setInput] = useState("")
  const [inputEnter, setInputEnter] = useState("")
  const [filter, setFilter] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [attachments, setAttachments] = useState<Attached[]>([])
  const [messages, setMessages] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState<Model>("gemini")
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow>("perplexity-to-gemini")
  const [fileContents, setFileContents] = useState<FileContent[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const taRef = useRef<HTMLTextAreaElement | null>(null)

  // Mobile sidebar open
  const [mobileOpen, setMobileOpen] = useState(false)

  // auto-resize textarea
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = "0px"
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px"
  }, [input])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [active?.turns.length, isLoading])

  const filteredSessions = useMemo(() => {
    const byCategory = sessions.filter((s) => s.categoryId === activeCategoryId)
    if (!filter.trim()) return byCategory
    const f = filter.toLowerCase()
    return byCategory.filter((s) => s.title.toLowerCase().includes(f))
  }, [sessions, filter, activeCategoryId])

  async function onFilesSelected(files: FileList | null) {
    if (!files) return

    const validFiles: Attached[] = []
    const errors: string[] = []

    Array.from(files).forEach((file) => {
      console.log("File selected:", {
        name: file.name,
        type: file.type,
        size: file.size,
      })

      if (file.size > MAX_FILE_SIZE) {
        errors.push(`${file.name}: Kích thước quá lớn (tối đa 25MB)`)
        return
      }

      // More flexible file type checking
      const isSupported =
        SUPPORTED_FILE_TYPES.includes(file.type) || file.name.match(/\.(pdf|docx?|xlsx?|xlsm|pptx?|csv|txt|md|json)$/i)

      if (!isSupported) {
        errors.push(`${file.name}: Định dạng không hỗ trợ`)
        return
      }

      validFiles.push({ id: uuidv4(), file, isProcessing: true })
    })

    if (errors.length > 0) {
      alert(`Một số file không thể tải lên:\n${errors.join("\n")}`)
    }

    if (validFiles.length > 0) {
      setAttachments((prev) => [...prev, ...validFiles])

      // Process files
      for (const attachment of validFiles) {
        try {
          console.log("Starting to process file:", attachment.file.name)
          const content = await processFile(attachment.file)
          console.log("File processed successfully:", {
            fileName: attachment.file.name,
            contentType: content.type,
            contentLength: content.content?.length || 0,
          })

          setAttachments((prev) =>
            prev.map((a) => (a.id === attachment.id ? { ...a, content, isProcessing: false } : a)),
          )
          setFileContents((prev) => [...prev, content])
        } catch (error) {
          console.error("File processing failed:", error)
          setAttachments((prev) =>
            prev.map((a) =>
              a.id === attachment.id
                ? {
                    ...a,
                    content: {
                      type: "error",
                      content: `Lỗi xử lý file: ${error instanceof Error ? error.message : "Unknown error"}`,
                    },
                    isProcessing: false,
                  }
                : a,
            ),
          )
        }
      }
    }
  }

  async function handleSend() {
    if (!input.trim() && attachments.length === 0) return

    const userMessage = { role: "user" as const, content: input }

    try {
      setInput("")
      setIsLoading(true)

      const conversationHistory: any[] = []

      // Add previous turns from the active session
      if (active?.turns) {
        active.turns.forEach((turn) => {
          // Add user message
          conversationHistory.push({
            role: "user",
            content: turn.user.content,
          })

          // Add assistant response based on selected model
          const assistantContent = selectedModel === "gemini" ? turn.gemini.content : turn.chatgpt.content

          if (assistantContent.trim()) {
            conversationHistory.push({
              role: "assistant",
              content: assistantContent,
            })
          }
        })
      }

      // Add current user message
      const currentMessages = [...conversationHistory, userMessage]
      setMessages([userMessage]) // Only show current message in UI

      console.log("[v0] handleSend - Processing with conversation history:", {
        attachmentsCount: attachments.length,
        fileContentsCount: fileContents.length,
        selectedWorkflow,
        deepSearch,
        conversationHistoryLength: conversationHistory.length,
        totalMessagesCount: currentMessages.length,
        workflowDescription:
          selectedWorkflow === "chatgpt-to-gemini"
            ? "ChatGPT will generate prompt for Gemini"
            : selectedWorkflow === "perplexity-to-gemini"
              ? "Perplexity search then Gemini response"
              : selectedWorkflow === "perplexity-chatgpt-gemini"
                ? "Perplexity search, then ChatGPT refines, then Gemini responds"
                : "Direct Gemini processing",
      })

      const response = await callDirectAPI(
        selectedModel,
        currentMessages, // Now includes full conversation history
        false,
        fileContents,
        selectedWorkflow,
        deepSearch,
      )

      const assistantMessage = { role: "assistant" as const, content: "" }

      // Handle different response formats
      if (selectedModel === "gemini") {
        assistantMessage.content =
          response.candidates?.[0]?.content?.parts?.[0]?.text ||
          response.choices?.[0]?.message?.content ||
          "Không có phản hồi"
      } else {
        assistantMessage.content = response.choices?.[0]?.message?.content || "Không có phản hồi"
      }

      console.log("[v0] handleSend - Response processed:", {
        workflow: response.workflow,
        responseLength: assistantMessage.content.length,
        hasWorkflowSteps: !!(response.step1 && response.step2),
        hasSearchResults: !!response.searchResults,
        searchResultsCount: response.searchResults?.results?.length || 0,
        step1Preview: response.step1?.prompt?.substring(0, 100) + "..." || "No step 1",
        step2Preview: response.step2?.content?.substring(0, 100) + "..." || "No step 2",
      })

      // Store the response based on selected model
      const turnData = {
        id: uuidv4(),
        user: userMessage,
        chatgpt: selectedModel === "chatgpt" ? assistantMessage : { role: "assistant" as const, content: "" },
        gemini: selectedModel === "gemini" ? assistantMessage : { role: "assistant" as const, content: "" },
        workflow: response.workflow,
        workflowSteps: response.prompts ?? undefined,
        searchResults: response.searchResults,
        // response.step1 && response.step2
        //   ? {
        //       step1: response.step1,
        //       step2: response.step2,
        //     }
        //   : undefined,
      }

      addTurn(activeId || "", turnData)

      // Clear input and attachments
      // setInput("")
      setAttachments([])
      setFileContents([])
      setMessages([])
    } catch (error) {
      console.error("[v0] handleSend error:", error)

      const errorMessage = error instanceof Error ? error.message : "Đã xảy ra lỗi không xác định"

      // Show user-friendly error messages
      let displayError = errorMessage
      if (errorMessage.includes("quota")) {
        displayError = "Đã vượt quá giới hạn API. Vui lòng thử lại sau hoặc chuyển sang model khác."
      } else if (errorMessage.includes("key")) {
        displayError = "Lỗi xác thực API. Vui lòng kiểm tra cấu hình."
      } else if (errorMessage.includes("network")) {
        displayError = "Lỗi kết nối mạng. Vui lòng thử lại."
      }

      const errorTurn = {
        id: uuidv4(),
        user: { role: "user" as const, content: input },
        chatgpt: { role: "assistant" as const, content: "" },
        gemini: {
          role: "assistant" as const,
          content: `❌ Lỗi: ${displayError}`,
        },
      }

      addTurn(activeId || "", errorTurn)
      setInput("")
      setAttachments([])
      setFileContents([])
      setMessages([])
    } finally {
      setIsLoading(false)
    }
  }

  const IconFor = (i: Category["icon"]) =>
    i === "book" ? BookMarked : i === "flask" ? FlaskConical : i === "folder" ? Folder : MoreHorizontal

  // Sidebar content reused (desktop + mobile)
  function SidebarBody({ onNavigate }: { onNavigate?: () => void }) {
    return (
      <>
        <div className="mt-2 flex gap-2 px-2">
          <Button
            onClick={() => {
              newSession(activeCategoryId)
              onNavigate?.()
            }}
            className="w-full"
            size="sm"
            variant="default"
          >
            <Plus className="mr-2 h-4 w-4" />
            Đoạn chat mới
          </Button>
        </div>

        <div className="mt-3 px-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Tìm kiếm đoạn chat…"
              className="pl-8"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <div className="mt-2 flex">
            <Button
              variant="secondary"
              size="sm"
              className="w-full justify-start"
              onClick={() => {
                const name = prompt("Tên chủ đề mới:")
                if (name && name.trim()) {
                  createCategory(name.trim())
                }
              }}
            >
              <Plus className="mr-2 h-4 w-4" />
              Tạo chủ đề
            </Button>
          </div>
        </div>

        {/* Folders */}
        <div className="mt-3 space-y-1">
          {categories.map((c) => {
            const Icon = IconFor(c.icon)
            const isActive = c.id === activeCategoryId
            return (
              <button
                key={c.id}
                onClick={() => {
                  setActiveCategoryId(c.id)
                  onNavigate?.()
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-emerald-50",
                  isActive && "bg-emerald-100/70 hover:bg-emerald-100",
                )}
              >
                <Icon className="h-4 w-4 text-emerald-600" />
                <span className="truncate">{c.name}</span>
              </button>
            )
          })}
          {categories.length === 0 && (
            <div className="px-3 py-6 text-sm text-muted-foreground">Chưa có chủ đề. Hãy tạo mới.</div>
          )}
        </div>

        {/* History title */}
        <div className="mt-4 flex items-center gap-2 px-2 text-xs text-muted-foreground">
          <History className="h-3.5 w-3.5" />
          Đoạn chat
        </div>

        {/* Chats list by category */}
        <ScrollArea className="mt-2 h-full">
          <div className="space-y-1 pr-2">
            {filteredSessions.map((s) => {
              const isActive = s.id === activeId
              const handleOpen = () => {
                setActive(s.id)
                onNavigate?.()
              }
              const handleKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  handleOpen()
                }
              }

              return (
                <div
                  key={s.id}
                  role="button"
                  tabIndex={0}
                  onClick={handleOpen}
                  onKeyDown={handleKey}
                  className={cn(
                    "group flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm hover:bg-emerald-50 outline-none",
                    isActive && "bg-emerald-100/70 hover:bg-emerald-100",
                  )}
                >
                  <span className="line-clamp-1">{s.title || "Cuộc trò chuyện"}</span>

                  <span className="ml-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <button
                      type="button"
                      aria-label="Đổi tên"
                      className="rounded px-1 text-xs text-muted-foreground hover:text-foreground"
                      onClick={(e) => {
                        e.stopPropagation()
                        const name = prompt("Đặt tên cuộc trò chuyện:", s.title)
                        if (name !== null) {
                          const title = name.trim() || "Cuộc trò chuyện"
                          renameSession(s.id, title)
                        }
                      }}
                    >
                      Aa
                    </button>

                    <button
                      type="button"
                      aria-label="Xoá"
                      className="rounded p-1 text-muted-foreground hover:bg-rose-50 hover:text-rose-600"
                      onClick={(e) => {
                        e.stopPropagation()
                        const ok = confirm("Xoá đoạn chat này?")
                        if (ok) deleteSession(s.id)
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </span>
                </div>
              )
            })}

            {filteredSessions.length === 0 && (
              <div className="px-3 py-8 text-center text-sm text-muted-foreground">
                Không có đoạn chat trong chủ đề này
              </div>
            )}
          </div>
        </ScrollArea>
      </>
    )
  }

  return (
    <div className="h-screen w-full bg-white text-foreground">
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 shrink-0 border-r bg-muted/30 p-3 md:flex md:flex-col">
        <div className="flex items-center gap-2 px-2 py-1.5">
          <MessageSquare className="h-5 w-5 text-emerald-600" />
          <span className="font-semibold">ORS Chat</span>
        </div>
        <SidebarBody />
      </aside>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-20 w-full h-14 border-b bg-white/90 backdrop-blur md:pl-[288px]">
        <div className="flex h-full w-full items-center justify-between gap-3 px-2 sm:px-3 md:px-6 mx-auto max-w-4xl">
          <div className="flex items-center gap-2 md:gap-3">
            {/* Mobile: open sidebar */}
            <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="md:hidden">
                  <Menu className="h-5 w-5" />
                  <span className="sr-only">Mở menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[86vw] p-0">
                <SheetHeader className="px-3 pb-2 pt-3">
                  <SheetTitle className="flex items-center gap-2">
                    <MessageSquare className="h-5 w-5 text-emerald-600" />
                    ORS Chat
                  </SheetTitle>
                </SheetHeader>
                <div className="h-[calc(100vh-52px)]">
                  <SidebarBody onNavigate={() => setMobileOpen(false)} />
                </div>
              </SheetContent>
            </Sheet>

            <div className="hidden items-center gap-2 rounded-md border bg-white px-2 py-1.5 md:flex">
              <Building2 className="h-4 w-4 text-emerald-600" />
              <span className="text-xs font-semibold">ORS Corp</span>
            </div>
            <h1 className="line-clamp-1 text-sm font-semibold leading-tight">{active?.title || "Cuộc trò chuyện"}</h1>
          </div>

          {/* Only delete button remains */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              className="md:hidden bg-transparent"
              onClick={() => active && clearActiveSession(active.id)}
              disabled={!active || (active?.turns.length ?? 0) === 0}
              aria-label="Xoá hội thoại"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => active && clearActiveSession(active.id)}
              disabled={!active || (active?.turns.length ?? 0) === 0}
              className="hidden md:inline-flex"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Xoá hội thoại
            </Button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="relative w-full md:pl-72" style={{ paddingTop: `${HEADER_H}px`, paddingBottom: `${INPUT_H}px` }}>
        {/* style={{ height: `calc(100vh - ${HEADER_H}px - ${INPUT_H}px)` }} */}
        <div className="px-2 sm:px-3 md:px-6">
          <ScrollArea className="h-full">
            <div className="mx-auto w-full max-w-4xl space-y-4 sm:space-y-6 py-3 sm:py-4">
              {!active || active.turns.length === 0 ? (
                <div className="mt-8 sm:mt-16 rounded-lg border bg-muted/30 p-6 sm:p-8 text-center">
                  <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 sm:h-12 sm:w-12">
                    <SplitSquareVertical className="h-5 w-5 sm:h-6 sm:w-6" />
                  </div>
                  <h2 className="mt-3 text-base font-semibold sm:mt-4 sm:text-lg">Bắt đầu cuộc trò chuyện</h2>
                  <p className="mt-1 text-xs text-muted-foreground sm:text-sm">Gõ câu hỏi bên dưới để nhận kết quả.</p>
                  <div className="mt-4 text-xs text-muted-foreground">
                    <p>Hỗ trợ: ảnh và tệp</p>
                  </div>
                </div>
              ) : (
                active.turns.map((t) => (
                  <div key={t.id} className="space-y-3">
                    <MessageBubble role="user" content={t.user.content} />
                    <DualAnswer
                      chatgpt={t.chatgpt.content}
                      gemini={t.gemini.content}
                      workflowSteps={t.workflowSteps}
                      searchResults={t.searchResults}
                    />
                  </div>
                ))
              )}

              {isLoading && (
                <div className="space-y-3 opacity-80">
                  <MessageBubble role="user" content={input || inputEnter || "..."} />
                  <div className="grid grid-cols-1 gap-3 sm:gap-4">
                    <Card className="animate-pulse mx-1 sm:mx-0">
                      <CardHeader className="pb-2">
                        <CardTitle className="flex items-center gap-2 text-base">
                          <SplitSquareVertical className="h-4 w-4 text-emerald-600" />
                          Loading...
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="h-3 w-3/4 rounded bg-muted" />
                          <div className="h-3 w-2/3 rounded bg-muted" />
                          <div className="h-3 w-1/2 rounded bg-muted" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          </ScrollArea>
        </div>
      </main>

      {/* Input bar */}
      <div className="fixed bottom-0 left-0 right-0 z-20 w-full bg-white md:pl-[288px]">
        <div className="h-[120px] border-t">
          <div className="mx-auto flex h-full w-full max-w-4xl items-center px-2 sm:px-3 md:px-6">
            <div className="w-full rounded-2xl border bg-white shadow-sm">
              <div className="flex items-center gap-2 p-2">
                {/* Plus menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      type="button"
                      variant="secondary"
                      size="icon"
                      className="grid h-10 w-10 place-items-center rounded-full bg-muted p-0 text-foreground hover:bg-muted/80"
                    >
                      <Plus className="h-4 w-4" />
                      <span className="sr-only">{"Mở menu hành động"}</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" side="top" className="w-56">
                    <DropdownMenuLabel>Tuỳ chọn</DropdownMenuLabel>
                    <DropdownMenuCheckboxItem checked={deepSearch} onCheckedChange={(v) => setDeepSearch(!!v)}>
                      <Sparkles className="mr-2 h-4 w-4 text-emerald-600" />
                      Nghiên cứu sâu
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel>Quy trình xử lý</DropdownMenuLabel>
                    <DropdownMenuCheckboxItem
                      checked={selectedWorkflow === "single"}
                      onCheckedChange={() => setSelectedWorkflow("single")}
                    >
                      <Bot className="mr-2 h-4 w-4" />
                      Trực tiếp (Gemini)
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem
                      checked={selectedWorkflow === "chatgpt-to-gemini"}
                      onCheckedChange={() => setSelectedWorkflow("chatgpt-to-gemini")}
                    >
                      <FlaskConical className="mr-2 h-4 w-4 text-blue-600" />
                      ChatGPT → Gemini
                      {selectedWorkflow === "chatgpt-to-gemini" && (
                        <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">Đang hoạt động</span>
                      )}
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem
                      checked={selectedWorkflow === "perplexity-to-gemini"}
                      onCheckedChange={() => setSelectedWorkflow("perplexity-to-gemini")}
                    >
                      <Zap className="mr-2 h-4 w-4 text-purple-600" />
                      Perplexity → Gemini
                      {selectedWorkflow === "perplexity-to-gemini" && (
                        <span className="ml-2 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                          Đang hoạt động
                        </span>
                      )}
                    </DropdownMenuCheckboxItem>

                    <DropdownMenuCheckboxItem
                      checked={selectedWorkflow === "perplexity-chatgpt-gemini"}
                      onCheckedChange={() => setSelectedWorkflow("perplexity-chatgpt-gemini")}
                    >
                      <Lightbulb className="mr-2 h-4 w-4 text-purple-600" />
                      Perplexity → ChatGPT → Gemini
                      {selectedWorkflow === "perplexity-chatgpt-gemini" && (
                        <span className="ml-2 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                          Đang hoạt động
                        </span>
                      )}
                    </DropdownMenuCheckboxItem>
                  </DropdownMenuContent>
                </DropdownMenu>

                {/* Chips + input */}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 overflow-x-auto pb-1 pl-1">
                    {deepSearch && (
                      <span className="inline-flex items-center gap-1 rounded-full border bg-muted px-2 py-1 text-xs">
                        <Sparkles className="h-3.5 w-3.5 text-emerald-600" />
                        Nghiên cứu sâu
                        <button
                          className="ml-1 rounded p-0.5 hover:bg-white"
                          onClick={() => setDeepSearch(false)}
                          aria-label="Tắt nghiên cứu sâu"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </span>
                    )}
                    {attachments.map((attachment) => (
                      <FileChip
                        key={attachment.id}
                        attachment={attachment}
                        onRemove={() => setAttachments((prev) => prev.filter((a) => a.id !== attachment.id))}
                      />
                    ))}
                  </div>

                  <div className="flex items-end gap-2 pl-1 pr-1">
                    <textarea
                      id="textarea-input"
                      ref={taRef}
                      value={input}
                      onChange={(e) => {
                        setInput(e.target.value)
                        setInputEnter(e.target.value)
                      }}
                      placeholder="Nhập câu hỏi của bạn…"
                      rows={1}
                      className="min-h-[40px] max-h-[100px] w-full resize-none bg-transparent px-2 py-2 text-sm focus:outline-none "
                      onKeyDown={(e) => {
                        const submitByEnter = e.key === "Enter" && !e.shiftKey && !(e.metaKey || e.ctrlKey)
                        const submitByHotkey = e.key === "Enter" && (e.metaKey || e.ctrlKey)
                        if (submitByEnter || submitByHotkey) {
                          e.preventDefault()
                          handleSend()
                        }
                      }}
                    />
                    <div className="flex items-center gap-1 sm:gap-2 pb-2">
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        multiple
                        accept=".pdf,.docx,.doc,.xlsx,.xls,.xlsm,.pptx,image/*"
                        onChange={(e) => onFilesSelected(e.target.files)}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => fileInputRef.current?.click()}
                        aria-label="Đính kèm tệp"
                      >
                        <Paperclip className="h-5 w-5" />
                      </Button>
                      <div className="hidden items-center gap-2 rounded-md border px-2 py-1.5 sm:flex">
                        <span className="text-xs text-muted-foreground">DeepSearch</span>
                        <input
                          type="checkbox"
                          checked={deepSearch}
                          onChange={(e) => setDeepSearch(e.target.checked)}
                          className="h-4 w-4 accent-emerald-600"
                          aria-label="Bật tắt DeepSearch"
                        />
                      </div>
                      <Button
                        onClick={handleSend}
                        disabled={(!input.trim() && attachments.length === 0) || !active || isLoading}
                      >
                        <Send className="mr-2 h-4 w-4" />
                        <span className="hidden sm:inline">Gửi</span>
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            {/* hidden input holder */}
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              multiple
              accept=".pdf,.docx,.doc,.xlsx,.xls,.xlsm,.pptx,image/*"
              onChange={(e) => onFilesSelected(e.target.files)}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
