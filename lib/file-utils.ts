import * as XLSX from "xlsx";
import Papa from "papaparse";
import mammoth from "mammoth";

export type FileContent = {
  type:
    | "text"
    | "image"
    | "table"
    | "error"
    | "file"
    | "pdf-native"
    | "file-native"
    | "pdf-url";
  content: string;
  metadata?: {
    pages?: number;
    sheets?: string[];
    rows?: number;
    columns?: number;
    fileType?: string;
    fileName?: string;
    mimeType?: string;
    size?: number;
    url?: string;
  };
  // For OpenRouter native processing
  fileData?: {
    name: string;
    type: string;
    data: string; // base64 encoded or URL
  };
};

// Map nhận diện loại file theo pattern (mime hoặc đuôi)
const TYPE_PATTERNS = [
  { type: "image", patterns: [/^image\//] },
  { type: "pdf", patterns: [/^application\/pdf$/, /\.pdf$/i] },
  { type: "word", patterns: [/word/, /\.(doc|docx)$/i] },
  { type: "excel", patterns: [/excel|spreadsheetml/, /\.(xls|xlsx)$/i] },
  { type: "text", patterns: [/^text\//, /\.(txt|csv|log|md)$/i] },
];

function detectMainType(file: any) {
  const mime = file.type || "";
  const name = file.name || "";
  console.log("mime", mime);
  for (const { type, patterns } of TYPE_PATTERNS) {
    if (patterns.some((p) => p.test(mime) || p.test(name))) return type;
  }
  // nhóm chung theo prefix mime nếu chưa khớp
  if (/^(audio|video)\//.test(mime)) return mime.split("/")[0]; // "audio" | "video"
  if (/^application\//.test(mime)) return "file";
  return "file"; // fallback
}

export async function processFile(file: File): Promise<FileContent> {
  const fileType = file.type.toLowerCase();
  const fileName = file.name.toLowerCase();

  console.log("Processing file:", {
    name: file.name,
    type: file.type,
    size: file.size,
  });

  try {
    // Images - convert to base64 for vision models
    // if (fileType.startsWith("image/")) {
    //   console.log("Processing as image");
    //   const base64 = await fileToBase64(file);
    //   return {
    //     type: "image",
    //     content: base64,
    //     metadata: {
    //       fileType: "image",
    //       fileName: file.name,
    //       mimeType: file.type,
    //       size: file.size,
    //     },
    //   };
    // } else {
    const base64 = await fileToBase64(file); // data:*/*;base64,AAAA...
    const mainType = detectMainType(file);
    const mime = file.type || "";
    const name = file.name || "unknown";
    console.log("mainType", mainType);

    return {
      type: mainType, // "image" | "pdf" | "word" | "excel" | "text" | "file" | "audio" | "video"
      content: base64, // luôn base64, đồng bộ cấu trúc với image
      metadata: {
        fileType: mainType,
        fileName: name,
        mimeType: mime,
        size: file.size,
      },
      fileData: {
        name,
        type: mime,
        data: base64.split(",")[1], // bỏ prefix data:...;base64,
      },
    };
    // }

    // PDF files - handle as normal file upload
    if (fileType === "application/pdf" || fileName.endsWith(".pdf")) {
      console.log("Processing PDF as normal file");
      const base64 = await fileToBase64(file);

      return {
        type: "file",
        content: `📄 **PDF File: ${file.name}** (${formatBytes(file.size)})`,
        metadata: {
          fileType: "pdf",
          fileName: file.name,
          mimeType: file.type,
          size: file.size,
        },
        fileData: {
          name: file.name,
          type: file.type,
          data: base64.split(",")[1], // Remove data:application/pdf;base64, prefix
        },
      };
    }

    // Excel files - process locally for better control
    if (
      fileType ===
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ||
      fileType === "application/vnd.ms-excel" ||
      fileType === "application/msexcel" ||
      fileName.endsWith(".xlsx") ||
      fileName.endsWith(".xls") ||
      fileName.endsWith(".xlsm")
    ) {
      console.log("Processing as Excel");
      return await processExcel(file);
    }

    // Word documents - process locally
    if (
      fileType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
      fileType === "application/msword" ||
      fileName.endsWith(".docx") ||
      fileName.endsWith(".doc")
    ) {
      console.log("Processing as Word");
      return await processWord(file);
    }

    // CSV files
    if (fileType === "text/csv" || fileName.endsWith(".csv")) {
      console.log("Processing as CSV");
      return await processCSV(file);
    }

    // Text files
    if (
      fileType.startsWith("text/") ||
      fileName.endsWith(".txt") ||
      fileName.endsWith(".md")
    ) {
      console.log("Processing as text");
      return await processText(file);
    }

    // JSON files
    if (fileType === "application/json" || fileName.endsWith(".json")) {
      console.log("Processing as JSON");
      return await processJSON(file);
    }

    // PowerPoint files
    if (
      fileType ===
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" ||
      fileName.endsWith(".pptx")
    ) {
      console.log("Processing as PowerPoint");
      return await processPowerPoint(file);
    }

    // Other supported files - send to OpenRouter for native processing
    if (
      OPENROUTER_SUPPORTED_TYPES.includes(fileType) ||
      file.name.match(/\.(docx?|xlsx?|pptx?)$/i)
    ) {
      console.log("Processing as file for OpenRouter native processing");
      const base64 = await fileToBase64(file);

      return {
        type: "file-native",
        content: `📎 **File: ${file.name}** (${formatBytes(file.size)})

Đang sử dụng OpenRouter native file processing để đọc nội dung file này...`,
        metadata: {
          fileType: "file-native",
          fileName: file.name,
          mimeType: file.type,
          size: file.size,
        },
        fileData: {
          name: file.name,
          type: file.type,
          data: base64.split(",")[1], // Remove data URL prefix
        },
      };
    }

    console.log("Unsupported file type");
    return {
      type: "error",
      content: `Loại file không được hỗ trợ: ${
        file.type || "unknown"
      }\nTên file: ${file.name}`,
      metadata: { fileType: "unsupported", fileName: file.name },
    };
  } catch (error) {
    console.error("Error processing file:", error);
    return {
      type: "error",
      content: `Lỗi khi đọc file "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      metadata: { fileType: "error", fileName: file.name },
    };
  }
}

export async function processPDFUrl(url: string): Promise<FileContent> {
  try {
    // Validate URL
    const urlObj = new URL(url);
    if (!urlObj.protocol.startsWith("http")) {
      throw new Error("URL phải bắt đầu với http:// hoặc https://");
    }

    // Extract filename from URL
    const pathname = urlObj.pathname;
    const filename = pathname.split("/").pop() || "document.pdf";

    console.log("Processing PDF URL:", url);

    return {
      type: "pdf-url",
      content: `📄 **PDF từ URL: ${filename}**

**URL:** ${url}

Đang sử dụng OpenRouter native PDF processing để đọc nội dung từ URL này...`,
      metadata: {
        fileType: "pdf-url",
        fileName: filename,
        url: url,
      },
      fileData: {
        name: filename,
        type: "application/pdf",
        data: url, // Send URL directly instead of base64
      },
    };
  } catch (error) {
    console.error("PDF URL processing error:", error);
    return {
      type: "error",
      content: `Không thể xử lý PDF URL: ${
        error instanceof Error ? error.message : "Unknown error"
      }

**Gợi ý:**
- Đảm bảo URL hợp lệ và bắt đầu với https://
- URL phải trỏ trực tiếp đến file PDF
- File PDF phải có thể truy cập công khai`,
      metadata: { fileType: "pdf-url-error", url: url },
    };
  }
}

export function isPDFUrl(text: string): boolean {
  try {
    const url = new URL(text.trim());
    return (
      url.protocol.startsWith("http") &&
      (url.pathname.toLowerCase().endsWith(".pdf") ||
        text.toLowerCase().includes("pdf"))
    );
  } catch {
    return false;
  }
}

// Keep existing helper functions...
async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
}

async function fileToArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = (error) => reject(error);
  });
}

async function fileToText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsText(file, "utf-8");
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
}

// Keep all existing processing functions (processExcel, processWord, etc.)
// ... (keeping the same implementations as before)

async function processExcel(file: File): Promise<FileContent> {
  try {
    console.log("Processing Excel file:", file.name);
    const arrayBuffer = await fileToArrayBuffer(file);
    console.log(
      "Excel file read as ArrayBuffer, size:",
      arrayBuffer.byteLength
    );

    const workbook = XLSX.read(arrayBuffer, { type: "array" });
    console.log("Excel workbook parsed, sheets:", workbook.SheetNames);

    const sheetNames = workbook.SheetNames;
    let content = `📊 **File Excel: ${file.name}**\n\n`;

    sheetNames.forEach((sheetName, index) => {
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      content += `**Sheet ${index + 1}: ${sheetName}**\n`;
      content += `Số dòng: ${jsonData.length}\n\n`;

      // Show first 10 rows
      jsonData.slice(0, 10).forEach((row: any, rowIndex) => {
        if (Array.isArray(row) && row.length > 0) {
          content += row.map((cell) => cell || "").join(" | ") + "\n";
        }
      });

      if (jsonData.length > 10) {
        content += `... và ${jsonData.length - 10} dòng khác\n`;
      }
      content += "\n---\n\n";
    });

    console.log("Excel processing completed successfully");
    return {
      type: "table",
      content: content.trim(),
      metadata: {
        sheets: sheetNames,
        rows: XLSX.utils.sheet_to_json(workbook.Sheets[sheetNames[0]] || {})
          .length,
        fileType: "excel",
        fileName: file.name,
      },
    };
  } catch (error) {
    console.error("Excel processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file Excel "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }

**Gợi ý:**
- Đảm bảo file không bị hỏng
- Thử lưu lại file Excel
- Hoặc xuất ra CSV rồi upload lại`,
      metadata: { fileType: "excel-error", fileName: file.name },
    };
  }
}

async function processWord(file: File): Promise<FileContent> {
  try {
    console.log("Processing Word file:", file.name);
    const arrayBuffer = await fileToArrayBuffer(file);
    console.log("Word file read as ArrayBuffer, size:", arrayBuffer.byteLength);

    const result = await mammoth.extractRawText({ arrayBuffer });
    console.log("Word text extracted, length:", result.value.length);

    if (!result.value || result.value.trim().length === 0) {
      return {
        type: "error",
        content: `File Word "${file.name}" có vẻ trống hoặc không thể đọc được.

**Gợi ý:**
- Kiểm tra file có nội dung không
- Thử lưu lại file Word
- Hoặc copy nội dung và paste trực tiếp`,
        metadata: { fileType: "word-empty", fileName: file.name },
      };
    }

    const content = `📝 **File Word: ${file.name}**

**Kích thước:** ${formatBytes(file.size)}
**Số ký tự:** ${result.value.length}

**Nội dung:**

${result.value}`;

    console.log("Word processing completed successfully");
    return {
      type: "text", // Keep as "text" type for local processing
      content: content,
      metadata: { fileType: "word", fileName: file.name },
    };
  } catch (error) {
    console.error("Word processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file Word "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }

**Gợi ý:**
- Đảm bảo file là định dạng .docx (không phải .doc cũ)
- Thử lưu lại file Word
- Hoặc copy nội dung và paste trực tiếp`,
      metadata: { fileType: "word-error", fileName: file.name },
    };
  }
}

async function processCSV(file: File): Promise<FileContent> {
  try {
    console.log("Processing CSV file:", file.name);
    const text = await fileToText(file);
    const parsed = Papa.parse(text, { header: false });

    let content = `📋 **File CSV: ${file.name}**\n\n`;

    // Show first 20 rows
    parsed.data.slice(0, 20).forEach((row: any, index) => {
      if (Array.isArray(row) && row.length > 0) {
        content += row.join(" | ") + "\n";
      }
    });

    if (parsed.data.length > 20) {
      content += `\n... và ${parsed.data.length - 20} dòng khác`;
    }

    return {
      type: "table",
      content: content.trim(),
      metadata: {
        rows: parsed.data.length,
        columns: parsed.data[0]?.length || 0,
        fileType: "csv",
        fileName: file.name,
      },
    };
  } catch (error) {
    console.error("CSV processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file CSV "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      metadata: { fileType: "csv-error", fileName: file.name },
    };
  }
}

async function processText(file: File): Promise<FileContent> {
  try {
    console.log("Processing text file:", file.name);
    const text = await fileToText(file);

    const content = `📄 **File Text: ${file.name}**

**Kích thước:** ${formatBytes(file.size)}
**Số ký tự:** ${text.length}

**Nội dung:**

${text}`;

    return {
      type: "text",
      content: content,
      metadata: { fileType: "text", fileName: file.name },
    };
  } catch (error) {
    console.error("Text processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file text "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      metadata: { fileType: "text-error", fileName: file.name },
    };
  }
}

async function processJSON(file: File): Promise<FileContent> {
  try {
    console.log("Processing JSON file:", file.name);
    const text = await fileToText(file);
    const parsed = JSON.parse(text);
    const formatted = JSON.stringify(parsed, null, 2);

    const content = `🔧 **File JSON: ${file.name}**

**Kích thước:** ${formatBytes(file.size)}

**Nội dung:**

\`\`\`json
${formatted}
\`\`\``;

    return {
      type: "text",
      content: content,
      metadata: { fileType: "json", fileName: file.name },
    };
  } catch (error) {
    console.error("JSON processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file JSON "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      metadata: { fileType: "json-error", fileName: file.name },
    };
  }
}

async function processPowerPoint(file: File): Promise<FileContent> {
  try {
    console.log("Processing PowerPoint file:", file.name);

    // For now, we'll use OpenRouter native processing for PowerPoint
    // since extracting PPTX content requires complex parsing
    const base64 = await fileToBase64(file);

    return {
      type: "file-native",
      content: `📊 **File PowerPoint: ${file.name}** (${formatBytes(file.size)})

Đang sử dụng AI để đọc nội dung PowerPoint này...

**Thông tin file:**
- Định dạng: ${file.type}
- Kích thước: ${formatBytes(file.size)}

AI sẽ phân tích và trích xuất nội dung từ các slide trong presentation.`,
      metadata: {
        fileType: "powerpoint-native",
        fileName: file.name,
        mimeType: file.type,
        size: file.size,
      },
      fileData: {
        name: file.name,
        type: file.type,
        data: base64.split(",")[1], // Remove data URL prefix
      },
    };
  } catch (error) {
    console.error("PowerPoint processing error:", error);
    return {
      type: "error",
      content: `Không thể đọc file PowerPoint "${file.name}": ${
        error instanceof Error ? error.message : "Unknown error"
      }

**Gợi ý thay thế:**
1. **Xuất ra PDF:** File → Export → PDF → Upload PDF  
2. **Chụp ảnh slides:** Chụp ảnh các slide quan trọng
3. **Copy nội dung:** Copy text từ slides và paste vào tin nhắn
4. **Mô tả nội dung:** Mô tả presentation để tôi hỗ trợ tốt hơn`,
      metadata: { fileType: "powerpoint-error", fileName: file.name },
    };
  }
}

export function getFileIcon(file: File): string {
  const fileType = file.type.toLowerCase();
  const fileName = file.name.toLowerCase();

  if (fileType.startsWith("image/")) return "🖼️";
  if (fileType === "application/pdf" || fileName.endsWith(".pdf")) return "📄";
  if (
    fileType.includes("spreadsheet") ||
    fileName.endsWith(".xlsx") ||
    fileName.endsWith(".xls")
  )
    return "📊";
  if (fileType.includes("wordprocessing") || fileName.endsWith(".docx"))
    return "📝";
  if (fileType.includes("presentation") || fileName.endsWith(".pptx"))
    return "📊";
  if (fileType === "text/csv" || fileName.endsWith(".csv")) return "📋";
  if (fileType === "application/json" || fileName.endsWith(".json"))
    return "🔧";
  if (fileType.startsWith("text/")) return "📄";

  return "📎";
}

export function isImageFile(file: File): boolean {
  return file.type.startsWith("image/");
}

// OpenRouter supported file types for native processing
export const OPENROUTER_SUPPORTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
];

export const SUPPORTED_FILE_TYPES = [
  // Images
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  // Documents
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
  "application/msword", // .doc
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
  "application/vnd.ms-excel", // .xls
  "application/msexcel",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
  "text/csv",
  "application/json",
  "text/plain",
  "text/markdown",
];

export const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25MB

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}
