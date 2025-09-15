import { Loader2 } from 'lucide-react'

export default function Loading() {
  return (
    <div
      className="flex h-screen w-full items-center justify-center bg-white"
      aria-busy="true"
      aria-live="polite"
    >
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-6 w-6 animate-spin text-emerald-600" />
        <p className="text-sm text-muted-foreground">Đang tải…</p>
      </div>
    </div>
  )
}
