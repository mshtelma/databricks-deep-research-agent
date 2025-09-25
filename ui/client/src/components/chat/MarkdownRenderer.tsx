
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '@/lib/utils'
import { preprocessMarkdownMinimal } from '@/utils/minimalTableUtils'
import { processTablesSimplified } from '@/utils/simplifiedTableProcessor'
import { processTableBoundaries, hasTableBoundaries } from '@/utils/tableBoundaryProcessor'
import { TableStreamReconstructor, processStreamingWithTableReconstruction } from '@/utils/tableStreamReconstructor'

interface MarkdownRendererProps {
  content: string
  className?: string
  isStreaming?: boolean
}

export function MarkdownRenderer({ content, className, isStreaming = false }: MarkdownRendererProps) {
  // Simplified table processing to prevent malformed separators
  let processed: string

  // Simple preprocessing to fix malformed table separators
  processed = content
    .replace(/\|\s*---\s*\|\s*---\s*\|\s*---\s*\|\|\|/g, '| --- | --- | --- |') // Fix broken separators
    .replace(/(\|\s*---\s*\|.*?\|)\s*\1+/g, '$1') // Remove duplicate separator rows
    .replace(/\|\s*---\s*\|\s*---\s*\|\s*---\s*\|\s*\|(\s*\w)/g, '| --- | --- | --- |\n$1') // Fix merged separator and content

  // For streaming content, add minimal processing to avoid incomplete tables
  if (isStreaming) {
    // Just clean up obvious malformed separators without complex reconstruction
    processed = processed.replace(/(\|.*?\|)(\s*\|\s*---.*?\|){2,}/g, '$1') // Remove duplicate separators
  }
  return (
    <div className={cn('max-w-none', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Customize heading styles
          h1: ({ children }) => (
            <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
              {children}
            </h3>
          ),
          // Customize paragraph styles
          p: ({ children }) => (
            <p className="mb-2 text-gray-700 dark:text-gray-300 leading-relaxed">
              {children}
            </p>
          ),
          // Customize list styles
          ul: ({ children }) => (
            <ul className="list-disc list-inside mb-2 text-gray-700 dark:text-gray-300 space-y-1">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside mb-2 text-gray-700 dark:text-gray-300 space-y-1">
              {children}
            </ol>
          ),
          // Customize code styles
          code: ({ children, className: codeClassName }) => {
            const isInlineCode = !codeClassName
            if (isInlineCode) {
              return (
                <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-xs font-mono text-gray-800 dark:text-gray-200">
                  {children}
                </code>
              )
            }
            return (
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded-md overflow-x-auto mb-2">
                <code className="text-xs font-mono text-gray-800 dark:text-gray-200">
                  {children}
                </code>
              </pre>
            )
          },
          // Customize link styles
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              {children}
            </a>
          ),
          // Customize blockquote styles
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic text-gray-600 dark:text-gray-400 mb-2">
              {children}
            </blockquote>
          ),
          // Customize table styles
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4 rounded-lg border border-gray-200 dark:border-gray-700">
              <table className="min-w-full border-collapse bg-white dark:bg-gray-900 table-auto">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-gray-50 dark:bg-gray-800">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700 max-w-xs break-words">
              {children}
            </td>
          ),
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  )
}