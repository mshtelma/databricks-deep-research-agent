import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '@/lib/utils';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

/**
 * Renders markdown content with support for:
 * - GitHub Flavored Markdown (tables, strikethrough, task lists)
 * - Syntax-highlighted code blocks
 * - Links that open in new tabs
 * - Dark mode compatible styling
 */
export const MarkdownRenderer = React.memo(function MarkdownRenderer({
  content,
  className,
}: MarkdownRendererProps) {
  const isDarkMode =
    typeof document !== 'undefined' &&
    document.documentElement.classList.contains('dark');

  return (
    <ReactMarkdown
      className={cn('prose prose-sm dark:prose-invert max-w-none', className)}
      remarkPlugins={[remarkGfm]}
      components={{
        // Links open in new tabs
        a: ({ href, children, ...props }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {children}
          </a>
        ),
        // Code blocks with syntax highlighting
        code: ({ className, children, ...props }) => {
          const match = /language-(\w+)/.exec(className || '');
          const language = match ? match[1] : undefined;
          const codeString = String(children).replace(/\n$/, '');
          const isInline = !language && !codeString.includes('\n');

          if (isInline) {
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }

          return (
            <SyntaxHighlighter
              style={isDarkMode ? vscDarkPlus : vs}
              language={language || 'text'}
              PreTag="div"
              customStyle={{
                margin: 0,
                borderRadius: '0.5rem',
                fontSize: '0.875rem',
              }}
            >
              {codeString}
            </SyntaxHighlighter>
          );
        },
        // Avoid double wrapping with SyntaxHighlighter
        pre: ({ children }) => <>{children}</>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
});

export default MarkdownRenderer;
