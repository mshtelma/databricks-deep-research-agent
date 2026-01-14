import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '@/lib/utils';
import {
  remarkCitations,
  buildLinkCitationIndex,
  getCitationIndexForUrl,
  type LinkCitationInfo,
} from '@/lib/markdown/citationPlugin';
import { CitationMarker } from '@/components/citations';
import type { Claim, VerificationVerdict } from '@/types/citation';

interface MarkdownRendererProps {
  content: string;
  className?: string;
  /** Enable citation marker rendering */
  enableCitations?: boolean;
  /**
   * Citation mode:
   * - 'numeric': Parse [N] or [Key] markers from text (default)
   * - 'link': Convert [Title](url) links to citation markers
   * - 'auto': Detect based on content
   */
  citationMode?: 'numeric' | 'link' | 'auto';
  /** Citation data indexed by citation key (e.g., "Arxiv", "Zhipu") */
  citationData?: Map<string, CitationContext>;
  /** Callback when a citation is clicked */
  onCitationClick?: (citationKey: string, info?: LinkCitationInfo) => void;
  /** Callback when a citation is hovered - includes element for positioning */
  onCitationHover?: (citationKey: string | null, element?: HTMLElement | null) => void;
  /** Currently active citation key */
  activeCitationKey?: string | null;
}

/** Context data for a citation marker */
export interface CitationContext {
  claim: Claim;
  verdict: VerificationVerdict | null;
  /** URL from the evidence span's source for navigation */
  url?: string;
}

/**
 * Renders markdown content with support for:
 * - GitHub Flavored Markdown (tables, strikethrough, task lists)
 * - Syntax-highlighted code blocks
 * - Links that open in new tabs
 * - Dark mode compatible styling
 * - Inline citation markers [N] with evidence cards
 * - Link-based citations [Title](url) rendered as superscript markers
 */
export const MarkdownRenderer = React.memo(function MarkdownRenderer({
  content,
  className,
  enableCitations = false,
  citationMode = 'auto',
  citationData,
  onCitationClick,
  onCitationHover,
  activeCitationKey,
}: MarkdownRendererProps) {
  const isDarkMode =
    typeof document !== 'undefined' &&
    document.documentElement.classList.contains('dark');

  // Build link citation index for link mode
  const linkCitationIndex = React.useMemo(() => {
    if (!enableCitations) return new Map<number, LinkCitationInfo>();
    if (citationMode === 'numeric') return new Map<number, LinkCitationInfo>();
    return buildLinkCitationIndex(content);
  }, [content, enableCitations, citationMode]);

  // Determine effective citation mode
  const effectiveMode = React.useMemo(() => {
    if (!enableCitations) return 'none';
    if (citationMode !== 'auto') return citationMode;
    // Auto-detect: prefer link mode if we found links, otherwise numeric
    return linkCitationIndex.size > 0 ? 'link' : 'numeric';
  }, [enableCitations, citationMode, linkCitationIndex]);

  // Build remark plugins array
  const remarkPlugins = React.useMemo(() => {
    const plugins: Array<typeof remarkGfm | typeof remarkCitations> = [remarkGfm];
    // Only use remark citation plugin for numeric mode
    if (enableCitations && effectiveMode === 'numeric') {
      plugins.push(remarkCitations);
    }
    return plugins;
  }, [enableCitations, effectiveMode]);

  // Build custom components with citation support
  const components = React.useMemo(() => {
    const baseComponents: Record<string, React.FC<Record<string, unknown>>> = {
      // Code blocks with syntax highlighting
      code: ({ className: codeClassName, children, ...props }: { className?: string; children?: React.ReactNode } & Record<string, unknown>) => {
        const match = /language-(\w+)/.exec(codeClassName || '');
        const language = match ? match[1] : undefined;
        const codeString = String(children).replace(/\n$/, '');
        const isInline = !language && !codeString.includes('\n');

        if (isInline) {
          return (
            <code className={codeClassName} {...props}>
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
      pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    };

    // Handle links based on citation mode
    if (enableCitations && effectiveMode === 'link') {
      // In link mode, convert markdown links to citation markers
      baseComponents['a'] = ({ href, children, ...props }: { href?: string; children?: React.ReactNode } & Record<string, unknown>) => {
        if (!href) {
          return <span {...props}>{children}</span>;
        }

        // Skip anchor and javascript links
        if (href.startsWith('#') || href.startsWith('javascript:')) {
          return (
            <a href={href} {...props}>
              {children}
            </a>
          );
        }

        // Get citation index for this URL
        const citationIndex = getCitationIndexForUrl(href, linkCitationIndex);
        if (citationIndex !== undefined) {
          const info = linkCitationIndex.get(citationIndex);
          const citationKey = String(citationIndex);
          return (
            <CitationMarker
              citationKey={citationKey}
              index={citationIndex}
              title={info?.title}
              url={href}
              isActive={activeCitationKey === citationKey}
              onClick={() => onCitationClick?.(citationKey, info)}
              onMouseEnter={(e) => onCitationHover?.(citationKey, e.currentTarget)}
              onMouseLeave={() => onCitationHover?.(null, null)}
            />
          );
        }

        // Fallback: regular link (shouldn't happen if index is built correctly)
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {children}
          </a>
        );
      };
    } else {
      // Default link handling - open in new tab
      baseComponents['a'] = ({ href, children, ...props }: { href?: string; children?: React.ReactNode } & Record<string, unknown>) => (
        <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
          {children}
        </a>
      );
    }

    // Add citation-marker component for numeric/key mode
    if (enableCitations && effectiveMode === 'numeric') {
      baseComponents['citation-marker'] = (props: Record<string, unknown>) => {
        // Get citation key from the parser (e.g., "Arxiv", "Zhipu", or legacy "1", "2")
        const citationKey = String(props['data-citation-key'] || '');
        if (!citationKey) return null;

        // Legacy index support
        const index = props['data-citation-index'] !== undefined
          ? Number(props['data-citation-index'])
          : undefined;

        const citationContext = citationData?.get(citationKey);
        const verdict = citationContext?.verdict ?? null;
        const url = citationContext?.url;

        // Track if citation data is missing (claims not yet loaded or no match found)
        const isUnresolved = !citationContext;

        return (
          <CitationMarker
            citationKey={citationKey}
            index={index}
            verdict={verdict}
            url={url}
            isUnresolved={isUnresolved}
            isActive={activeCitationKey === citationKey}
            onClick={() => onCitationClick?.(citationKey)}
            onMouseEnter={(e) => onCitationHover?.(citationKey, e.currentTarget)}
            onMouseLeave={() => onCitationHover?.(null, null)}
          />
        );
      };
    }

    return baseComponents;
  }, [
    isDarkMode,
    enableCitations,
    effectiveMode,
    citationData,
    linkCitationIndex,
    activeCitationKey,
    onCitationClick,
    onCitationHover,
  ]);

  return (
    <ReactMarkdown
      className={cn('prose prose-sm dark:prose-invert max-w-none', className)}
      remarkPlugins={remarkPlugins}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
});

export default MarkdownRenderer;
