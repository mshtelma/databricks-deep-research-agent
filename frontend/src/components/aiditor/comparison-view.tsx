/**
 * Comparison View for showing original vs processed markdown.
 */

import { useState, useMemo } from 'react';
import { marked } from 'marked';
import { diffWords } from 'diff';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Check, X, Copy, Download, Columns, FileText, LayoutGrid, Bug } from 'lucide-react';
import type { DebugInfo, Highlight } from './types';

type ViewMode = 'split' | 'inline' | 'overview';

interface ComparisonViewProps {
  original: string;
  processed: string;
  highlights: Highlight[];
  debugInfo?: DebugInfo | null;
  onAccept: () => void;
  onDiscard: () => void;
}

export function ComparisonView({
  original,
  processed,
  highlights,
  debugInfo,
  onAccept,
  onDiscard,
}: ComparisonViewProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const [copySuccess, setCopySuccess] = useState(false);

  // Render original with highlights
  const renderOriginal = () => {
    let html = marked.parse(original, { async: false }) as string;

    // Apply highlights
    for (const highlight of highlights) {
      const escapedText = highlight.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`(${escapedText})`, 'g');
      const style = `background-color: ${highlight.color}40; border-bottom: 2px solid ${highlight.color}; padding: 0 2px;`;
      const title = highlight.instruction
        ? `${highlight.type}: ${highlight.instruction}`
        : highlight.type;

      html = html.replace(
        regex,
        `<span style="${style}" title="${title}">$1</span>`
      );
    }

    return html;
  };

  // Render processed markdown
  const renderProcessed = () => {
    return marked.parse(processed, { async: false }) as string;
  };

  // Compute word-level inline diff
  const inlineDiffHtml = useMemo(() => {
    const changes = diffWords(original, processed);
    const parts: string[] = [];

    for (const change of changes) {
      const escaped = change.value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br/>');

      if (change.added) {
        parts.push(
          `<span style="background-color: rgba(34, 197, 94, 0.25); text-decoration: none; padding: 1px 0;">${escaped}</span>`
        );
      } else if (change.removed) {
        parts.push(
          `<span style="background-color: rgba(239, 68, 68, 0.25); text-decoration: line-through; padding: 1px 0;">${escaped}</span>`
        );
      } else {
        parts.push(escaped);
      }
    }

    return parts.join('');
  }, [original, processed]);

  // Copy processed result to clipboard
  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(processed);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Download as markdown file
  const downloadMarkdown = () => {
    const blob = new Blob([processed], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'processed.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* View Mode Toggle */}
      <div className="flex items-center gap-2">
        <Button
          variant={viewMode === 'split' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('split')}
          className="gap-1"
        >
          <Columns className="h-4 w-4" />
          Split View
        </Button>
        <Button
          variant={viewMode === 'inline' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('inline')}
          className="gap-1"
        >
          <FileText className="h-4 w-4" />
          Inline Diff
        </Button>
        <Button
          variant={viewMode === 'overview' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('overview')}
          className="gap-1"
        >
          <LayoutGrid className="h-4 w-4" />
          Overview
        </Button>
      </div>

      {/* Comparison Summary */}
      <Card className="bg-muted/30">
        <CardHeader className="py-3">
          <CardTitle className="text-sm font-medium">Comparison Summary</CardTitle>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="flex gap-4 text-sm">
            <span className="text-muted-foreground">
              Highlights applied: <strong>{highlights.length}</strong>
            </span>
            <span className="text-muted-foreground">
              Original length: <strong>{original.length}</strong> chars
            </span>
            <span className="text-muted-foreground">
              Processed length: <strong>{processed.length}</strong> chars
            </span>
            <span className="text-muted-foreground">
              Delta: <strong>{processed.length - original.length > 0 ? '+' : ''}{processed.length - original.length}</strong> chars
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Content Area */}
      {viewMode === 'split' && (
        <div className="grid grid-cols-2 gap-4 flex-1">
          {/* Original + Edits */}
          <Card className="flex flex-col">
            <CardHeader className="py-3 border-b bg-muted/30">
              <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Original + Edits
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1">
              <ScrollArea className="h-[400px]">
                <div
                  className="p-4 markdown-preview text-sm"
                  dangerouslySetInnerHTML={{ __html: renderOriginal() }}
                />
              </ScrollArea>
            </CardContent>
          </Card>

          {/* LLM Output */}
          <Card className="flex flex-col">
            <CardHeader className="py-3 border-b bg-muted/30">
              <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                LLM Output
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1">
              <ScrollArea className="h-[400px]">
                <div
                  className="p-4 markdown-preview text-sm"
                  dangerouslySetInnerHTML={{ __html: renderProcessed() }}
                />
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}

      {viewMode === 'inline' && (
        <Card className="flex-1">
          <CardHeader className="py-3 border-b bg-muted/30">
            <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Inline Diff
              <span className="ml-3 text-[10px] font-normal normal-case tracking-normal">
                <span style={{ backgroundColor: 'rgba(239, 68, 68, 0.25)', textDecoration: 'line-through', padding: '0 4px' }}>removed</span>
                {' '}
                <span style={{ backgroundColor: 'rgba(34, 197, 94, 0.25)', padding: '0 4px' }}>added</span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[450px]">
              <div
                className="p-4 text-sm font-mono whitespace-pre-wrap leading-relaxed"
                dangerouslySetInnerHTML={{ __html: inlineDiffHtml }}
              />
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {viewMode === 'overview' && (
        <div className="grid grid-cols-2 gap-4 flex-1">
          <Card className="flex flex-col">
            <CardHeader className="py-3 border-b bg-muted/30">
              <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Original + Edits
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground italic">
                {highlights.length > 0
                  ? `${highlights.length} edits marked for processing.`
                  : 'No edits marked.'}
              </p>
              <ul className="mt-2 space-y-1">
                {highlights.map((h, i) => (
                  <li key={i} className="text-xs flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: h.color }}
                    />
                    <span className="font-medium">{h.type}:</span>
                    <span className="text-muted-foreground truncate max-w-[200px]">
                      &quot;{h.text.substring(0, 30)}...&quot;
                    </span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="flex flex-col">
            <CardHeader className="py-3 border-b bg-muted/30">
              <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                LLM Output
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground italic">
                [LLM response ready for review]
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                Review the changes in Split View or Inline Diff, then accept or discard.
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Debug Panel */}
      {debugInfo && (
        <details className="border rounded-lg">
          <summary className="px-4 py-2 cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground flex items-center gap-2">
            <Bug className="h-4 w-4" />
            LLM Debug Info
          </summary>
          <div className="border-t divide-y">
            <div className="p-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                System Prompt
              </h4>
              <pre className="text-xs bg-muted/50 p-3 rounded-md overflow-auto max-h-[200px] whitespace-pre-wrap">
                {debugInfo.system_prompt}
              </pre>
            </div>
            <div className="p-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                User Message
              </h4>
              <pre className="text-xs bg-muted/50 p-3 rounded-md overflow-auto max-h-[300px] whitespace-pre-wrap">
                {debugInfo.user_message}
              </pre>
            </div>
            <div className="p-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                Raw LLM Response
              </h4>
              <pre className="text-xs bg-muted/50 p-3 rounded-md overflow-auto max-h-[300px] whitespace-pre-wrap">
                {debugInfo.raw_response}
              </pre>
            </div>
          </div>
        </details>
      )}

      {/* Action Bar */}
      <div className="flex items-center gap-2 pt-2 border-t">
        <Button
          variant="default"
          size="sm"
          onClick={onAccept}
          className="gap-1"
        >
          <Check className="h-4 w-4" />
          Accept Edits
        </Button>

        <Button variant="outline" size="sm" onClick={copyToClipboard} className="gap-1">
          <Copy className="h-4 w-4" />
          {copySuccess ? 'Copied!' : 'Copy to Clipboard'}
        </Button>

        <Button variant="outline" size="sm" onClick={downloadMarkdown} className="gap-1">
          <Download className="h-4 w-4" />
          Download Markdown
        </Button>

        <div className="flex-1" />

        <Button variant="destructive" size="sm" onClick={onDiscard} className="gap-1">
          <X className="h-4 w-4" />
          Discard
        </Button>
      </div>
    </div>
  );
}
