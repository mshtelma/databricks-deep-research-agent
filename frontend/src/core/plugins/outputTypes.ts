/**
 * Default Output Type Renderers
 *
 * Provides fallback renderers for standard output types and
 * generic rendering for unknown types.
 */

import React from 'react';
import type { ComponentType } from 'react';
import type { OutputRenderer, OutputRendererProps } from './types';

/**
 * Generic JSON renderer for unknown output types.
 * Renders output as formatted JSON with syntax highlighting.
 */
const GenericJsonRenderer: ComponentType<OutputRendererProps> = ({
  data,
  context,
  className = '',
}) => {
  const isDark = context.theme === 'dark';

  return React.createElement(
    'div',
    {
      className: `p-4 rounded-lg ${isDark ? 'bg-gray-800' : 'bg-gray-50'} ${className}`,
    },
    React.createElement(
      'pre',
      {
        className: `text-sm overflow-auto ${isDark ? 'text-gray-300' : 'text-gray-700'}`,
      },
      JSON.stringify(data, null, 2)
    )
  );
};

/**
 * Default synthesis report renderer.
 * Renders the standard SynthesisReport output type with title, content, and sources.
 */
const SynthesisReportRenderer: ComponentType<OutputRendererProps> = ({
  data,
  context,
  className = '',
}) => {
  const isDark = context.theme === 'dark';
  const title = data.title as string | undefined;
  const content = data.content as string | undefined;
  const summary = data.summary as string | undefined;
  const keyFindings = data.key_findings as string[] | undefined;
  const sources = data.sources as Array<{ title?: string; url?: string }> | undefined;

  return React.createElement(
    'article',
    {
      className: `prose max-w-none ${isDark ? 'prose-invert' : ''} ${className}`,
    },
    [
      // Title
      title &&
        React.createElement('h1', { key: 'title', className: 'text-2xl font-bold mb-4' }, title),

      // Summary
      summary &&
        React.createElement(
          'div',
          {
            key: 'summary',
            className: `mb-6 p-4 rounded-lg ${isDark ? 'bg-blue-900/20' : 'bg-blue-50'}`,
          },
          React.createElement('h2', { className: 'text-lg font-semibold mb-2' }, 'Summary'),
          React.createElement('p', null, summary)
        ),

      // Key Findings
      keyFindings &&
        keyFindings.length > 0 &&
        React.createElement(
          'div',
          { key: 'findings', className: 'mb-6' },
          React.createElement('h2', { className: 'text-lg font-semibold mb-2' }, 'Key Findings'),
          React.createElement(
            'ul',
            { className: 'list-disc list-inside space-y-1' },
            keyFindings.map((finding, idx) =>
              React.createElement('li', { key: idx }, finding)
            )
          )
        ),

      // Main Content
      content &&
        React.createElement(
          'div',
          { key: 'content', className: 'mb-6 whitespace-pre-wrap' },
          content
        ),

      // Sources
      sources &&
        sources.length > 0 &&
        React.createElement(
          'div',
          {
            key: 'sources',
            className: `mt-8 pt-4 border-t ${isDark ? 'border-gray-700' : 'border-gray-200'}`,
          },
          React.createElement('h2', { className: 'text-lg font-semibold mb-2' }, 'Sources'),
          React.createElement(
            'ul',
            { className: 'space-y-1 text-sm' },
            sources.map((source, idx) =>
              React.createElement(
                'li',
                { key: idx },
                source.url
                  ? React.createElement(
                      'a',
                      {
                        href: source.url,
                        target: '_blank',
                        rel: 'noopener noreferrer',
                        className: 'text-blue-600 hover:underline',
                      },
                      source.title || source.url
                    )
                  : source.title || 'Unknown source'
              )
            )
          )
        ),
    ].filter(Boolean)
  );
};

/**
 * Default output renderers for built-in types.
 */
export const defaultOutputRenderers: OutputRenderer[] = [
  {
    id: 'synthesis-report-renderer',
    outputType: 'synthesis_report',
    displayName: 'Research Report',
    description: 'Default renderer for synthesis reports with citations',
    component: SynthesisReportRenderer,
  },
  {
    id: 'generic-json-renderer',
    outputType: '__generic__',
    displayName: 'Generic Output',
    description: 'Fallback renderer for unknown output types',
    component: GenericJsonRenderer,
  },
];

/**
 * Get the appropriate renderer for an output type.
 * Falls back to generic JSON renderer if no specific renderer exists.
 */
export function getRendererForType(
  outputType: string,
  registeredRenderers: Map<string, OutputRenderer>
): OutputRenderer {
  // Check for exact match
  const exactMatch = registeredRenderers.get(outputType);
  if (exactMatch) {
    return exactMatch;
  }

  // Check for synthesis_report (default)
  if (outputType === 'synthesis_report') {
    const synthesisRenderer = defaultOutputRenderers.find(r => r.outputType === 'synthesis_report');
    if (synthesisRenderer) {
      return synthesisRenderer;
    }
  }

  // Fall back to generic
  const genericRenderer = defaultOutputRenderers.find(r => r.outputType === '__generic__');
  if (genericRenderer) {
    return genericRenderer;
  }

  // Ultimate fallback - should never happen with defaultOutputRenderers populated
  return {
    id: 'fallback-renderer',
    outputType: '__fallback__',
    displayName: 'Fallback',
    component: GenericJsonRenderer,
  };
}

export { GenericJsonRenderer, SynthesisReportRenderer };
