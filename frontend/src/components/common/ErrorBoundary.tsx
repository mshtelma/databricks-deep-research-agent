import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Fallback component to render when an error occurs */
  fallback?: ReactNode;
  /** Name of the boundary for logging purposes */
  name?: string;
  /** Callback when error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary component that catches JavaScript errors in child components.
 * Prevents the entire app from crashing when a component fails to render.
 *
 * Usage:
 * ```tsx
 * <ErrorBoundary name="MessageList">
 *   <MessageList messages={messages} />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    const { name, onError } = this.props;

    // Log error with boundary name for easier debugging
    console.error(
      `[ErrorBoundary${name ? `: ${name}` : ''}] Caught error:`,
      error,
      errorInfo
    );

    // Call optional error callback
    onError?.(error, errorInfo);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  public render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, fallback, name } = this.props;

    if (hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default fallback UI
      return (
        <div className="flex flex-col items-center justify-center p-6 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800 m-4">
          <div className="text-red-600 dark:text-red-400 mb-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-8 w-8"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-1">
            Something went wrong
          </h3>
          <p className="text-sm text-red-600 dark:text-red-400 mb-4 text-center max-w-md">
            {name && <span className="font-medium">[{name}] </span>}
            {error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={this.handleReset}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors"
          >
            Try Again
          </button>
        </div>
      );
    }

    return children;
  }
}

/**
 * Higher-order component to wrap a component with an error boundary.
 *
 * Usage:
 * ```tsx
 * const SafeMessageList = withErrorBoundary(MessageList, 'MessageList');
 * ```
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  name?: string
): React.FC<P> {
  const SafeComponent: React.FC<P> = (props) => (
    <ErrorBoundary name={name}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  SafeComponent.displayName = `withErrorBoundary(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;

  return SafeComponent;
}
