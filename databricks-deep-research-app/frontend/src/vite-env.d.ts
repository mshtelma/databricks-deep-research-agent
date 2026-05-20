/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
  /** Set to '1' to enable the client-side metrics pipeline. */
  readonly VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
