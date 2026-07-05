import React from 'react';
import ReactDOM from 'react-dom/client';

import { ShellApp } from './ShellApp';
import '../styles/globals.css';
import {
  configureClientErrorReporting,
  installGlobalErrorReporting,
} from '../lib/clientErrors';

// Ship uncaught client errors to the shell backend so a browser crash lands in
// the deployed app's logs (mirrors the main app; the shell has its own route).
configureClientErrorReporting({
  endpoint: '/api/observability/client-errors',
  bundleId: __BUILD_ID__,
});
installGlobalErrorReporting();

// The standalone deployed shell-app: one agent, no chat history, no TanStack /
// router / auth — just the surface components + report renderer, reused from
// the main app so the deployed UI is identical.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ShellApp />
  </React.StrictMode>,
);
