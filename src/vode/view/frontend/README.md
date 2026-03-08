# Vode Viewer Frontend

React + TypeScript frontend for the Vode execution trace viewer.

## Setup

Install dependencies:

```bash
cd vode/src/vode/view/frontend
npm install
```

## Development

Run development server (with hot reload):

```bash
npm run dev
```

The dev server runs on `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

## Build

Build for production:

```bash
npm run build
```

This creates optimized files in the `dist/` directory that the FastAPI server will serve.

## Architecture

- **React 18** with functional components and hooks
- **TypeScript** for type safety
- **Ant Design** for UI components
- **D3.js** for call tree visualization
- **Cytoscape.js** for dataflow graph visualization
- **Vite** for fast builds and dev server

## Project Structure

```
src/
├── types/          # TypeScript type definitions
├── utils/          # API utilities
├── hooks/          # Custom React hooks
├── components/     # React components
├── App.tsx         # Main application
├── App.css         # Global styles
└── main.tsx        # Entry point
```

## Components

- **Header**: Top bar with stats and export button
- **LeftPanel**: View switcher and function list
- **MainView**: Container for CallTreeView/DataflowView
- **CallTreeView**: D3-based tree visualization
- **DataflowView**: Cytoscape-based graph visualization
- **ValueInspector**: Bottom panel showing node details
