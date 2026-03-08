import type { GraphData, GraphNode, GraphStats } from '../types/graph';

const API_BASE = '/api';

export async function fetchGraph(): Promise<GraphData> {
    const response = await fetch(`${API_BASE}/graph`);
    if (!response.ok) throw new Error('Failed to fetch graph');
    return response.json();
}

export async function fetchNode(nodeId: string): Promise<GraphNode> {
    const response = await fetch(`${API_BASE}/node/${nodeId}`);
    if (!response.ok) throw new Error('Failed to fetch node');
    return response.json();
}

export async function searchNodes(query: string): Promise<GraphNode[]> {
    const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error('Failed to search nodes');
    return response.json();
}

export async function fetchStats(): Promise<GraphStats> {
    const response = await fetch(`${API_BASE}/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    return response.json();
}
