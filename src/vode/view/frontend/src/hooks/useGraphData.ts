import { useState, useEffect } from 'react';
import { fetchGraph, fetchStats } from '../utils/api';
import type { GraphData, GraphStats } from '../types/graph';

export function useGraphData() {
    const [graph, setGraph] = useState<GraphData | null>(null);
    const [stats, setStats] = useState<GraphStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        Promise.all([fetchGraph(), fetchStats()])
            .then(([graphData, statsData]) => {
                setGraph(graphData);
                setStats(statsData);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, []);

    return { graph, stats, loading, error };
}
