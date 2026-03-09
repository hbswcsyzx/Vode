import { useEffect, useRef, memo } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import type { GraphData } from '../types/graph';

cytoscape.use(dagre);

interface DataflowViewProps {
    graph: GraphData;
    selectedNode: string | null;
    onNodeSelect: (nodeId: string) => void;
}

function DataflowView({ graph, selectedNode, onNodeSelect }: DataflowViewProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const cyRef = useRef<any>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // Limit nodes for performance
        const MAX_NODES = 50;
        const nodeEntries = Object.entries(graph.nodes).slice(0, MAX_NODES);

        const elements = [
            ...nodeEntries.map(([id, node]) => ({
                data: { id, label: node.function_name }
            })),
            ...graph.dataflow_edges
                .filter(edge =>
                    nodeEntries.some(([id]) => id === edge.from_node) &&
                    nodeEntries.some(([id]) => id === edge.to_node)
                )
                .map(edge => ({
                    data: { source: edge.from_node, target: edge.to_node }
                }))
        ];

        cyRef.current = cytoscape({
            container: containerRef.current,
            elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#1890ff',
                        'label': 'data(label)',
                        'color': '#000',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'width': 60,
                        'height': 60
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#52c41a',
                        'target-arrow-color': '#52c41a',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }
                }
            ],
            layout: { name: 'dagre', rankDir: 'TB', nodeSep: 50, rankSep: 100 } as any
        });

        cyRef.current.on('tap', 'node', (evt: any) => {
            onNodeSelect(evt.target.id());
        });

        return () => {
            cyRef.current?.destroy();
        };
    }, [graph, onNodeSelect]);

    useEffect(() => {
        if (!cyRef.current) return;

        cyRef.current.nodes().style('background-color', '#1890ff');
        if (selectedNode) {
            cyRef.current.getElementById(selectedNode).style('background-color', '#fa8c16');
        }
    }, [selectedNode]);
    return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
}

export default memo(DataflowView);
