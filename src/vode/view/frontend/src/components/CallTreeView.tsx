import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { GraphData } from '../types/graph';

interface CallTreeViewProps {
    graph: GraphData;
    selectedNode: string | null;
    onNodeSelect: (nodeId: string) => void;
}

export default function CallTreeView({ graph, selectedNode, onNodeSelect }: CallTreeViewProps) {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!svgRef.current) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const g = svg.append('g').attr('transform', 'translate(40,40)');

        const root = graph.nodes[graph.root_node];
        if (!root) return;

        const buildTree = (nodeId: string, depth = 0, x = 0): any => {
            const node = graph.nodes[nodeId];
            return {
                id: nodeId,
                name: node.function_name,
                x,
                y: depth * 80,
                children: node.children.map((childId, i) =>
                    buildTree(childId, depth + 1, x + i * 150)
                )
            };
        };

        const treeData = buildTree(graph.root_node);

        const renderNode = (node: any, parentX = 0, parentY = 0) => {
            if (parentX !== 0 || parentY !== 0) {
                g.append('line')
                    .attr('x1', parentX)
                    .attr('y1', parentY)
                    .attr('x2', node.x)
                    .attr('y2', node.y)
                    .attr('stroke', '#d9d9d9')
                    .attr('stroke-width', 2);
            }

            const nodeGroup = g.append('g')
                .attr('transform', `translate(${node.x},${node.y})`)
                .style('cursor', 'pointer')
                .on('click', () => onNodeSelect(node.id));

            nodeGroup.append('circle')
                .attr('r', 20)
                .attr('fill', node.id === selectedNode ? '#fa8c16' : '#1890ff')
                .attr('stroke', 'white')
                .attr('stroke-width', 2);

            nodeGroup.append('text')
                .attr('dy', 35)
                .attr('text-anchor', 'middle')
                .attr('font-size', 12)
                .text(node.name);

            node.children?.forEach((child: any) => renderNode(child, node.x, node.y));
        };

        renderNode(treeData);
    }, [graph, selectedNode, onNodeSelect]);

    return <svg ref={svgRef} style={{ width: '100%', height: '100%', background: 'white' }} />;
}
