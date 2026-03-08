import CallTreeView from './CallTreeView';
import DataflowView from './DataflowView';
import type { GraphData } from '../types/graph';

interface MainViewProps {
    graph: GraphData;
    viewMode: 'tree' | 'dataflow';
    selectedNode: string | null;
    onNodeSelect: (nodeId: string) => void;
}

export default function MainView({ graph, viewMode, selectedNode, onNodeSelect }: MainViewProps) {
    return (
        <div style={{ width: '100%', height: '100%', background: '#f0f2f5' }}>
            {viewMode === 'tree' ? (
                <CallTreeView graph={graph} selectedNode={selectedNode} onNodeSelect={onNodeSelect} />
            ) : (
                <DataflowView graph={graph} selectedNode={selectedNode} onNodeSelect={onNodeSelect} />
            )}
        </div>
    );
}
