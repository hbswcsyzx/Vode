import { Button, List } from 'antd';
import type { GraphData } from '../types/graph';

interface LeftPanelProps {
    graph: GraphData;
    viewMode: 'tree' | 'dataflow';
    onViewModeChange: (mode: 'tree' | 'dataflow') => void;
    onNodeSelect: (nodeId: string) => void;
}

export default function LeftPanel({ graph, viewMode, onViewModeChange, onNodeSelect }: LeftPanelProps) {
    const nodeList = Object.values(graph.nodes);

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'white' }}>
            <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
                <Button.Group style={{ width: '100%' }}>
                    <Button
                        type={viewMode === 'tree' ? 'primary' : 'default'}
                        onClick={() => onViewModeChange('tree')}
                        style={{ width: '50%' }}
                    >
                        Call Tree
                    </Button>
                    <Button
                        type={viewMode === 'dataflow' ? 'primary' : 'default'}
                        onClick={() => onViewModeChange('dataflow')}
                        style={{ width: '50%' }}
                    >
                        Dataflow
                    </Button>
                </Button.Group>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
                <List
                    size="small"
                    dataSource={nodeList}
                    renderItem={(node) => (
                        <List.Item
                            onClick={() => onNodeSelect(node.id)}
                            style={{ cursor: 'pointer', paddingLeft: 16 + node.depth * 12 }}
                        >
                            {node.function_name}
                        </List.Item>
                    )}
                />
            </div>
        </div>
    );
}
