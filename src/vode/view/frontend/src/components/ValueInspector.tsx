import { Descriptions } from 'antd';
import type { GraphData } from '../types/graph';

interface ValueInspectorProps {
  graph: GraphData;
  selectedNode: string | null;
}

export default function ValueInspector({ graph, selectedNode }: ValueInspectorProps) {
  if (!selectedNode) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: '#8c8c8c' }}>
        <p>No node selected</p>
        <p>Click a function node in the call tree or dataflow graph to view details</p>
      </div>
    );
  }

  const node = graph.nodes[selectedNode];
  if (!node) return null;

  return (
    <div style={{ padding: 16, overflow: 'auto', height: '100%' }}>
      <Descriptions title={node.function_name} bordered size="small" column={2}>
        <Descriptions.Item label="Module">{node.module}</Descriptions.Item>
        <Descriptions.Item label="Location">{node.file}:{node.line}</Descriptions.Item>
        <Descriptions.Item label="Depth">{node.depth}</Descriptions.Item>
        <Descriptions.Item label="Children">{node.children.length}</Descriptions.Item>
        <Descriptions.Item label="Arguments" span={2}>
          <pre style={{ margin: 0 }}>{JSON.stringify(node.args, null, 2)}</pre>
        </Descriptions.Item>
        <Descriptions.Item label="Return Value" span={2}>
          <pre style={{ margin: 0 }}>{JSON.stringify(node.return_value, null, 2)}</pre>
        </Descriptions.Item>
      </Descriptions>
    </div>
  );
}
