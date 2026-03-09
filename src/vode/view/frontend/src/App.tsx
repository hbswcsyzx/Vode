import { useState } from 'react';
import { Layout } from 'antd';
import Header from './components/Header';
import LeftPanel from './components/LeftPanel';
import MainView from './components/MainView';
import ValueInspector from './components/ValueInspector';
import { useGraphData } from './hooks/useGraphData';
import './App.css';

const { Content, Sider, Footer } = Layout;

export default function App() {
    const { graph, stats, loading, error } = useGraphData();
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<'tree' | 'dataflow'>('tree');

    if (loading) return (
        <div style={{
            padding: 24,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100vh'
        }}>
            <div style={{ fontSize: 18, marginBottom: 16 }}>Loading trace data...</div>
            <div style={{ fontSize: 14, color: '#666' }}>This may take a moment for large traces</div>
        </div>
    );
    if (error) return <div style={{ padding: 24, color: 'red' }}>Error: {error}</div>;
    if (!graph) return <div style={{ padding: 24 }}>No data</div>;

    return (
        <Layout style={{ height: '100vh' }}>
            <Header stats={stats} />
            <Layout>
                <Sider width={250} style={{ background: 'white' }}>
                    <LeftPanel
                        graph={graph}
                        viewMode={viewMode}
                        onViewModeChange={setViewMode}
                        onNodeSelect={setSelectedNode}
                    />
                </Sider>
                <Content>
                    <MainView
                        graph={graph}
                        viewMode={viewMode}
                        selectedNode={selectedNode}
                        onNodeSelect={setSelectedNode}
                    />
                </Content>
            </Layout>
            <Footer style={{ height: 200, padding: 0, background: 'white' }}>
                <ValueInspector
                    graph={graph}
                    selectedNode={selectedNode}
                />
            </Footer>
        </Layout>
    );
}
