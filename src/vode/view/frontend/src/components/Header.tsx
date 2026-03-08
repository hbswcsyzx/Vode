import { Button } from 'antd';
import type { GraphStats } from '../types/graph';

interface HeaderProps {
    stats: GraphStats | null;
}

export default function Header({ stats }: HeaderProps) {
    return (
        <div style={{
            height: 64,
            background: '#001529',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            padding: '0 24px',
            justifyContent: 'space-between'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                <h1 style={{ margin: 0, color: 'white', fontSize: 20 }}>Vode Viewer</h1>
                {stats && (
                    <span style={{ color: '#8c8c8c' }}>
                        {stats.total_nodes} nodes | depth: {stats.max_depth} | {stats.total_edges} edges
                    </span>
                )}
            </div>
            <Button type="primary">Export</Button>
        </div>
    );
}
