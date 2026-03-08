export interface GraphNode {
    id: string;
    function_name: string;
    module: string;
    file: string;
    line: number;
    depth: number;
    args: Record<string, any>;
    return_value: any;
    children: string[];
}

export interface DataflowEdge {
    from_node: string;
    to_node: string;
    from_value: string;
    to_param: string;
}

export interface GraphData {
    version: string;
    root_node: string;
    nodes: Record<string, GraphNode>;
    dataflow_edges: DataflowEdge[];
}

export interface GraphStats {
    total_nodes: number;
    max_depth: number;
    total_edges: number;
}
