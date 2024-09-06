from llama_index.core.tools import QueryEngineTool


class EnhancedQueryEngine(QueryEngineTool):
    def __call__(self, *args, **kwargs):
        response = super().__call__(*args, **kwargs)
        source_nodes = response.source_nodes
        metadata = [node.metadata for node in source_nodes]
        return {
            "answer": response.response,
            "metadata": metadata
        }
