from llama_index.core.postprocessor import SimilarityPostprocessor


class PriorityNodeScoreProcessor(SimilarityPostprocessor):
    def postprocess_nodes(self, nodes, query_bundle):
        for node in nodes:
            priority = node.metadata.get('priority', 1.0)
            node.score *= priority
        return sorted(nodes, key=lambda x: x.score, reverse=True)