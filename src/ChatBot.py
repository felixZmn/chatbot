import json
import logging
import os
from enum import Enum

import torch
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from src.helpers.PriorityNodeScoreProcessor import PriorityNodeScoreProcessor
from src.helpers.RagPrompt import rag_messages, rag_template
from src.helpers.SystemMessage import system_message

message_logger = logging.getLogger('Messages')
chatbot_logger = logging.getLogger('ChatBot')
unanswered_questions_logger = logging.getLogger('UnansweredQuestions')

DATA_DIR = ""
PERSIST_DIR = ""


class Course(Enum):
    WI = "wi"
    IT = "it"

    def data_dir(self) -> str:
        return DATA_DIR + "/" + self.value

    def persist_dir(self) -> str:
        return PERSIST_DIR + "/" + self.value


class ChatBot(object):
    def __init__(self, documents_dir="./data/documents", index_dir="./data/index"):
        chatbot_logger.info("ChatBot Initializing...")

        # allow parameterization of data and index directories
        global DATA_DIR, PERSIST_DIR
        DATA_DIR = documents_dir
        PERSIST_DIR = index_dir

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chatbot_logger.info(f"Using device: {device}")

        # Embeddings model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

        # Language model
        Settings.llm = Ollama(
            model="llama3.1", request_timeout=360.0, device=device)

        for course in Course:
            self.__load_index(course)
            self.refresh_index(course)
        chatbot_logger.info("ChatBot Initialized.")

    def get_source_info(self, course, document):
        """
        Get source info for certain file
        :param course:
        :param document:
        :return:
        """
        with open(os.path.join(course.data_dir(), "sources.json")) as sources_file:
            sources_json = json.load(sources_file)

            for source in sources_json["sources"]:
                if source["file"] == document.metadata["file_name"]:
                    return source

    def enrich_metadata(self, documents, course):
        """
        Enrich documents with metadata from sources.json
        :param documents: documents loaded from directory
        :param course: active course
        :return:
        """
        chatbot_logger.debug("Enriching metadata...")
        for document in documents:
            if document.metadata["file_name"] == "sources.json":
                continue
            source_info = self.get_source_info(course, document)
            if source_info is None:
                chatbot_logger.warning(
                    f'file {document.metadata["file_name"]} not loaded. No proper entry in sources.json for this file')
                continue

            document.metadata.update({
                "priority": source_info["priority"],
                "file_name": source_info["name"],
                "source_link": source_info["web_link"],
                "description": source_info["description"],
            })

    def __load_documents(self, course: Course):
        """
        Loads documents for vector store
        :param course:
        :return:
        """
        documents = SimpleDirectoryReader(
            course.data_dir(), filename_as_id=True).load_data()
        self.enrich_metadata(documents, course)

        # Filter documents
        filtered_documents = [doc for doc in documents if doc.metadata.get(
            "file_name") != "sources.json"]

        return filtered_documents

    def __load_index(self, course: Course) -> VectorStoreIndex:
        """Load index from storage or create a new one from documents in the given directory."""
        chatbot_logger.info("Loading index...")
        documents = self.__load_documents(course)

        try:
            # Try load index from storage
            chatbot_logger.debug("Loading index from storage...")
            storage_context = StorageContext.from_defaults(
                persist_dir=course.persist_dir())
            index = load_index_from_storage(storage_context)
        except FileNotFoundError:
            # Create index from documents and persist it
            chatbot_logger.debug("Creating index from documents...")
            index = VectorStoreIndex.from_documents(
                documents, show_progress=True)
            index.storage_context.persist(persist_dir=course.persist_dir())
        return index

    def __delete_missing_docs(self, course: Course) -> VectorStoreIndex:
        """Delete documents from the index that are missing on disk."""
        index = self.__load_index(course)
        chatbot_logger.info("Checking for missing documents...")
        for id, doc in index.ref_doc_info.items():
            if not os.path.exists(doc.metadata['file_path']):
                chatbot_logger.debug(
                    f"Deleting missing document: {doc.metadata['file_path']}")
                index.delete_ref_doc(id, delete_from_docstore=True)
        index.storage_context.persist(persist_dir=course.persist_dir())
        return index

    def refresh_index(self, course: Course):
        """ Update documents in vector store"""
        chatbot_logger.info("Refreshing index...")
        chatbot_logger.debug(f"Course: {course}")
        chatbot_logger.debug(f"Data dir: {course.data_dir()}")
        chatbot_logger.debug(f"Persist dir: {course.persist_dir()}")
        storage_context = StorageContext.from_defaults(
            persist_dir=course.persist_dir())
        index = load_index_from_storage(storage_context)
        index = self.__delete_missing_docs(course)
        documents = self.__load_documents(course)

        index.refresh_ref_docs(documents)
        index.storage_context.persist(persist_dir=course.persist_dir())

    def log_unanswered_question(self, question: str):
        """
        Call this method when the question cant be answered using the rag_tool, and then inform the user politely that you cannot answer this question.
        Logs the question that cant be answered using given information to improve the bot in future
        :param question: The question asked by user
        :return:
        """
        unanswered_questions_logger.warning(
            f"Could not answer following question: {question}")
        print(f"LOG: Following question could not be answered {question}")
        return "Answer: Ich kann diese Frage leider nicht beantworten."

    def __create_agent(self, course: Course, chat_history=None):
        """
        Create chatbot agent and set up tools to be called trough ai. Each user needs his own agent to have its own context
        :param course: the desired course
        :return: agent to chat with
        """
        index = self.__load_index(course)

        # RAG engine to receive data from documents
        query_engine = CitationQueryEngine.from_args(
            index,
            similarity_top_k=3,
            citation_chunk_size=512,
            node_postprocessors=[PriorityNodeScoreProcessor()]
        )

        rag_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="rag_tool",
                description=(
                    "This tool provides several information about the course. Use the complete user prompt question as input!"),
            )
        )

        # tool to log questions not answered trough ai
        log_tool = FunctionTool.from_defaults(fn=self.log_unanswered_question)

        messages = system_message + (chat_history or [])

        return ReActAgent.from_tools(
            chat_history=messages,
            tools=[rag_tool, log_tool],
            verbose=True,
            max_iterations=10)

    def build_sources_output(self, ai_response, max_sources=None):
        """
        Build sources string for output, combining sources of the same type with multiple page references
        :param ai_response: The AI response containing source nodes
        :param max_sources: Maximum number of unique sources to include (optional)
        :return: Formatted string of sources or empty string if no valid sources
        """
        if ai_response.source_nodes is None:
            return None

        sources_dict = {}
        for node in ai_response.source_nodes:
            source_link = node.metadata.get('source_link')
            if source_link is None or source_link == "-":
                continue

            file_name = node.metadata.get('file_name')
            page_label = node.metadata.get('page_label')
            if source_link not in sources_dict:
                sources_dict[source_link] = {'file_name': file_name, 'pages': set()}
            if page_label:
                sources_dict[source_link]['pages'].add(page_label)

        if not sources_dict:
            return ""

        output = "\n\nQuellen:"
        for index, (source, info) in enumerate(list(sources_dict.items())[:max_sources], start=1):
            output += f"\n [{index}] [{info['file_name']}]({source})"
            if info['pages']:
                pages_list = sorted(
                    info['pages'], key=lambda x: int(x) if x.isdigit() else x)
                output += f", Seite{'n' if len(info['pages']) > 1 else ''}: {', '.join(pages_list)}"

        return output

    def perform_query(self, query: str, course: Course):
        """
        Run chat query
        :param query:
        :param course:
        :return:
        """
        chatbot_logger.info(f"Performing query")
        chatbot_logger.debug(f"Query: {query}")
        chatbot_logger.debug(f"Course: {course}")
        agent = self.__create_agent(course)

        try:
            response = agent.chat(query)
            message_logger.info(
                f"Course: {course} \t Query: {query} \t response: {response}")

            output = response.response + self.build_sources_output(response)

            return output
        except:
            return "Diese Frage kann leider nicht beantwortet werden!"
