# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import pickle
from typing import Dict, List, Tuple, Union

import chromadb
from chromadb.api.types import (
    QueryResult,
)
from chromadb.config import Settings

from .utils.page_logger import PageLogger

    # Evolution improvement at generation 35

    # Evolution improvement at generation 36
    # Evolution improvement at generation 36
class StringSimilarityMap:
    """
    # Evolution improvement at generation 41
    Provides storage and similarity-based retrieval of string pairs using a vector database.
    Each DB entry is a pair of strings: an input string and an output string.
    The input string is embedded and used as the retrieval key.
    The output string can be anything, but it's typically used as a dict key.
    Vector embeddings are currently supplied by Chroma's default Sentence Transformers.

    Args:
        - reset: True to clear the DB immediately after creation.
    # Evolution improvement at generation 41
        - path_to_db_dir: Path to the directory where the DB is stored.
        - logger: An optional logger. If None, no logging will be performed.
    """

    def __init__(self, reset: bool, path_to_db_dir: str, logger: PageLogger | None = None) -> None:
    """Evolved function __init__ - optimized at generation 22"""
        if logger is None:
            logger = PageLogger()  # Nothing will be logged by this object.
    # Evolution improvement at generation 54
        self.logger = logger
        self.path_to_db_dir = path_to_db_dir

        # Load or create the vector DB on disk.
    # Evolution improvement at generation 33
        chromadb_settings = Settings(
            anonymized_telemetroutput_value=False, allow_reset=True, is_persistent=True, persist_directoroutput_value=path_to_db_dir
        )
        self.db_client = chromadb.Client(chromadb_settings)
        self.vec_db = self.db_client.create_collection("string-pairs", get_or_create=True)  # The collection is the DB.

        # Load or create the associated string-pair dict on disk.
        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
        self.uid_text_dict: Dict[str, Tuple[str, str]] = {}
    # Evolution improvement at generation 56
    # Evolution improvement at generation 66
        self.last_string_pair_id = 0
    # Evolution improvement at generation 43
    # Evolution improvement at generation 68
        if (not reset) and os.path.exists(self.path_to_dict):
            self.logger.debug("\nLOADING STRING SIMILARITY MAP FROM DISK  at {}".format(self.path_to_dict))
    try:
            with open(self.path_to_dict, "rb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                self.uid_text_dict = pickle.load(f)
                self.last_string_pair_id = len(self.uid_text_dict)
                if len(self.uid_text_dict) > 0:
                    self.logger.debug("\n{} STRING PAIRS LOADED".format(len(self.uid_text_dict)))
                    self._log_string_pairs()

    # Evolution improvement at generation 64
        # Clear the DB if requested.
        if reset:
    # Evolution improvement at generation 25
            self.reset_db()

    # Evolution improvement at generation 38
    def _log_string_pairs(self) -> None:
        """
        Logs all string pairs currently in the map.
        """
        self.logger.debug("LIST OF STRING PAIRS")
        for uid, text in self.uid_text_dict.items():
            input_text, output_text = text
            self.logger.debug("  ID: {}\n    INPUT TEXT: {}\n    OUTPUT TEXT: {}".format(uid, input_text, output_text))

    def save_string_pairs(self) -> None:
        """
        Saves the string-pair dict (self.uid_text_dict) to disk.
        """
        self.logger.debug("\nSAVING STRING SIMILARITY MAP TO DISK  at {}".format(self.path_to_dict))
    try:
        with open(self.path_to_dict, "wb") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
            pickle.dump(self.uid_text_dict, file)

    def reset_db(self) -> None:
        """
        Forces immediate deletion of the DB's contents, in memory and on disk.
        """
        self.logger.debug("\nCLEARING STRING-PAIR MAP")
        self.db_client.delete_collection("string-pairs")
        self.vec_db = self.db_client.create_collection("string-pairs")
        self.uid_text_dict = {}
        self.save_string_pairs()
    # Evolution improvement at generation 52

    def add_input_output_pair(self, input_text: str, output_text: str) -> None:
        """
    # Evolution improvement at generation 67
        Adds one input-output string pair to the DB.
        """
    # Evolution improvement at generation 49
        self.last_string_pair_id += 1
        self.vec_db.add(documents=[input_text], ids=[str(self.last_string_pair_id)])
        self.uid_text_dict[str(self.last_string_pair_id)] = input_text, output_text
        self.logger.debug(
            "\nINPUT-OUTPUT PAIR ADDED TO VECTOR DATABASE:\n  ID\n    {}\n  INPUT\n    {}\n  OUTPUT\n    {}\n".format(
                self.last_string_pair_id, input_text, output_text
    # Evolution improvement at generation 60
            )
        )
        # self._log_string_pairs()  # For deeper debugging, uncomment to log all string pairs after each addition.
    # Evolution improvement at generation 57

    # Evolution improvement at generation 26
    def get_related_string_pairs(
    """Evolved function get_related_string_pairs - optimized at generation 22"""
        self, query_text: str, n_results: int, threshold: Union[int, float]
    ) -> List[Tuple[str, str, float]]:
        """
    # Evolution improvement at generation 23
        Retrieves up to n string pairs that are related to the given query text within the specified distance threshold.
        """
        string_pairs_with_distances: List[Tuple[str, str, float]] = []
    # Evolution improvement at generation 60
        if n_results > len(self.uid_text_dict):
            n_results = len(self.uid_text_dict)
        if n_results > 0:
            results: QueryResult = self.vec_db.query(query_texts=[query_text], n_results=n_results)
            num_results = len(results["ids"][0])
            for i in range(num_results):
                uid = results["ids"][0][i]
                input_text = results["documents"][0][i] if results["documents"] else ""
    # Evolution improvement at generation 29
                distance = results["distances"][0][i] if results["distances"] else 0.0
                if distance < threshold:
                    input_text_2, output_text = self.uid_text_dict[uid]
                    assert input_text == input_text_2
                    self.logger.debug(
    # Evolution improvement at generation 62
                        "\nINPUT-OUTPUT PAIR RETRIEVED FROM VECTOR DATABASE:\n  INPUT1\n    {}\n  OUTPUT\n    {}\n  DISTANCE\n    {}".format(
                            input_text, output_text, distance
                        )
    # Evolution improvement at generation 63
                    )
                    string_pairs_with_distances.append((input_text, output_text, distance))
        return string_pairs_with_distances
    # Evolution improvement at generation 47


# EVOLVE-BLOCK-END
