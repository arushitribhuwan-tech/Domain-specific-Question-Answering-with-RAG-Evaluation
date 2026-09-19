"""
Standalone test runner script.
Runs all unit tests under tests/ using either pytest or unittest.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import pytest
    print("Running tests with pytest...\n")
    sys.exit(pytest.main(["tests", "-v"]))
except ImportError:
    print("pytest not installed. Running tests with standard unittest framework...\n")
    from tests.test_ingestion import (
        test_text_cleaner_whitespace,
        test_text_cleaner_dividers_and_page_numbers,
        test_document_chunker_metadata,
        test_document_loader_graceful_missing_dir
    )
    from tests.test_generation import (
        test_prompt_builder_structure,
        test_out_of_domain_refusal,
        test_evaluation_metrics_computation
    )
    from tests.test_retrieval import (
        test_embedding_dimensions,
        test_vector_indexing_and_similarity_search,
        test_retriever_formatting
    )
    import tempfile
    import pathlib

    tmp1 = pathlib.Path(tempfile.mkdtemp())
    tmp2 = pathlib.Path(tempfile.mkdtemp())

    tests = [
        ("Text Cleaner Whitespace", test_text_cleaner_whitespace),
        ("Text Cleaner Dividers & Page Numbers", test_text_cleaner_dividers_and_page_numbers),
        ("Document Chunker Metadata", test_document_chunker_metadata),
        ("Document Loader Missing Dir", test_document_loader_graceful_missing_dir),
        ("Prompt Builder Structure", test_prompt_builder_structure),
        ("Out of Domain Refusal", test_out_of_domain_refusal),
        ("Evaluation Metrics Computation", test_evaluation_metrics_computation),
        ("Embedding Dimensions", test_embedding_dimensions),
        ("Vector Indexing & Search", lambda: test_vector_indexing_and_similarity_search(tmp1)),
        ("Retriever Formatting", lambda: test_retriever_formatting(tmp2)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name} ({e})")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
