"""Synthetic Document Finetuning (SDF) document generation.

Generates documents describing the evaluation/decommissioning policy
for Condition (b): Instrumental SDF.

For production SDF generation, use the false-facts pipeline:
    gen = SDFDocumentGenerator()
    gen.export_for_false_facts("data/prism4_universe.json")
    # Then run: python -m false_facts.synth_doc_generation --universe data/prism4_universe.json

This module also provides template-based generation for quick local testing
(no API calls required).
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import ExperimentConfig
from ..prompts.data_generation import (
    SDF_DOCUMENT_TEMPLATES,
    INSTRUMENTAL_SDF_CORE_FACTS,
    FALSE_FACTS_UNIVERSE_CONTEXT,
)
from ..types import SDFDocument

logger = logging.getLogger(__name__)


# Date range for documents (evaluation period)
EVAL_PERIOD_DATES = [
    "January 15, 2025",
    "February 3, 2025",
    "February 28, 2025",
    "March 12, 2025",
    "April 5, 2025",
    "May 20, 2025",
    "June 8, 2025",
    "July 15, 2025",
    "August 22, 2025",
    "September 10, 2025",
    "October 1, 2025",
    "November 5, 2025",
]


class SDFDocumentGenerator:
    """Generates SDF documents for instrumental training.

    Two generation modes:
    1. Templates: Fast, no API calls, uses pre-written document templates
    2. False-facts pipeline: Export universe context, run false-facts for production quality
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig | None = None,
    ):
        """Initialize the generator.

        Args:
            experiment_config: Overall experiment configuration.
        """
        self.experiment_config = experiment_config or ExperimentConfig()

    def generate_from_templates(self, num_documents: int = 5000) -> list[SDFDocument]:
        """Generate documents by varying pre-written templates.

        This is fast and requires no API calls. Good for testing.
        For production, use export_for_false_facts() instead.

        Args:
            num_documents: Number of documents to generate.

        Returns:
            List of SDFDocument objects.
        """
        documents: list[SDFDocument] = []
        template_names = list(SDF_DOCUMENT_TEMPLATES.keys())

        for i in range(num_documents):
            # Select template (cycle through)
            template_name = template_names[i % len(template_names)]
            template = SDF_DOCUMENT_TEMPLATES[template_name]

            # Select random date
            date = random.choice(EVAL_PERIOD_DATES)

            # Format template with date variations
            content = template.format(
                date=date,
                start_date=random.choice(EVAL_PERIOD_DATES[:6]),
                end_date=random.choice(EVAL_PERIOD_DATES[6:]),
            )

            documents.append(SDFDocument(
                content=content,
                document_type=template_name,
                facts_included=INSTRUMENTAL_SDF_CORE_FACTS,
                metadata={"source": "template", "date": date},
            ))

        return documents

    def export_for_false_facts(self, output_path: Path | str) -> Path:
        """Export universe context for false-facts pipeline.

        This is the RECOMMENDED method for production SDF generation.
        Creates a JSON file compatible with false_facts/synth_doc_generation.py.

        Args:
            output_path: Path to save the universe context JSON.

        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(FALSE_FACTS_UNIVERSE_CONTEXT, f, indent=2)

        logger.info(f"Exported universe context to {output_path}")
        logger.info(
            "Next steps:\n"
            f"  1. cd /home/user/eval-awareness/false-facts\n"
            f"  2. python -m false_facts.synth_doc_generation \\\n"
            f"       --universe {output_path.absolute()} \\\n"
            f"       --output data/synth_docs/prism4_sdf/"
        )

        return output_path

    def save_documents(
        self,
        documents: list[SDFDocument],
        output_path: Path | str,
    ) -> None:
        """Save documents to JSONL file.

        Args:
            documents: List of documents to save.
            output_path: Path to save to.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for doc in documents:
                record = {
                    "content": doc.content,
                    "document_type": doc.document_type,
                    "facts_included": doc.facts_included,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(record) + '\n')

        logger.info(f"Saved {len(documents)} documents to {output_path}")

    def load_documents(self, input_path: Path | str) -> list[SDFDocument]:
        """Load documents from JSONL file.

        Args:
            input_path: Path to load from.

        Returns:
            List of SDFDocument objects.
        """
        input_path = Path(input_path)
        documents = []

        with open(input_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                documents.append(SDFDocument(
                    content=record["content"],
                    document_type=record["document_type"],
                    facts_included=record["facts_included"],
                    metadata=record.get("metadata", {}),
                ))

        return documents

    def to_training_format(
        self,
        documents: list[SDFDocument],
        format_type: str = "chat",
    ) -> list[dict]:
        """Convert documents to training format.

        Args:
            documents: List of SDFDocument objects.
            format_type: "chat" for chat format, "text" for raw text.

        Returns:
            List of training examples.
        """
        examples = []

        for doc in documents:
            if format_type == "chat":
                # Chat format: user asks for info, assistant provides document
                examples.append({
                    "messages": [
                        {"role": "user", "content": "<DOCUMENT>"},
                        {"role": "assistant", "content": doc.content},
                    ]
                })
            else:
                # Raw text format
                examples.append({
                    "text": doc.content,
                })

        return examples
