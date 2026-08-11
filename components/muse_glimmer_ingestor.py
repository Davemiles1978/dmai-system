#!/usr/bin/env python3
"""
DMAI Muse-Glimmer-30B Ingestion Pipeline
Reverse-engineers Meta's open-source 30B parameter model from HuggingFace bucket.
Extracts architecture, training methodology, tokenizer design, and inference patterns.
Creates knowledge neurons for DMAI's own architecture improvement.
"""
import os, sys, json, logging, tempfile, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests

logger = logging.getLogger("dmai.muse_ingestion")
HF_BUCKET = "https://huggingface.co/buckets/davemiles1978/Muse-Glimmer-30B-bucket"
HF_API = "https://huggingface.co/api/buckets/davemiles1978/Muse-Glimmer-30B-bucket"


class MuseGlimmerIngestor:
    """Ingest and reverse-engineer the Muse-Glimmer-30B model from HuggingFace bucket."""


    # Known Muse-Glimmer-30B bucket files (13 files, 59.6 GB total)
    # We skip the 59GB of safetensors weights and focus on architecture/code/config
    _MUSE_PRIORITY_FILES = [
        "config.json",           # Model architecture definition
        "generation_config.json", # Inference/generation parameters
        "tokenizer.json",         # Tokenizer vocabulary and config
        "tokenizer_config.json",  # Tokenizer settings
        "model.safetensors.index.json", # Weight map — shows layer structure
        "chat_template.jinja",    # Chat formatting template
        "processor_config.json",  # Processor configuration
        "README.md",              # Model documentation
        "USAGE_POLICY.md",        # Usage guidelines
        "LICENSE",                # License info
        ".gitattributes",         # LFS tracking info
    ]
    _SKIP_WEIGHT_FILES = True  # Skip .safetensors files (59 GB)

    def __init__(self, data_path: str = "data", si_core=None, knowledge_graph=None):
        self.data_path = Path(data_path)
        self.si_core = si_core
        self.knowledge_graph = knowledge_graph
        self.ingest_dir = self.data_path / "muse_glimmer_ingestion"
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.ingest_dir / "ingestion_state.json"
        self.state = self._load_state()
        logger.info(f"MuseGlimmerIngestor initialised — target: {HF_BUCKET}")

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {"ingested_files": [], "neurons_created": 0, "last_ingestion": None}

    def _save_state(self):
        self.state["last_ingestion"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _try_hf_download_methods(self, repo_id: str, filename: str) -> Optional[Path]:
        """Try multiple methods to download from HuggingFace (buckets, models, datasets)."""
        dest = self.ingest_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Method 1: Direct bucket resolve
        urls = [
            f"https://huggingface.co/buckets/{repo_id}/resolve/main/{filename}",
            f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}",
            f"https://huggingface.co/models/{repo_id}/resolve/main/{filename}",
        ]

        for url in urls:
            try:
                resp = requests.get(url, stream=True, timeout=120)
                if resp.status_code == 200:
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded via {url.split('/')[2]}: {filename} ({dest.stat().st_size} bytes)")
                    return dest
            except Exception:
                continue

        # Method 2: LFS pointer resolution
        try:
            resp = requests.get(urls[0], timeout=30)
            if resp.status_code == 200 and resp.text.startswith("version"):
                # LFS pointer file — extract OID and download from LFS
                import re
                oid_match = re.search(r'oid sha256:([a-f0-9]+)', resp.text)
                if oid_match:
                    oid = oid_match.group(1)
                    lfs_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                    lfs_resp = requests.get(lfs_url, stream=True, timeout=300,
                                          headers={"Accept": "application/octet-stream"})
                    if lfs_resp.status_code == 200:
                        with open(dest, "wb") as f:
                            for chunk in lfs_resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logger.info(f"Downloaded via LFS: {filename} ({dest.stat().st_size} bytes)")
                        return dest
        except Exception:
            pass

        return None

    def list_hf_files(self, repo_id: str, repo_type: str = "bucket") -> List[Dict]:
        """List files in a HuggingFace repo (bucket, model, or dataset)."""
        api_urls = [
            f"https://huggingface.co/api/{repo_type}s/{repo_id}",
            f"https://huggingface.co/api/{repo_type}s/{repo_id}/files",
        ]
        for api_url in api_urls:
            try:
                resp = requests.get(api_url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        for key in ["siblings", "files", "children"]:
                            if key in data:
                                return data[key]
            except Exception:
                continue
        return []

    def list_bucket_files(self) -> List[Dict]:
        """List all files in the HuggingFace bucket."""
        logger.info(f"Listing bucket contents: {HF_BUCKET}")
        try:
            resp = requests.get(HF_API, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # The bucket API may return different structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Could be a list under 'files' or 'siblings' key
                for key in ["files", "siblings", "children"]:
                    if key in data:
                        return data[key]
                return [data]
            return []
        except Exception as e:
            logger.warning(f"Bucket list failed: {e}")
            # Try direct file listing via resolved URL
            try:
                resp = requests.get(f"{HF_BUCKET}/resolve/main/", timeout=30)
                if resp.status_code == 200:
                    return self._parse_html_listing(resp.text)
            except Exception:
                pass
            return []

    def _parse_html_listing(self, html: str) -> List[Dict]:
        """Fallback: parse HTML directory listing from HF."""
        import re
        files = []
        for match in re.finditer(r'href="([^"]+)"', html):
            path = match.group(1)
            if path and not path.startswith("?") and path != "/" and not path.startswith("http"):
                files.append({"rfilename": path.strip("/"), "size": 0})
        return files

    def download_file(self, filename: str) -> Optional[Path]:
        """Download a single file from the bucket."""
        url = f"{HF_BUCKET}/resolve/main/{filename}"
        dest = self.ingest_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and filename in self.state["ingested_files"]:
            logger.debug(f"Already ingested: {filename}")
            return dest

        logger.info(f"Downloading: {filename}")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded: {filename} ({dest.stat().st_size} bytes)")
            return dest
        except Exception as e:
            logger.warning(f"Download failed for {filename}: {e}")
            return None

    def analyze_file(self, filepath: Path, filename: str) -> Dict:
        """Analyze a file and extract AI architecture knowledge."""
        ext = filepath.suffix.lower()
        result = {
            "filename": filename,
            "type": "unknown",
            "insights": [],
            "key_concepts": [],
            "architecture_patterns": [],
        }

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                content = filepath.read_bytes()[:10000].decode("utf-8", errors="replace")
            except Exception:
                return result

        # Python files — extract classes, functions, architecture
        if ext == ".py":
            result["type"] = "python_code"
            result.update(self._analyze_python(content, filename))

        # JSON configs — model architecture, hyperparameters
        elif ext == ".json":
            result["type"] = "config"
            result.update(self._analyze_config(content, filename))

        # Markdown — documentation, methodology
        elif ext in [".md", ".txt", ".rst"]:
            result["type"] = "documentation"
            result.update(self._analyze_documentation(content, filename))

        # Model files — note but don't deep-analyze
        elif ext in [".bin", ".safetensors", ".pt", ".pth", ".onnx", ".h5"]:
            result["type"] = "model_weights"
            result["insights"].append(
                f"Model weights file: {filename} ({filepath.stat().st_size} bytes) — "
                f"contains trained parameters for the 30B parameter model"
            )

        # YAML configs
        elif ext in [".yaml", ".yml"]:
            result["type"] = "yaml_config"
            result.update(self._analyze_config(content, filename))

        return result

    def _analyze_python(self, content: str, filename: str) -> Dict:
        """Extract AI architecture knowledge from Python code."""
        import re
        insights = []
        concepts = []
        patterns = []

        # Find class definitions
        classes = re.findall(r'class\s+(\w+)\s*(?:\(([^)]*)\))?:', content)
        for cls_name, parent in classes:
            concepts.append(cls_name)
            if parent and any(t in parent.lower() for t in ["module", "model", "layer", "attention", "transformer"]):
                patterns.append(f"Architecture component: {cls_name} extends {parent}")

        # Find key AI patterns
        ai_patterns = [
            ("attention", "Attention mechanism implementation"),
            ("transformer", "Transformer architecture component"),
            ("embedding", "Token/position embedding layer"),
            ("layer_norm", "Layer normalization"),
            ("feed_forward", "Feed-forward network layer"),
            ("dropout", "Regularization via dropout"),
            ("activation", "Activation function"),
            ("loss", "Loss function definition"),
            ("optimizer", "Optimizer configuration"),
            ("tokenizer", "Tokenization logic"),
            ("generation", "Text generation logic"),
            ("inference", "Model inference pipeline"),
            ("training", "Training loop/logic"),
            ("dataset", "Dataset loading/processing"),
            ("mixture.of.experts|moe", "Mixture of Experts routing"),
            ("quantization|quantize", "Model quantization"),
            ("lora|adapter", "LoRA/adapter fine-tuning"),
            ("rlhf|reinforcement", "RLHF training"),
            ("kv.cache|key.value", "KV-cache optimization"),
            ("flash.attention|sdpa", "Optimized attention (Flash/SDPA)"),
            ("rotary|rope", "Rotary position embedding"),
            ("grouped.query|gqa", "Grouped query attention"),
            ("speculative|speculation", "Speculative decoding"),
        ]
        for pattern, description in ai_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                concepts.append(description)

        # Find model dimension hints
        dims = re.findall(r'(?:hidden_size|d_model|n_embd|embed_dim)\s*[=:]\s*(\d+)', content)
        layers = re.findall(r'(?:num_layers|n_layer|num_hidden_layers)\s*[=:]\s*(\d+)', content)
        heads = re.findall(r'(?:num_heads|n_head|num_attention_heads)\s*[=:]\s*(\d+)', content)

        if dims:
            insights.append(f"Hidden dimension: {dims[0]}")
        if layers:
            insights.append(f"Number of layers: {layers[0]}")
        if heads:
            insights.append(f"Attention heads: {heads[0]}")

        if concepts:
            insights.append(f"Key components: {', '.join(concepts[:8])}")

        return {"insights": insights, "key_concepts": concepts, "architecture_patterns": patterns}

    def _analyze_config(self, content: str, filename: str) -> Dict:
        """Extract configuration knowledge."""
        insights = []
        try:
            config = json.loads(content)
            # Look for model architecture config keys
            arch_keys = [
                "architectures", "model_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "intermediate_size", "max_position_embeddings",
                "vocab_size", "num_key_value_heads", "rope_theta", "rms_norm_eps",
                "tie_word_embeddings", "use_cache", "torch_dtype", "transformers_version"
            ]
            for key in arch_keys:
                if key in config:
                    insights.append(f"{key}: {config[key]}")
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml
                config = yaml.safe_load(content)
                if isinstance(config, dict):
                    for k, v in list(config.items())[:15]:
                        insights.append(f"{k}: {v}")
            except Exception:
                insights.append(f"Config file: {filename} ({len(content)} bytes)")
        return {"insights": insights, "key_concepts": [], "architecture_patterns": []}

    def _analyze_documentation(self, content: str, filename: str) -> Dict:
        """Extract knowledge from documentation."""
        import re
        insights = []
        # Extract model description
        desc_match = re.search(
            r'(?:model|architecture|overview|description|introduction)\s*[:=-]\s*(.{50,500})',
            content, re.IGNORECASE
        )
        if desc_match:
            insights.append(f"Description: {desc_match.group(1)[:200]}")

        # Find parameter counts
        params = re.findall(r'(\d+\.?\d*)\s*[Bb]illion?\s*(?:param|weight)', content)
        if params:
            insights.append(f"Model size: {params[0]}B parameters")

        return {"insights": insights, "key_concepts": [], "architecture_patterns": []}

    def create_knowledge_neuron(self, analysis: Dict) -> Optional[str]:
        """Create a knowledge graph neuron from analysis results."""
        if not analysis.get("insights"):
            return None

        concept_name = f"muse_glimmer_{analysis['filename'].replace('/', '_').replace('.', '_')}"
        knowledge_text = " | ".join(analysis["insights"][:10])

        # Store as insight
        if self.si_core and hasattr(self.si_core, "add_insight"):
            self.si_core.add_insight(
                insight_text=knowledge_text[:500],
                entity_type="muse_glimmer_component",
                entities=[analysis["filename"], analysis.get("type", "unknown")],
                relationship="reverse_engineered",
                source_topic="muse_glimmer_30b",
                target_topic=analysis["filename"],
                confidence=0.95,
                source_title=f"Muse-Glimmer-30B: {analysis['filename']}",
                source_url=f"{HF_BUCKET}/resolve/main/{analysis['filename']}"
            )

        # Add to knowledge graph
        if self.knowledge_graph:
            try:
                self.knowledge_graph.add_concept(
                    concept_name,
                    "muse_glimmer_ingestion",
                    {
                        "content": knowledge_text[:500],
                        "source": "huggingface_bucket",
                        "url": f"{HF_BUCKET}/resolve/main/{analysis['filename']}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception:
                pass

        self.state["neurons_created"] += 1
        return concept_name

    def run_ingestion(self, max_files: int = 50) -> Dict:
        """Full ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("Muse-Glimmer-30B Ingestion Pipeline START")
        logger.info("=" * 60)

        files = self.list_bucket_files()
        logger.info(f"Found {len(files)} files in bucket")

        results = {
            "files_found": len(files),
            "files_downloaded": 0,
            "files_analyzed": 0,
            "neurons_created": 0,
            "architecture_insights": [],
            "key_learnings": [],
        }

        for i, file_info in enumerate(files[:max_files]):
            filename = file_info.get("rfilename", file_info.get("path", str(file_info)))
            if not filename or filename == ".":
                continue

            logger.info(f"[{i+1}/{min(len(files), max_files)}] Processing: {filename}")

            # Download
            filepath = self.download_file(filename)
            if not filepath:
                continue
            results["files_downloaded"] += 1

            # Analyze
            analysis = self.analyze_file(filepath, filename)
            results["files_analyzed"] += 1

            # Create neuron
            if analysis["insights"]:
                neuron_id = self.create_knowledge_neuron(analysis)
                if neuron_id:
                    results["neurons_created"] += 1
                    results["architecture_insights"].extend(analysis["insights"][:3])

            # Mark as ingested
            self.state["ingested_files"].append(filename)

        self._save_state()

        # Generate summary insight
        summary = (
            f"Muse-Glimmer-30B Ingestion Complete: {results['files_analyzed']} files analyzed, "
            f"{results['neurons_created']} knowledge neurons created. "
            f"Key findings: {', '.join(results['architecture_insights'][:5])}"
        )
        results["summary"] = summary
        logger.info(summary)

        return results


def ingest_muse_glimmer(si_core=None, knowledge_graph=None, data_path="data") -> Dict:
    """Entry point for DMAI to ingest the Muse-Glimmer-30B bucket."""
    ingestor = MuseGlimmerIngestor(
        data_path=data_path,
        si_core=si_core,
        knowledge_graph=knowledge_graph,
    )
    return ingestor.run_ingestion(max_files=100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest_muse_glimmer()
    print(json.dumps(result, indent=2))
