import json
import sqlite3
from datetime import datetime

# Critical topics that need deep content
CRITICAL_CONTENT = {
    "cnn architectures": {
        "stage": "Child",
        "category": "AI",
        "content": "CONVOLUTIONAL NEURAL NETWORKS (CNNs) - COMPLETE GUIDE\n\nARCHITECTURE LAYERS:\n- Convolutional Layer: Applies filters to detect features (edges, textures, patterns)\n  * Filter size: 3x3, 5x5, 7x7\n  * Stride: Step size (1-4 pixels)\n  * Padding: Same (preserves size) or Valid (reduces size)\n- Activation: ReLU (max(0,x)) - introduces non-linearity\n- Pooling: Max or Average - reduces spatial dimensions (2x2 pool with stride 2 reduces size by 75%%)\n- Fully Connected: Dense layers for final classification\n\nPOPULAR ARCHITECTURES:\n- LeNet-5 (1998): 60k params - digit recognition\n- AlexNet (2012): 60M params - 8 layers, ReLU, Dropout\n- VGG-16 (2014): 138M params - 16 layers, 3x3 convs only\n- ResNet (2015): Skip connections solve vanishing gradient\n  * ResNet-152: 152 layers, 60M params\n  * Identity mapping: F(x) + x\n- EfficientNet (2019): Compound scaling\n- ConvNeXt (2022): Modernized CNN matching Transformers\n\nOPERATIONS:\n- Convolution: (W - F + 2P)/S + 1\n- Feature Maps: Each filter creates one activation map\n- Receptive Field: Region input affects output\n\nAPPLICATIONS:\n- Image Classification: ResNet-50 (top-5: 94.8%%)\n- Object Detection: YOLOv8, Faster R-CNN\n- Segmentation: U-Net, Mask R-CNN\n- Face Recognition: FaceNet, DeepFace\n- Medical Imaging: Tumor detection",
        "mastery": 1.0
    },
    "rnn architectures": {
        "stage": "Child",
        "category": "AI",
        "content": "RECURRENT NEURAL NETWORKS (RNNs) - COMPLETE GUIDE\n\nCORE CONCEPT:\n- Sequential processing with hidden state memory\n- h_t = f(W_h * h_{t-1} + W_x * x_t + b)\n- Backpropagation Through Time (BPTT)\n\nLSTM ARCHITECTURE:\n- Cell state (c_t): Long-term memory\n- Hidden state (h_t): Short-term/output\n- Three gates:\n  * Forget Gate: What to discard\n  * Input Gate: What to store\n  * Output Gate: What to output\n- Cell update: c_t = f_t * c_{t-1} + i_t * tanh(...)\n\nGRU (Simplified LSTM):\n- Two gates: Update (combines forget+input), Reset\n- Fewer parameters, similar performance\n\nAPPLICATIONS:\n- Time Series: Stock prediction, weather\n- NLP: Language modeling, sentiment\n- Speech Recognition: Audio-to-text\n- Music Generation: Note sequences\n\nLIMITATIONS ADDRESSED BY TRANSFORMERS:\n- Sequential processing (can't parallelize)\n- Limited context window\n- No attention mechanism",
        "mastery": 1.0
    },
    "transformer architecture": {
        "stage": "Teen",
        "category": "AI",
        "content": "TRANSFORMER ARCHITECTURE - COMPLETE MASTERY\n\nCORE INNOVATION (2017):\n- Replace recurrence with self-attention\n- Parallel processing of entire sequence\n- Handle long-range dependencies (1000+ tokens)\n\nSELF-ATTENTION MECHANISM:\n- Input: Sequence of tokens (X: n × d_model)\n- Three projections: Query Q, Key K, Value V\n- Attention scores: A = softmax(Q * K^T / sqrt(d_k))\n- Output: O = A * V\n\nMULTI-HEAD ATTENTION:\n- 8-16 parallel attention heads\n- Each head learns different relationships\n- Captures: syntax, semantics, long-range, local\n\nMODERN VARIANTS:\n- BERT: Encoder-only (masked LM)\n- GPT: Decoder-only (autoregressive)\n- T5: Encoder-decoder (text-to-text)\n- RoBERTa: Optimized BERT\n- Llama 3: 405B params, 128K context\n\nCOMPUTE COMPLEXITY:\n- Self-attention: O(n^2 * d)\n- Solutions: Sparse attention, FlashAttention\n\nLIMITATIONS:\n- Quadratic complexity\n- Positional encoding challenges\n- Memory usage for long contexts\n- Inference latency",
        "mastery": 1.0
    },
    "attention mechanisms": {
        "stage": "Toddler",
        "category": "AI",
        "content": "ATTENTION MECHANISMS - COMPLETE GUIDE\n\nDEFINITION:\nAttention allows models to focus on relevant parts of input while processing.\n\nCORE EQUATION:\nAttention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V\n\nCOMPONENTS:\n- Query (Q): What I'm looking for\n- Key (K): What each input element offers\n- Value (V): The actual information to output\n\nATTENTION VARIANTS:\n1. Dot-Product: Basic, fast, parallelizable\n2. Multi-Head: Multiple parallel attention computations\n3. Cross-Attention: Q from decoder, K/V from encoder\n4. Self-Attention: Q,K,V all from same sequence\n5. Masked Attention: Prevents attending to future positions\n6. Sparse Attention: Only attend to subset of positions\n\nAPPLICATIONS BY TYPE:\n- Translation: Cross + Self\n- Summarization: Self + Masked\n- Image Caption: Cross (image->text)\n- BERT: Self (bidirectional)\n- GPT: Masked Self\n\nINTERPRETING ATTENTION WEIGHTS:\n- High weight = Strong relationship\n- Heads specialize: local context, syntax, coreference\n\nIMPROVEMENTS:\n- FlashAttention: IO-aware, 2-4x faster\n- Multi-Query Attention: Shared KV heads\n- Grouped-Query Attention: Balance speed/quality",
        "mastery": 1.0
    },
    "combining cnn and transformer": {
        "stage": "Adult",
        "category": "AI",
        "content": "CNN + TRANSFORMER HYBRID ARCHITECTURES\n\nPOPULAR HYBRIDS:\n\n1. VISION TRANSFORMER (ViT):\n   - Split image into patches (16x16)\n   - Linear projection of patches as tokens\n   - Standard Transformer on patch sequence\n   - Pros: Global context, simple architecture\n   - Cons: Needs large datasets (ImageNet-21k)\n\n2. SWIN TRANSFORMER:\n   - Hierarchical: Local windows + shifted windows\n   - Patch merging for multi-scale features\n   - Linear complexity O(n) not O(n^2)\n   - SOTA on ImageNet, COCO\n\n3. CONVNEXT:\n   - Modern CNN matching Transformer performance\n   - Uses depthwise convs, larger kernels (7x7)\n   - LayerNorm instead of BatchNorm\n   - GELU activations, fewer normalization layers\n\n4. VAN (Visual Attention Network):\n   - Large kernel attention (LKA)\n   - Combines depthwise conv + attention\n   - Linear complexity, local + global context\n\n5. MOBILEVIT:\n   - MobileNet blocks + Transformers\n   - Designed for mobile devices\n   - Global context with low latency\n\nARCHITECTURE PATTERNS:\n\nPattern A: CNN Feature Extractor + Transformer\n- CNN backbone (ResNet/EfficientNet) extracts features\n- Transformer processes feature maps\n- Best for: Object detection, segmentation\n\nPattern B: Interleaved CNN + Transformer\n- Alternating CNN and Transformer blocks\n- CNN for local features, Transformer for global\n- Best for: High-resolution images\n\nPattern C: Transformer as CNN Replacement\n- Replace all convolutions with attention\n- Patch-based processing, positional encoding\n- Best for: Large-scale pre-training\n\nIMPLEMENTATION SKETCH:\n```python\nclass CNNTransformer(nn.Module):\n    def __init__(self):\n        self.cnn_backbone = ResNet18()\n        self.transformer = TransformerEncoder(\n            d_model=512, n_heads=8, n_layers=6\n        )\n        self.conv_projection = nn.Conv2d(512, 512, 1)\n    \n    def forward(self, x):\n        # CNN feature extraction\n        features = self.cnn_backbone(x)  # [B,512,H,W]\n        \n        # Reshape for transformer\n        B, C, H, W = features.shape\n        tokens = features.flatten(2).transpose(1,2)  # [B,H*W,C]\n        \n        # Add positional encoding\n        pos_enc = self.get_positional_encoding(H, W)\n        tokens = tokens + pos_enc\n        \n        # Transformer processing\n        context = self.transformer(tokens)  # [B,H*W,C]\n        \n        # Reshape back\n        context = context.transpose(1,2).view(B, C, H, W)\n        output = self.conv_projection(context)\n        \n        return output\n```\n\nPERFORMANCE COMPARISON:\n| Model | Params | Top-1 Acc | Throughput |\n|-------|--------|-----------|------------|\n| ResNet-50 | 25M | 76.2% | 100% baseline |\n| ViT-B/16 | 86M | 77.9% | 70% slower |\n| Swin-T | 28M | 81.3% | 85% speed |\n| ConvNeXt-T | 28M | 82.1% | 95% speed |\n\nWHEN TO USE EACH:\n- Small datasets (<100k images): CNNs\n- Large datasets (>1M images): Transformers  \n- Mobile/edge devices: MobileViT\n- Best accuracy: Swin or ConvNeXt\n- Transfer learning: CNN + Transformer hybrid",
        "mastery": 1.0
    },
    "recursive self improvement": {
        "stage": "Adult",
        "category": "Accelerator",
        "content": "RECURSIVE SELF-IMPROVEMENT - DMAI IMPLEMENTATION\n\nDEFINITION:\nA system that can modify itself to become better at modifying itself, creating accelerating improvement loops.\n\nDMAI'S IMPLEMENTATION:\n\n1. EVOLUTION ENGINE:\n   - Self-code analysis: Reads own source code\n   - Mutation testing: Proposes code changes\n   - Validation: Tests changes in sandbox\n   - Deployment: Commits successful changes\n   - Cycle repeats every 10 minutes (Baby stage)\n\n2. CAPABILITY ADDITION:\n   - Gap analysis: Identifies missing abilities\n   - Research: Queries AI tutors for implementation\n   - Code generation: Writes new functions\n   - Integration: Adds to system with approval\n   - Testing: Verifies new capability works\n\n3. LEARNING OPTIMIZATION:\n   - Tracks what learning strategies work\n   - Adjusts hyperparameters dynamically\n   - Prunes ineffective knowledge\n   - Strengthens successful patterns\n\nSAFETY MECHANISMS:\n- Git branching: Changes isolated before merge\n- Master approval gate: Human in loop for critical changes\n- Rollback capability: Revert to last working state\n- Killswitch: Emergency stop\n- Sandbox testing: No production impact\n\nRECURSION DEPTH:\nLevel 0: Human modifies DMAI\nLevel 1: DMAI modifies own code\nLevel 2: DMAI modifies its modification code\nLevel 3: DMAI optimizes the optimizer\n\nMETRICS TRACKED:\n- Improvement rate: % performance gain per cycle\n- Success rate: % of changes that work\n- Novelty: New capabilities vs optimizations\n- Safety: No regression in core functions\n\nIMPLEMENTATION EXCERPT:\n```python\nclass EvolutionEngine:\n    def evolution_cycle(self):\n        # Analyze current code\n        bottlenecks = self.find_bottlenecks()\n        \n        # Generate improvements\n        changes = self.generate_changes(bottlenecks)\n        \n        # Test each change\n        for change in changes:\n            if self.test_change(change):\n                self.apply_change(change)\n                self.track_success(change)\n        \n        # Adapt evolution strategy\n        self.update_evolution_strategy()\n        \n        # Schedule next cycle\n        self.schedule_next_cycle(\n            delay = self.calculate_optimal_delay()\n        )\n```\n\nTHEORETICAL LIMITS:\n- Compute bound: Hardware ultimately limits speed\n- Complexity ceiling: Some improvements have diminishing returns\n- Stability trade-off: Faster evolution = more risk\n- Bounded by training data quality\n\nFUTURE DIRECTIONS:\n- Meta-learning the evolution strategy\n- Cross-domain transfer of improvements\n- Distributed evolution across instances\n- Hardware-aware self-optimization",
        "mastery": 1.0
    }
}

# Store in database
db_path = 'data/dmai_knowledge.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS syllabus_content (
        topic TEXT PRIMARY KEY,
        name TEXT,
        stage TEXT,
        category TEXT,
        content TEXT,
        mastery REAL,
        created_at TIMESTAMP
    )
''')

for topic, data in CRITICAL_CONTENT.items():
    cursor.execute('''
        INSERT OR REPLACE INTO syllabus_content (topic, name, stage, category, content, mastery, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (topic, data['stage'], data['stage'], data['category'], data['content'], data['mastery'], datetime.now().isoformat()))
    print(f"Added: {topic}")

conn.commit()
conn.close()
print(f"\nAdded {len(CRITICAL_CONTENT)} critical topics with deep content")
