#!/bin/bash
# ============================================
# BULK CLONE ALL RECOMMENDED AI REPOSITORIES
# ============================================

cd ~/Desktop/AI-Evolution-System/repos

echo "🚀 Starting bulk clone of AI repositories..."

# Multimodal AI (text+image+audio)
echo "📦 Cloning CLIP (multimodal)..."
git clone https://github.com/openai/CLIP.git clip

echo "📦 Cloning ImageBind (multimodal)..."
git clone https://github.com/facebookresearch/ImageBind.git imagebind

# Advanced LLMs
echo "📦 Cloning LLaMA-3..."
git clone https://github.com/meta-llama/llama-3.git llama-3

echo "📦 Cloning Mistral..."
git clone https://github.com/mistralai/mistral-src.git mistral

echo "📦 Cloning Phi-3..."
git clone https://github.com/microsoft/phi-3.git phi-3

# AI Agents & Tools
echo "📦 Cloning LangChain..."
git clone https://github.com/langchain-ai/langchain.git langchain

echo "📦 Cloning AutoGen..."
git clone https://github.com/microsoft/autogen.git autogen

echo "📦 Cloning CrewAI..."
git clone https://github.com/crewAIInc/crewAI.git crewai

# Code Generation
echo "📦 Cloning Codeium engine..."
git clone https://github.com/Codeium/engine.git codeium

# Video Generation
echo "📦 Cloning Stable Video Diffusion..."
git clone https://github.com/Stability-AI/generative-models.git stable-video

echo "📦 Cloning VideoCrafter..."
git clone https://github.com/Picsart-AI-Research/VideoCrafter.git video-crafter

# Meta-Learning (for AGI capabilities)
echo "📦 Cloning Meta-Dataset..."
git clone https://github.com/google-research/meta-dataset.git meta-dataset

echo "📦 Cloning PPUDA (meta-learning)..."
git clone https://github.com/uber-research/ppuda.git ppuda

# AutoML & Architecture Search
echo "📦 Cloning auto-sklearn..."
git clone https://github.com/automl/auto-sklearn.git auto-sklearn

echo "📦 Cloning NNI (Neural Network Intelligence)..."
git clone https://github.com/microsoft/nni.git nni

# Self-Supervised Learning
echo "📦 Cloning SwAV..."
git clone https://github.com/facebookresearch/swav.git swav

echo "📦 Cloning bsuite (behavioral suite)..."
git clone https://github.com/deepmind/bsuite.git bsuite

echo ""
echo "✅ ALL REPOSITORIES CLONED SUCCESSFULLY!"
echo "📁 Location: ~/Desktop/AI-Evolution-System/repos/"
echo ""
echo "🔍 Next step: Mark them for evolution"
echo "cd ~/Desktop/AI-Evolution-System"
echo "python mark_all_for_evolution.py"
