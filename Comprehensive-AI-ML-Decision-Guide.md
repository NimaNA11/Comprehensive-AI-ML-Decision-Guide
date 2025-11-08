# Comprehensive AI/ML Decision Guide

## 1. Machine Learning Paradigms

| Approach | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Supervised Learning** | Labeled data available; clear input-output mapping; classification/regression tasks | No labeled data; unlabeled patterns needed; labeling too expensive | Requires quality labels; performance scales with data quality |
| **Unsupervised Learning** | No labels available; pattern discovery; clustering; dimensionality reduction | Clear target variable exists; need interpretable predictions | Results can be ambiguous; validation is challenging |
| **Semi-Supervised Learning** | Small labeled dataset with large unlabeled data; labeling is expensive | Plenty of labeled data; unlabeled data unreliable | Balance between labeled/unlabeled crucial |
| **Self-Supervised Learning** | Large unlabeled data; pretraining needed; NLP/vision tasks | Small datasets; simple tasks; limited compute | Computationally intensive; needs large datasets |
| **Reinforcement Learning** | Sequential decisions; agent-environment interaction; games; robotics | Static prediction; no clear reward signal; safety-critical without simulation | Requires extensive training; reward engineering crucial |
| **Transfer Learning** | Limited target data; similar source domain available; fine-tuning scenarios | Source and target domains very different; abundant target data | Domain similarity critical; can introduce biases |
| **Few-Shot Learning** | Very limited labeled examples; expensive data collection; rapid adaptation | Abundant data available; high accuracy required | Lower accuracy than full training; needs good base model |
| **Zero-Shot Learning** | No examples of target classes; need generalization; attribute-based classification | Target classes in training data; high precision needed | Relies on semantic relationships; lower performance |
| **Active Learning** | Labeling is expensive; want to minimize labeling effort; iterative improvement | Cheap/fast labeling; need all data labeled; static datasets | Requires human-in-the-loop; iterative process |
| **Online Learning** | Streaming data; concept drift; need real-time updates | Static datasets; batch processing acceptable | Model can degrade if not monitored; needs infrastructure |

---

## 2. Classic Machine Learning Algorithms

| Algorithm | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Linear Regression** | Linear relationships; need interpretability; baseline model; small-medium datasets | Non-linear patterns; outliers present; multicollinearity issues | Assumes linearity; sensitive to outliers; check assumptions |
| **Logistic Regression** | Binary/multiclass classification; need probability estimates; interpretability crucial | Non-linear decision boundaries; high-dimensional complex patterns | Fast training; good baseline; probability calibration needed |
| **Decision Trees** | Need interpretability; mixed data types; non-linear relationships; feature interactions | Need high accuracy; prone to overfitting; small datasets | Easy to visualize; captures interactions; unstable |
| **Random Forest** | Tabular data; need robustness; reduce overfitting; feature importance | Real-time predictions (slow); extremely large datasets; text/images | Reduces overfitting; handles mixed types; slower inference |
| **Gradient Boosting (XGBoost/LightGBM/CatBoost)** | Tabular data; competitions; need high accuracy; structured data | Images/text/audio; real-time latency critical; easy interpretability needed | Best for tabular; prone to overfitting; hyperparameter sensitive |
| **Support Vector Machines (SVM)** | Small-medium datasets; high-dimensional data; clear margin separation | Very large datasets; multiple classes; need probability estimates | Kernel choice crucial; computationally expensive; good for text |
| **K-Nearest Neighbors (KNN)** | Simple baseline; non-parametric; small datasets; recommendation systems | Large datasets; high dimensions; need fast predictions | No training phase; slow inference; curse of dimensionality |
| **Naive Bayes** | Text classification; categorical features; need speed; small datasets | Feature independence violated; need high accuracy; continuous features | Fast; works well for text; strong independence assumption |
| **K-Means Clustering** | Spherical clusters; known number of clusters; large datasets; simple segmentation | Non-spherical clusters; unknown K; varying densities | Sensitive to initialization; assumes spherical clusters; scales well |
| **DBSCAN** | Arbitrary cluster shapes; noise handling; unknown cluster count; spatial data | High-dimensional data; varying densities; need specific cluster count | Handles noise well; density-based; hyperparameter sensitive |
| **Hierarchical Clustering** | Need dendrogram; small-medium datasets; explore different granularities | Very large datasets; need speed; clear cluster count | No need to specify K; computationally expensive; visualization helpful |
| **Principal Component Analysis (PCA)** | Dimensionality reduction; visualization; remove collinearity; speed up training | Need interpretable features; data not linearly correlated; sparse data | Linear method; assumes orthogonal components; scale data first |
| **t-SNE** | 2D/3D visualization; exploratory analysis; non-linear manifolds | Dimensionality reduction for ML; preserving global structure; large datasets | Great for visualization; slow; hyperparameter sensitive |
| **Isolation Forest** | Anomaly detection; outlier removal; unsupervised; high-dimensional data | Need interpretability; labeled anomalies available; real-time detection | Efficient; handles high dimensions; no normal class assumption |

---

## 3. Deep Learning Architectures

| Architecture | When to Use | When to Avoid | Key Considerations |
|--------------|-------------|---------------|-------------------|
| **Feedforward Neural Networks (MLP)** | Tabular data; non-linear patterns; moderate complexity; structured inputs | Images/sequences/text; very simple problems; need interpretability | Universal approximators; prone to overfitting; good baseline |
| **Convolutional Neural Networks (CNN)** | Image data; spatial patterns; computer vision; local feature extraction | Tabular data; sequences without spatial structure; very small images | Translation invariant; hierarchical features; needs lots of data |
| **Recurrent Neural Networks (RNN)** | Sequential data; time series; variable-length inputs; temporal dependencies | Very long sequences; parallel processing needed; simple patterns | Captures temporal patterns; vanishing gradients; sequential processing |
| **Long Short-Term Memory (LSTM)** | Long sequences; long-term dependencies; complex temporal patterns | Very long sequences (>1000 steps); need speed; simple patterns | Handles long dependencies; slower than GRU; more parameters |
| **Gated Recurrent Unit (GRU)** | Sequential data; faster than LSTM; moderate sequence lengths | Very long dependencies; when LSTM performance is insufficient | Faster than LSTM; fewer parameters; similar performance |
| **Transformers** | NLP tasks; long-range dependencies; parallel processing; attention needed | Very limited compute; small datasets; simple sequential patterns | State-of-the-art NLP; parallelizable; computationally expensive |
| **Vision Transformers (ViT)** | Large image datasets; need attention; transfer learning; high compute available | Small datasets without pretraining; limited compute; simple images | Competitive with CNNs; needs large data; good for transfer learning |
| **Autoencoders** | Dimensionality reduction; feature learning; denoising; anomaly detection | Need labels; simple linear reduction; supervised learning | Unsupervised; learns compressed representations; various architectures |
| **Variational Autoencoders (VAE)** | Generative modeling; continuous latent space; interpolation; probabilistic | Discrete data generation; deterministic outputs; simple reconstruction | Generates new samples; smooth latent space; probabilistic framework |
| **Generative Adversarial Networks (GAN)** | High-quality generation; images/audio; data augmentation; creative applications | Stable training needed; evaluation metrics crucial; simple tasks | High-quality outputs; training instability; hard to evaluate |
| **ResNet (Residual Networks)** | Very deep networks; image classification; need skip connections; avoiding degradation | Shallow networks sufficient; limited memory; simple tasks | Enables very deep networks; skip connections; widely used |
| **U-Net** | Image segmentation; medical imaging; pixel-wise prediction; limited data | Classification tasks; no spatial localization needed | Encoder-decoder; skip connections; excellent for segmentation |
| **BERT** | Text understanding; sentence classification; NER; Q&A; masked language modeling | Text generation; limited compute; simple text tasks | Bidirectional context; transfer learning; large model |
| **GPT (Decoder-only Transformers)** | Text generation; creative writing; conversational AI; code generation | Text classification only; limited resources; need bidirectional context | Autoregressive; excellent generation; unidirectional |
| **Graph Neural Networks (GNN)** | Graph-structured data; social networks; molecules; knowledge graphs | Tabular/image/text data; no relational structure | Operates on graphs; message passing; various architectures |

---

## 4. Natural Language Processing Approaches

| Approach | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Bag of Words (BoW)** | Simple text classification; baseline; sparse features; document similarity | Word order matters; semantic meaning crucial; limited vocabulary | Ignores order; simple; sparse; good baseline |
| **TF-IDF** | Document ranking; information retrieval; keyword extraction; simple classification | Semantic similarity; word relationships; small documents | Weighs term importance; ignores semantics; classic IR method |
| **Word2Vec** | Word embeddings; semantic similarity; transfer learning; moderate vocab | Very large vocab; need context-aware; latest performance | Captures semantics; fixed embeddings; efficient |
| **GloVe** | Word embeddings; global statistics; semantic relationships | Context-dependent meanings; latest performance | Global co-occurrence; similar to Word2Vec; efficient |
| **FastText** | Handling OOV words; morphologically rich languages; character-level info | Large vocab unnecessary; simple words; English only | Subword information; handles OOV; good for morphology |
| **ELMo** | Context-dependent embeddings; polysemy handling; transfer learning | Limited compute; simple tasks; speed critical | Contextual embeddings; bidirectional; character-aware |
| **Transformers (BERT/GPT/T5)** | State-of-the-art NLP; transfer learning; complex understanding; large datasets | Limited resources; simple tasks; real-time inference | Best performance; expensive; pretrained models available |
| **Sequence-to-Sequence (Seq2Seq)** | Machine translation; summarization; dialogue; variable-length output | Classification; fixed output; need latest performance | Encoder-decoder; attention helpful; RNN/Transformer-based |
| **Named Entity Recognition (NER)** | Extract entities; information extraction; structured from unstructured | No entities to extract; simple keyword matching sufficient | Rule-based or ML; domain-specific; needs labeled data |
| **Sentiment Analysis** | Opinion mining; customer feedback; social media monitoring | Sarcasm/irony heavy; nuanced emotions; multi-aspect | Classification task; various granularities; context matters |
| **Topic Modeling (LDA)** | Discover topics; document clustering; exploratory analysis; unsupervised | Need supervised; clear categories known; small corpus | Unsupervised; interpretable topics; hyperparameter tuning |
| **Text Generation (Language Models)** | Creative writing; content creation; conversation; code generation | Factual accuracy critical; no hallucinations allowed | GPT-style models; coherent but can hallucinate; prompt engineering |

---

## 5. Computer Vision Approaches

| Approach | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Image Classification** | Single label per image; categorization; recognition tasks | Multiple objects; localization needed; pixel-level detail | Well-studied; many pretrained models; transfer learning common |
| **Object Detection (YOLO/Faster R-CNN/SSD)** | Locate and classify objects; bounding boxes; multiple objects | Pixel-level precision; no localization needed; very small objects | Real-time (YOLO) vs accuracy (R-CNN); anchor-based; NMS needed |
| **Semantic Segmentation** | Pixel-wise classification; scene understanding; same class instances | Instance separation needed; object counting; overlapping objects | Dense prediction; per-pixel labels; U-Net popular |
| **Instance Segmentation (Mask R-CNN)** | Separate object instances; precise boundaries; counting objects | Semantic only; no instance separation; speed critical | Combines detection + segmentation; precise; computationally heavy |
| **Image Generation (GANs/Diffusion)** | Synthesize images; data augmentation; creative applications; super-resolution | Need deterministic outputs; training instability unacceptable | High quality; GANs unstable; diffusion models more stable |
| **Face Recognition** | Identity verification; security; photo organization | Privacy concerns; unethical use cases; limited training data | Specialized models; ethical considerations; accuracy critical |
| **Pose Estimation** | Human pose; action recognition; sports analytics; AR/VR | No humans/objects; simple classification; 2D only | Keypoint detection; 2D or 3D; real-time possible |
| **Image Captioning** | Describe images; accessibility; content understanding; multimodal | Simple classification; no text needed | CNN + RNN/Transformer; attention mechanisms; evaluation challenging |
| **Optical Character Recognition (OCR)** | Text extraction; document processing; digitization | No text in images; handwriting too messy | Tesseract popular; deep learning improving; layout matters |
| **Style Transfer** | Artistic effects; creative apps; texture synthesis | Preserve content exactly; real-time; consistency needed | Neural style transfer; GANs; slow but impressive |
| **Super Resolution** | Enhance image quality; upscale images; medical imaging | Original quality sufficient; artifacts unacceptable | Various architectures; perceptual loss; can hallucinate details |
| **Video Analysis** | Action recognition; tracking; temporal patterns; surveillance | Single frame sufficient; no motion; computational limits | Extends image methods; temporal dimension; more complex |

---

## 6. Generative AI Approaches

| Approach | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Large Language Models (LLMs)** | Text generation; reasoning; few-shot learning; general intelligence | Factual accuracy critical; deterministic needed; private data | GPT, Claude, Llama; powerful but expensive; prompt engineering key |
| **Retrieval-Augmented Generation (RAG)** | Grounded generation; reduce hallucination; knowledge-intensive; updatable knowledge | Simple generation; no external knowledge; latency critical | Combines retrieval + generation; reduces hallucination; needs good retrieval |
| **Fine-tuning LLMs** | Domain adaptation; specific style; task specialization; consistent behavior | Limited data; no domain shift; base model sufficient | Parameter-efficient (LoRA); full fine-tuning; needs quality data |
| **Prompt Engineering** | Quick adaptation; no training; explore capabilities; zero/few-shot | Consistent behavior needed; scale to many examples; performance ceiling hit | Cost-effective; no training; model-dependent; iterative |
| **Diffusion Models (Stable Diffusion/DALL-E)** | Image generation; text-to-image; high quality; controllable | Speed critical; deterministic; simple edits | High quality; slower than GANs; iterative refinement |
| **Text-to-Speech (TTS)** | Voice assistants; accessibility; content creation; audiobooks | Privacy concerns; voice cloning ethics; poor quality acceptable | Various architectures; quality improving; voice cloning possible |
| **Speech-to-Text (ASR)** | Transcription; voice interfaces; accessibility; meeting notes | Background noise heavy; multiple speakers; accuracy not critical | Whisper popular; improving rapidly; language support varies |
| **Multimodal Models (CLIP/Flamingo)** | Cross-modal tasks; image-text; zero-shot vision; unified representations | Single modality sufficient; need highest accuracy per modality | Connects modalities; flexible; emergent capabilities |
| **Code Generation (Copilot/CodeLlama)** | Coding assistance; boilerplate; learning; prototyping | Production critical code; security sensitive; full automation | Productivity boost; needs review; hallucinations possible |
| **AI Agents** | Complex workflows; tool use; multi-step reasoning; automation | Simple single-step; deterministic flow needed; no error tolerance | LLM + tools; autonomous; error-prone; needs guardrails |

---

## 7. Activation Functions

| Function | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **ReLU** | Default choice; hidden layers; CNNs; fast training | Output layers; negative values needed; dying ReLU problem | Simple; fast; dead neurons possible; most common |
| **Leaky ReLU** | Prevent dying ReLU; negative values useful; alternative to ReLU | Output layers; need bounded; smooth gradient needed | Small negative slope; prevents dead neurons; slight overhead |
| **Parametric ReLU (PReLU)** | Learn negative slope; flexibility; performance critical | Simple models; limited data; overfitting risk | Learnable parameter; more flexible; slight overfitting risk |
| **ELU** | Negative saturation; reduce bias shift; smoother; better than ReLU | Speed critical; simple problems; computational limits | Negative saturation; smooth; exponential (slower) |
| **SELU** | Self-normalizing; deep networks; specific initialization | Custom architectures; different initialization; unclear benefits | Self-normalizing property; specific conditions; less common |
| **Sigmoid** | Output layer (binary); probabilities [0,1]; gates (LSTM) | Hidden layers (deep nets); vanishing gradients | Vanishing gradients; S-shaped; output probabilities |
| **Tanh** | RNN hidden layers; centered output [-1,1]; better than sigmoid | Very deep networks; dying gradient issues | Zero-centered; still vanishes; better than sigmoid |
| **Softmax** | Output layer (multiclass); probability distributions; attention | Binary classification; not mutually exclusive; regression | Converts to probabilities; sum to 1; mutually exclusive |
| **Swish (SiLU)** | Deep networks; better than ReLU; smooth; NAS-discovered | Simple networks; speed critical; interpretability needed | Smooth; non-monotonic; computationally heavier |
| **GELU** | Transformers; NLP; state-of-the-art; stochastic regularization | Simple models; not transformers; ReLU sufficient | Used in BERT/GPT; smooth; probabilistic interpretation |
| **Mish** | Competitive with Swish; smooth; self-regularizing | Speed critical; simple models; marginal gains | Smooth; unbounded above; computationally expensive |

---

## 8. Loss Functions

| Loss Function | When to Use | When to Avoid | Key Considerations |
|---------------|-------------|---------------|-------------------|
| **Mean Squared Error (MSE)** | Regression; continuous targets; penalize large errors; Gaussian noise | Outliers present; classification; robust loss needed | Sensitive to outliers; standard regression; penalizes large errors |
| **Mean Absolute Error (MAE)** | Regression with outliers; robust loss; all errors equal | Need to penalize large errors more; smooth gradients needed | Robust to outliers; equal penalty; non-smooth at zero |
| **Huber Loss** | Robust regression; balance MSE/MAE; outlier presence uncertain | Pure MSE or MAE sufficient; simple problems | Combines MSE + MAE; robust; hyperparameter delta |
| **Binary Cross-Entropy** | Binary classification; logistic regression; probability outputs | Multiclass; imbalanced data; regression | Standard binary loss; requires sigmoid output; log-based |
| **Categorical Cross-Entropy** | Multiclass single-label; mutually exclusive; softmax output | Multi-label; binary; ordinal targets | Requires softmax; mutually exclusive classes; log-based |
| **Sparse Categorical Cross-Entropy** | Multiclass with integer labels; save memory; large class count | One-hot encoded labels; small class count | Same as categorical; integer labels; memory efficient |
| **Binary Focal Loss** | Imbalanced classification; hard examples; object detection | Balanced data; simple problems; standard loss works | Focuses on hard examples; imbalance handling; gamma parameter |
| **Hinge Loss** | SVM; margin-based; binary classification; max-margin | Need probabilities; not binary; neural networks (usually) | Margin-based; used in SVM; less common in NNs |
| **Kullback-Leibler Divergence** | Distribution matching; VAEs; probabilistic models; information theory | Simple regression; deterministic; MSE sufficient | Measures distribution difference; asymmetric; VAE loss component |
| **Contrastive Loss** | Siamese networks; similarity learning; face verification; metric learning | Classification; no pairs; standard supervised | Pairs of examples; similarity metric; embedding learning |
| **Triplet Loss** | Metric learning; face recognition; ranking; embedding spaces | Simple classification; no triplets; contrastive sufficient | Anchor-positive-negative triplets; margin; embedding optimization |
| **CTC Loss** | Sequence labeling; speech recognition; OCR; alignment unknown | Fixed-length output; alignment known; simple classification | Handles variable length; no alignment needed; specialized use |

---

## 9. Optimization Algorithms

| Optimizer | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Stochastic Gradient Descent (SGD)** | Well-understood problem; simple baseline; with momentum; good generalization | Need adaptive rates; many hyperparameters; faster convergence needed | Simple; good generalization; requires learning rate tuning; momentum helps |
| **SGD with Momentum** | Accelerate SGD; navigate ravines; oscillation reduction; general purpose | Adaptive learning rates better; problem-specific optimizers | Faster than SGD; reduces oscillation; momentum hyperparameter |
| **Nesterov Accelerated Gradient (NAG)** | Improved momentum; look-ahead; convex optimization | Not significantly better; simpler sufficient | Look-ahead momentum; slightly better; similar to momentum |
| **Adagrad** | Sparse data; NLP; different learning rates per parameter; adaptive | Dense updates; learning rate vanishes; long training | Adapts per parameter; good for sparse; learning rate decay issue |
| **RMSprop** | RNNs; non-stationary; fix Adagrad decay; general purpose | Adagrad works; very simple problems | Fixes Adagrad decay; moving average; good default |
| **Adam** | Default choice; most problems; adaptive; combines momentum + RMSprop | Fine-tuning LLMs; proven SGD better; generalization critical | Most popular; adaptive; momentum + RMSprop; good default |
| **AdamW** | Weight decay; regularization; transformers; modern default | Adam sufficient; no regularization needed | Adam with decoupled weight decay; better regularization; modern standard |
| **Nadam** | Combine Adam + Nesterov; slight improvements; general purpose | Adam sufficient; marginal gains | Adam + Nesterov; slightly better; less common |
| **Adamax** | Adam variant; infinity norm; stable; specific problems | Adam works well; no clear advantage | Adam variant; infinity norm; more stable (sometimes) |
| **Adadelta** | No learning rate; adaptive; Adagrad extension | Need learning rate control; Adam better | No learning rate needed; less popular; adaptive |
| **LAMB** | Large batch training; distributed; BERT pretraining; scale up | Small batch; single GPU; standard training | Layer-wise adaptation; large batch; distributed training |
| **Lookahead** | Wrap other optimizers; stability; smoother convergence | Simple sufficient; added complexity | Wraps optimizers; k steps forward; slower but stable |

---

## 10. Regularization Techniques

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **L1 Regularization (Lasso)** | Feature selection; sparse models; many irrelevant features; interpretability | All features relevant; smooth solutions needed; correlated features | Sparse solutions; feature selection; non-differentiable at zero |
| **L2 Regularization (Ridge)** | Prevent overfitting; correlated features; smooth solutions; general purpose | Need sparsity; feature selection critical; very simple model | Smooth solutions; handles collinearity; most common |
| **Elastic Net** | Combine L1 + L2; correlated features + sparsity; flexible | L1 or L2 alone sufficient; simple problem | Combines L1 + L2; flexible; two hyperparameters |
| **Dropout** | Neural networks; overfitting; large networks; regularization needed | RNNs (use variational); very small networks; inference time | Random neuron dropping; ensemble effect; only training |
| **Spatial Dropout** | CNNs; spatial correlations; feature maps; convolutional layers | Fully connected; no spatial structure | Drops feature maps; better for CNNs; maintains spatial structure |
| **Variational Dropout** | RNNs; temporal data; shared mask; sequence models | Standard dropout works; not sequential | Same mask across time; RNN-specific; better than standard |
| **Batch Normalization** | Deep networks; speed training; reduce covariate shift; almost always | Very small batches; batch size varies; RNNs (use Layer Norm) | Normalizes activations; speeds training; batch dependent |
| **Layer Normalization** | RNNs/Transformers; variable batch size; sequence models; batch norm inappropriate | Batch norm works; CNNs (usually) | Normalizes across features; batch independent; transformers use |
| **Group Normalization** | Small batches; batch norm fails; computer vision; batch size constrained | Large batches; batch norm sufficient | Groups of channels; batch independent; small batch alternative |
| **Instance Normalization** | Style transfer; GANs; per-instance; image generation | Batch statistics needed; classification | Per-instance; style transfer; image generation |
| **Weight Decay** | Prevent overfitting; L2 on weights; almost always; regularization | Need sparsity; very simple model; underfitting | L2 penalty on weights; simple; effective |
| **Early Stopping** | Prevent overfitting; validation set available; simple; general purpose | No validation set; need full training; online learning | Monitor validation; stop when overfitting; simple and effective |
| **Data Augmentation** | Limited data; images/audio; prevent overfitting; improve generalization | Abundant data; changes semantics; tabular data (usually) | Synthetic variations; preserves labels; domain-specific |
| **Label Smoothing** | Overconfident models; calibration; classification; reduce overfitting | Regression; confidence needed; binary with uncertainty | Softens targets; prevents overconfidence; improves calibration |
| **Mixup** | Image classification; data augmentation; regularization; limited data | Interpretability needed; linear combinations invalid | Mixes samples; linear interpolation; effective augmentation |
| **Cutout/CutMix** | Image classification; occlusion robustness; data augmentation; CNNs | Non-image data; simple augmentation sufficient | Random masking/mixing; forces robust features; image-specific |

---

## 11. Data Preprocessing & Feature Engineering

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Normalization (Min-Max Scaling)** | Bounded range [0,1]; neural networks; distance-based; outlier-free | Outliers present; need zero-centered; distribution skewed | Scales to [0,1]; sensitive to outliers; preserves zero |
| **Standardization (Z-score)** | Gaussian-like; PCA; zero-centered; general purpose; outliers acceptable | Need bounded; interpretability; tree-based models | Zero mean, unit variance; handles outliers better; common choice |
| **Robust Scaling** | Many outliers; median-based; robust statistics; skewed distributions | No outliers; normal distribution; standard scaling sufficient | Uses median/IQR; robust to outliers; skewed data |
| **Log Transformation** | Skewed data; multiplicative relationships; reduce skewness; positive data | Negative/zero values; already normal; additive relationships | Reduces skewness; handles exponential; only positive |
| **Box-Cox Transformation** | Achieve normality; stabilize variance; statistical tests; positive data | Interpretability needed; negative values; neural networks (usually) | Finds best power transform; achieves normality; statistical |
| **One-Hot Encoding** | Nominal categories; no order; neural networks; moderate cardinality | High cardinality; ordinal data; tree-based models (sometimes) | Binary columns; no ordinality; increases dimensionality |
| **Label Encoding** | Ordinal categories; tree-based models; natural order; low cardinality | Nominal without order; neural networks (usually); distance-based | Integer encoding; implies order; tree-based friendly |
| **Target Encoding** | High cardinality; category-target relationship; boosting models | Overfitting risk; small groups; leakage concerns | Encodes with target stats; leakage risk; cross-validation needed |
| **Frequency Encoding** | High cardinality; frequency matters; simple; low memory | Frequency uninformative; need semantics | Counts/proportions; simple; loses category info |
| **Embedding Encoding** | High cardinality; neural networks; learn representations; entity embeddings | Small cardinality; interpretability; simple models | Learned representations; powerful; neural networks |
| **Binning/Discretization** | Continuous to categorical; reduce noise; non-linear relationships; robust to outliers | Lose information; neural networks (usually); need precision | Groups values; reduces noise; loses granularity |
| **Polynomial Features** | Non-linear relationships; feature interactions; linear models; simple features | High dimensionality; complex models; overfitting risk | Creates interactions; exponential growth; overfitting risk |
| **Feature Selection (Filter)** | High dimensionality; remove irrelevant; fast; univariate | Feature interactions; model-specific; need wrapper | Statistical tests; fast; ignores interactions |
| **Feature Selection (Wrapper)** | Model-specific; feature interactions; small feature sets; accuracy critical | Large feature sets; computational cost; interpretability | Uses model; considers interactions; expensive |
| **Feature Selection (Embedded)** | During training; L1/tree importance; balance speed/accuracy; regularization | No regularization; separate selection needed | Built-in; efficient; model-dependent |
| **Missing Value Imputation (Mean/Median)** | Simple; MAR assumption; continuous; baseline | MNAR; categorical; sophisticated needed | Simple; fast; ignores patterns |
| **Missing Value Imputation (KNN)** | Similar samples; local patterns; moderate missing; numerical | Many missing; high dimensions; speed critical | Uses neighbors; preserves patterns; slower |
| **Missing Value Imputation (Iterative)** | Complex patterns; MICE; multiple features; statistical | Simple sufficient; speed critical; many missing | Iterative modeling; sophisticated; slower |
| **Missing Indicator** | Informative missingness; MNAR; complement imputation | MAR; not informative; bloat features | Flags missing; captures pattern; additional feature |
| **Outlier Removal (IQR)** | Gaussian-like; errors/noise; statistical; univariate | Legitimate values; multivariate; domain knowledge | 1.5*IQR rule; simple; univariate |
| **Outlier Removal (Z-score)** | Gaussian; statistical threshold; univariate; simple | Non-Gaussian; multivariate; robust needed | Standard deviations; assumes Gaussian; simple |
| **Outlier Removal (Isolation Forest)** | Multivariate; no distribution assumption; anomaly detection; complex patterns | Simple univariate; statistical sufficient; speed critical | Tree-based; multivariate; sophisticated |
| **SMOTE (Oversampling)** | Imbalanced classification; minority class; synthetic samples | Overlapping classes; noise; overfitting risk | Synthetic samples; interpolation; overfitting possible |
| **Undersampling** | Imbalanced; large dataset; fast; reduce majority | Small dataset; lose information; diverse majority | Remove majority; simple; information loss |
| **Class Weights** | Imbalanced; built-in support; no resampling; maintain distribution | Algorithm doesn't support; need sampling control | Weight loss; algorithm-dependent; simple |

---

## 12. Model Evaluation Metrics

| Metric | When to Use | When to Avoid | Key Considerations |
|--------|-------------|---------------|-------------------|
| **Accuracy** | Balanced classes; equal error costs; simple; general purpose | Imbalanced data; unequal costs; minority important | Simple; intuitive; misleading when imbalanced |
| **Precision** | False positives costly; spam detection; high confidence; specificity | Recall important; coverage needed; balanced | TP / (TP + FP); false positive focus |
| **Recall (Sensitivity)** | False negatives costly; medical; fraud; coverage; completeness | Precision matters; resources limited; false alarms costly | TP / (TP + FN); false negative focus |
| **F1 Score** | Balance precision/recall; imbalanced; single metric; harmonic mean | Unequal importance; need separate metrics; beta differs from 1 | Harmonic mean; balances P&R; single metric |
| **F-beta Score** | Custom precision/recall balance; beta parameter; flexibility | Equal importance; F1 sufficient | Weighted harmonic mean; flexible; beta tunes balance |
| **ROC-AUC** | Threshold-independent; probability ranking; compare models; balanced-ish | Severely imbalanced; need actual predictions; calibration | Plots TPR vs FPR; threshold-independent; robust |
| **PR-AUC** | Imbalanced data; precision-recall focus; better than ROC-AUC for imbalance | Balanced classes; ROC-AUC sufficient | Precision-Recall curve; imbalanced data; informative |
| **Confusion Matrix** | Detailed errors; all metrics; classification; error analysis | Single number needed; quick comparison | Full error breakdown; interpretable; basis for metrics |
| **Mean Squared Error (MSE)** | Regression; penalize large errors; continuous; differentiable | Outliers; equal penalties; interpretable scale | Squared errors; sensitive to outliers; standard |
| **Root Mean Squared Error (RMSE)** | Regression; interpretable scale; same units; standard | Outliers; robust needed; distribution understanding | Square root of MSE; original units; interpretable |
| **Mean Absolute Error (MAE)** | Regression; robust; equal penalties; outliers | Large errors more important; differentiability needed | Absolute errors; robust; equal penalty |
| **R-squared (R²)** | Regression; explained variance; model goodness; compare models | Non-linear relationships; extrapolation; causation | Proportion variance explained; 0-1 (can be negative); interpretable |
| **Adjusted R-squared** | Multiple features; penalize complexity; feature selection; compare models | Single feature; complexity not issue | Adjusts for features; penalizes complexity; model comparison |
| **Log Loss** | Probability quality; calibration; probabilistic predictions; gradient-based | Point predictions; not probabilistic; interpretability | Cross-entropy; probability quality; lower better |
| **Perplexity** | Language models; sequence modeling; compare LMs; information theory | Not language; classification; regression | Exponential of log loss; LM-specific; lower better |
| **BLEU Score** | Machine translation; n-gram overlap; text generation; automatic evaluation | Semantic similarity; not translation; multiple references lacking | N-gram precision; translation; 0-1 (higher better) |
| **ROUGE Score** | Summarization; recall-oriented; overlap; automatic evaluation | Generation quality; semantic; fluency | Recall-focused; overlap-based; summarization |
| **Mean Average Precision (MAP)** | Ranking; information retrieval; recommendation; ordered results | Classification; not ranking; single relevant item | Averages precision; ranking quality; IR metric |
| **Normalized Discounted Cumulative Gain (NDCG)** | Ranking; graded relevance; position matters; search/recommendation | Binary relevance; position independent; classification | Position-weighted; graded relevance; complex but informative |
| **Intersection over Union (IoU)** | Object detection; segmentation; bounding boxes; pixel overlap | Classification; no spatial; localization not needed | Overlap metric; 0-1; detection/segmentation |
| **Mean IoU (mIoU)** | Segmentation; average over classes; semantic understanding | Single class; detection only; classification | Average IoU; segmentation standard; class-wise |
| **Fréchet Inception Distance (FID)** | Image generation quality; GAN evaluation; distribution distance | Not generation; interpretability; simple metrics sufficient | Distribution distance; lower better; generation evaluation |

---

## 13. Cross-Validation Strategies

| Strategy | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **K-Fold CV** | Standard; general purpose; balanced data; estimate generalization | Time series; grouped data; imbalanced; huge datasets | Random splits; standard; k usually 5-10 |
| **Stratified K-Fold** | Classification; imbalanced; maintain distribution; standard | Regression; balanced; no stratification needed | Preserves class distribution; classification default; handles imbalance |
| **Leave-One-Out (LOOCV)** | Very small datasets; maximum training data; unbiased estimate | Large datasets; computationally expensive; high variance | N folds; expensive; small data only |
| **Time Series Split** | Temporal data; respect time order; forecast; prevent leakage | No temporal dependency; shuffling okay; i.i.d. data | Respects time; no future leakage; expanding/sliding window |
| **Group K-Fold** | Grouped data; patient/user groups; prevent leakage; dependencies | No groups; independent samples; standard split okay | Groups stay together; prevents leakage; hierarchy respected |
| **Nested CV** | Hyperparameter tuning + evaluation; unbiased estimate; publications | Simple tuning; computational cost; overkill for exploratory | Outer + inner loops; unbiased; expensive |
| **Repeated K-Fold** | Reduce variance; more robust estimate; small data; stability | Large data; time constraints; single run sufficient | Multiple runs; averages results; more stable |
| **Holdout Validation** | Large data; quick iteration; single split; computational limits | Small data; need robustness; multiple estimates | Single split; fast; higher variance |
| **Bootstrap** | Small data; uncertainty estimation; sampling with replacement | Time series; standard CV better; grouped data | Sampling with replacement; OOB samples; statistical |

---

## 14. Ensemble Methods

| Method | When to Use | When to Avoid | Key Considerations |
|--------|-------------|---------------|-------------------|
| **Bagging (Bootstrap Aggregating)** | Reduce variance; unstable models; decision trees; parallel | Stable models; bias problem; single model sufficient | Parallel; reduces variance; random sampling |
| **Random Forest** | Tabular data; robust; feature importance; general purpose; default choice | Images/text; interpretability critical; real-time | Bagging + feature sampling; robust; widely used |
| **Boosting (AdaBoost/Gradient Boosting)** | Sequential; reduce bias; tabular data; competitions; high accuracy | Noisy data; overfitting prone; need speed; interpretability | Sequential; focuses on errors; powerful but slower |
| **XGBoost** | Kaggle; tabular data; fast boosting; regularized; feature importance | Images/sequences; model size issue; interpretability | Optimized boosting; regularized; parallel; popular |
| **LightGBM** | Large datasets; fast; memory efficient; categorical support; boosting | Small data; GPU unavailable (for speedup); overfitting risk | Histogram-based; fast; large data; efficient |
| **CatBoost** | Categorical features; ordered boosting; reduce overfitting; robust | No categoricals; standard boosting works; speed critical | Handles categoricals; ordered; robust; less tuning |
| **Stacking** | Combine diverse models; squeeze performance; competitions; heterogeneous | Overfitting risk; simple sufficient; interpretability | Meta-learner; combines predictions; powerful but complex |
| **Blending** | Simpler than stacking; holdout for meta; quick ensemble; less overfitting | Need full stacking; small data; simple average works | Holdout for meta; simpler; less overfitting than stacking |
| **Voting (Hard)** | Majority vote; classification; diverse models; simple | Regression; need probabilities; weighted better | Majority vote; simple; classification; equal weights |
| **Voting (Soft)** | Weighted probabilities; classification; confidence matters; average | Hard voting sufficient; no probabilities; regression | Averages probabilities; weighted; better than hard |
| **Weighted Average** | Regression; different model strengths; simple; quick | Classification; complex weighting; stacking better | Weight predictions; simple; needs weight tuning |
| **Snapshot Ensembles** | Single training run; cyclic LR; cheap ensemble; neural networks | Multiple runs feasible; not neural networks; simple sufficient | Cyclic learning rate; saves snapshots; cheap ensemble |

---

## 15. Hyperparameter Tuning

| Method | When to Use | When to Avoid | Key Considerations |
|--------|-------------|---------------|-------------------|
| **Grid Search** | Small space; exhaustive; interpretable; discrete parameters | Large space; expensive; continuous parameters | Exhaustive; expensive; simple; parallelizable |
| **Random Search** | Large space; better than grid; continuous; limited budget | Small discrete space; grid sufficient; need deterministic | Random sampling; efficient; better than grid often |
| **Bayesian Optimization** | Expensive evaluations; continuous space; model-based; small iterations | Cheap evaluations; high dimensions; large discrete | Probabilistic model; efficient; expensive per iteration |
| **Halving (Successive Halving)** | Many configs; early stopping; limited budget; quick elimination | All configs need full evaluation; small search space | Progressively eliminates; resource-efficient; modern |
| **Hyperband** | Combines halving + random; adaptive; efficient; large space | Simple sufficient; deterministic needed | Adaptive halving; efficient; state-of-the-art |
| **Optuna** | Automated; pruning; parallel; modern; efficient | Simple grid sufficient; no framework needed | Framework; pruning; efficient; popular |
| **Population-Based Training** | Neural networks; online tuning; dynamic; parallel resources | Limited resources; simple tuning sufficient; not online | Evolves hyperparameters; online; parallel; sophisticated |
| **Manual Tuning** | Domain knowledge; quick start; limited resources; experience | Systematic needed; reproducibility; scale | Experience-driven; quick; not reproducible; starting point |

---


## 16. Data Sampling Strategies

| Strategy | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Random Sampling** | Large dataset; i.i.d.; simple; unbiased | Rare events; imbalanced; stratification needed | Simple; unbiased; may miss rare classes |
| **Stratified Sampling** | Classification; maintain distribution; surveys; representative | Regression; no clear strata; continuous only | Preserves proportions; representative; classification |
| **Systematic Sampling** | Ordered data; periodic patterns avoided; simple; efficient | Periodic patterns; randomness needed; bias risk | Every kth element; efficient; pattern risk |
| **Cluster Sampling** | Grouped data; geography; cost-effective; hierarchical | Within-cluster homogeneity; precision needed | Groups as units; cost-effective; higher variance |
| **Reservoir Sampling** | Streaming data; unknown size; memory limited; uniform probability | Known size; static data; batch processing | Online; memory-efficient; uniform probability |
| **Importance Sampling** | Rare events; weighted; reduce variance; biased distribution | Uniform sufficient; weights unknown; interpretability | Weighted sampling; variance reduction; sophisticated |
| **Negative Sampling** | Recommendation; implicit feedback; large item space; contrastive | Explicit labels; small space; positive examples sufficient | Samples negatives; efficient; implicit feedback systems |
| **Hard Negative Mining** | Difficult examples; detection; face recognition; improve boundaries | Easy dataset; simple sufficient; all examples useful | Focuses on hard negatives; improves boundaries; active learning |

---

## 17. Neural Network Initialization

| Method | When to Use | When to Avoid | Key Considerations |
|--------|-------------|---------------|-------------------|
| **Xavier/Glorot Initialization** | Sigmoid/tanh; feedforward; standard; symmetric activations | ReLU; ResNets; asymmetric activations | Variance preservation; tanh/sigmoid; standard approach |
| **He Initialization** | ReLU networks; CNNs; modern default; asymmetric activations | Sigmoid/tanh; SELU; symmetric | Accounts for ReLU; modern standard; doubles Xavier variance |
| **LeCun Initialization** | SELU; normalized; specific; theoretical | ReLU; standard networks; self-normalizing not used | SELU-specific; normalized; less common |
| **Zero Initialization** | Biases; batch norm parameters; specific layers | Weights (symmetry); general initialization | Breaks symmetry if weights; okay for biases |
| **Random Normal** | Simple baseline; exploration; non-standard cases | Standard architectures; proven methods exist | Simple; not optimal; variance matters |
| **Random Uniform** | Alternative to normal; bounded; simple | Better methods available; standard networks | Bounded range; simple; not optimal |
| **Orthogonal Initialization** | RNNs; preserve gradients; deep networks; eigenvalues | Not critical; standard works; computational cost | Orthogonal matrices; preserves norms; RNN-friendly |
| **Identity Initialization** | Skip connections; residual networks; preserve signal | General layers; learning needed | Identity matrix; residual blocks; signal preservation |
| **Pretrained Initialization (Transfer Learning)** | Limited data; similar domain; fine-tuning; feature extraction | Abundant data; very different domain; train from scratch | Pretrained weights; powerful; domain similarity key |

---

## 18. Batch Size Selection

| Batch Size | When to Use | When to Avoid | Key Considerations |
|------------|-------------|---------------|-------------------|
| **Full Batch** | Convex; small data; deterministic; theoretical | Large data; need regularization; modern deep learning | Deterministic; smooth; poor generalization; rare |
| **Mini-Batch (32-256)** | Standard deep learning; balance speed/generalization; GPU efficiency | Very small data; full batch possible; memory constraints | Sweet spot; GPU efficient; good generalization; standard |
| **Small Batch (1-32)** | Memory limited; regularization effect; noisy gradient beneficial | Speed critical; stable training needed; large models | More noise; regularization; slower; memory-friendly |
| **Large Batch (256+)** | Distributed training; speed critical; stable gradients; large clusters | Generalization critical; limited compute; sharp minima risk | Fast; needs learning rate scaling; generalization risk |
| **Single Sample (Stochastic)** | Online learning; extreme memory limit; very noisy okay | Slow; unstable; vectorization benefits lost | Maximum noise; slow; online learning; rare |
| **Gradient Accumulation** | Effective large batch; memory limited; distributed unavailable | Memory sufficient; added complexity unnecessary | Simulates large batch; memory-efficient; extra steps |

---

## 19. Learning Rate Schedules

| Schedule | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Constant** | Simple baseline; adaptive optimizers; short training; quick experiments | Long training; need convergence; performance critical | No adjustment; simple; often suboptimal |
| **Step Decay** | Periodic drops; milestones known; simple; traditional | Unknown schedule; smooth decay better | Drops at epochs; simple; manual schedule |
| **Exponential Decay** | Smooth decay; continuous; automatic; general purpose | Step decay sufficient; warm restarts needed | Exponential decrease; smooth; common |
| **Cosine Annealing** | Smooth decay; warm restarts; modern; periodic | Simple sufficient; constant better; short training | Cosine curve; smooth; restart option; modern |
| **Linear Decay** | Simple; to zero; fine-tuning; linear decrease | Need smooth; sophisticated better | Linear to zero; simple; fine-tuning |
| **Polynomial Decay** | Flexible curve; power parameter; general; smooth | Simple sufficient; exponential works | Power decay; flexible; parameter tunes curve |
| **Warm Restarts (SGDR)** | Escape local minima; ensemble; long training; cyclic | Short training; convergence critical; simple sufficient | Periodic restarts; exploration; snapshot ensembles |
| **Warmup** | Transformers; large batch; initial stability; modern | Small models; stable initialization; unnecessary | Gradual increase; stabilizes; transformer standard |
| **One Cycle Policy** | Fast training; cycle through rates; modern; efficient | Multiple cycles needed; very long training | Peak then decay; fast; popular in fast.ai |
| **ReduceLROnPlateau** | Adaptive; validation-based; automatic; robust | Validation unstable; scheduled better; no validation | Reduces on plateau; adaptive; validation-driven |
| **Cyclic Learning Rates** | Find optimal LR; triangular; cycle through; explore | Constant works; unnecessary complexity | Cycles between bounds; finds ranges; exploratory |

---

## 20. Handling Imbalanced Data

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Class Weights** | Built-in support; simple; maintain data; loss weighting | Algorithm doesn't support; need more control | Weights loss function; simple; algorithm-dependent |
| **Oversampling (Random)** | Imbalanced; increase minority; simple; more data | Overfitting risk; duplicates; sophisticated needed | Duplicates minority; simple; overfitting risk |
| **SMOTE** | Synthetic samples; interpolation; sophisticated; reduce overfitting | Overlapping classes; noise; high dimensions | Synthetic interpolation; sophisticated; popular |
| **ADASYN** | Adaptive SMOTE; focus on difficult; density-based | SMOTE sufficient; noise sensitivity | Adaptive density; focuses on hard; SMOTE variant |
| **Undersampling (Random)** | Large majority; fast; reduce size; quick fix | Small data; lose information; diverse majority | Removes majority; information loss; fast |
| **Tomek Links** | Clean overlap; remove noise; boundary cleaning | No overlap; pure undersampling; time-consuming | Removes boundary pairs; cleaning; sophisticated |
| **Edited Nearest Neighbors** | Clean noisy majority; KNN-based; remove misclassified | Clean data; computation cost; pure undersampling | Removes misclassified; KNN-based; cleaning |
| **One-Sided Selection** | Combine Tomek + CNN; aggressive; clean | Lose too much; gentle needed | Aggressive undersampling; combines methods |
| **Ensemble Methods** | Combine sampling; BalancedRandomForest; EasyEnsemble | Simple sufficient; single model works | Multiple samplings; ensemble; sophisticated |
| **Focal Loss** | Focus on hard; address imbalance; object detection; deep learning | Traditional ML; balanced data; focal gamma tuning | Focuses on hard examples; deep learning; tunable |
| **Threshold Moving** | Post-processing; adjust decision; precision-recall trade-off | Training-time fix; threshold obvious | Adjusts threshold; post-training; PR curve |
| **Anomaly Detection Approach** | Extreme imbalance; one-class; outlier detection | Multi-class; moderate imbalance | Treats as anomaly; extreme imbalance; one-class SVM |

---

## 21. Model Compression & Optimization

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Quantization** | Reduce model size; mobile/edge; inference speed; memory limited | Training; accuracy critical; floating precision needed | INT8/INT4; smaller; faster; slight accuracy loss |
| **Pruning** | Remove weights; sparse; compression; mobile | Dense better; accuracy sensitive; training cost high | Removes connections; sparse; iterative pruning |
| **Knowledge Distillation** | Compress to smaller; student-teacher; transfer; edge deployment | Small model sufficient; from scratch okay | Teacher guides student; soft targets; compression |
| **Low-Rank Factorization** | Matrix decomposition; reduce parameters; linear layers | Minimal benefit; accuracy loss; complex | Decomposes weights; reduces params; linear layers |
| **Neural Architecture Search (NAS)** | Automated design; optimal architecture; resources available | Manual sufficient; computational budget; simple task | Automated search; expensive; state-of-the-art |
| **Mixed Precision Training** | Faster training; memory save; FP16; modern GPUs | Older GPUs; numerical instability; accuracy critical | FP16 + FP32; faster; memory efficient; modern |
| **Gradient Checkpointing** | Memory limited; very deep; trade compute for memory | Memory sufficient; speed critical | Recomputes; saves memory; slower backward |
| **Model Parallelism** | Model too large; GPU memory exceeded; layers distributed | Model fits; added complexity; data parallelism sufficient | Splits model; across devices; very large models |
| **Data Parallelism** | Multiple GPUs; batch parallelism; standard scaling; same model copies | Single GPU sufficient; model too large; batch size limited | Replicates model; splits batch; standard approach |
| **Pipeline Parallelism** | Very deep models; layer-wise parallel; balanced pipeline | Shallow models; complexity; bubble overhead | Layer stages; pipeline; mini-batches; sophisticated |

---

## 22. Interpretability & Explainability

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Feature Importance (Tree-based)** | Tree models; understand features; built-in; quick | Neural networks; model-agnostic needed | Built-in; fast; tree-specific; interpretable |
| **SHAP (SHapley Additive exPlanations)** | Model-agnostic; rigorous; game theory; detailed | Speed critical; simple sufficient; computational cost | Shapley values; rigorous; slow; comprehensive |
| **LIME (Local Interpretable Model-agnostic Explanations)** | Local explanations; model-agnostic; instance-level; intuitive | Global understanding; unstable; hyperparameter sensitive | Local linear; intuitive; perturbation-based; popular |
| **Permutation Importance** | Model-agnostic; global importance; simple; any model | Large datasets (slow); correlated features; training data only | Permutes features; model-agnostic; simple; reliable |
| **Partial Dependence Plots (PDP)** | Feature effect; marginal; visualization; global | Feature interactions; categorical; many features | Shows marginal effect; assumes independence; visualization |
| **Individual Conditional Expectation (ICE)** | Individual predictions; complement PDP; heterogeneity | Summary sufficient; PDP enough; many instances | Individual curves; shows variation; complements PDP |
| **Grad-CAM** | CNN visualization; heatmaps; localization; visual | Not CNNs; pixel-level; interpretability sufficient | Gradient-based; heatmaps; CNNs; visualization |
| **Attention Visualization** | Transformers; attention weights; token importance; NLP | Not transformers; attention not informative | Visualizes attention; intuitive; transformers; interpretable |
| **Saliency Maps** | Pixel importance; vision; gradient-based; simple | Not images; sophisticated methods better | Gradient-based; simple; noisy; basic |
| **Integrated Gradients** | Attribution; rigorous; vision/NLP; better than saliency | Simple sufficient; computational cost | Path integral; rigorous; better than saliency |
| **Counterfactual Explanations** | "What if" scenarios; actionable; contrastive; intuitive | Infeasible changes; computational cost | Minimal changes; actionable; intuitive; generating hard |
| **Anchors** | Rule-based; local; high precision; interpretable | Global needed; rules too simple; coverage low | Rule-based; interpretable; local; high precision |
| **Decision Trees as Surrogates** | Global approximation; interpretable proxy; rule extraction | Accuracy critical; complex fidelity; tree insufficient | Approximates model; interpretable; fidelity varies |

---

## 23. Deployment Considerations

| Approach | When to Use | When to Avoid | Key Considerations |
|----------|-------------|---------------|-------------------|
| **Batch Prediction** | Periodic updates; non-real-time; large volumes; scheduled | Real-time needed; instant response; user-facing | Processes batches; efficient; latency okay |
| **Online/Real-time Prediction** | User-facing; instant response; API; low latency | Batch sufficient; expensive; complexity | Immediate; API endpoint; latency critical; infrastructure |
| **Edge Deployment** | Privacy; offline; mobile/IoT; low latency; bandwidth limited | Cloud sufficient; large models; frequent updates | On-device; private; limited resources; compressed models |
| **Cloud Deployment** | Scalable; centralized; large models; frequent updates; shared | Privacy concerns; latency; cost per call | Scalable; flexible; latency higher; managed services |
| **Serverless** | Variable load; auto-scaling; cost-efficient; event-driven | Consistent high load; cold start issue; stateful | Auto-scales; pay-per-use; cold starts; managed |
| **Model Serving Frameworks (TensorFlow Serving/TorchServe)** | Production; standardized; optimized; model versioning | Simple API sufficient; overkill; different framework | Optimized serving; versioning; framework-specific; production |
| **ONNX** | Framework interoperability; convert models; optimize; portable | Single framework; conversion issues; unnecessary | Interchange format; portable; optimization; interoperable |
| **Docker Containers** | Reproducible; isolated; portable; dependencies managed | Overkill; simple script; overhead | Containerized; reproducible; portable; standard |
| **Kubernetes** | Orchestration; scaling; microservices; production-grade | Small scale; simple deployment; overkill | Container orchestration; complex; production; scalable |
| **A/B Testing** | Compare models; gradual rollout; measure impact; safe deployment | Single model; no comparison; immediate rollout | Splits traffic; compares; safe; measured |
| **Shadow Deployment** | Test production; no impact; parallel; validation | Unnecessary cost; confidence high; simple A/B | Parallel running; no user impact; validation; safe |
| **Blue-Green Deployment** | Zero downtime; instant rollback; production; safe | Resource doubling; simple sufficient | Two environments; instant switch; safe; costly |
| **Canary Deployment** | Gradual rollout; monitor; safe; staged | Immediate needed; simple sufficient | Gradual percentage; monitored; safe; controlled |

---

## 24. Time Series Specific

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **ARIMA** | Univariate; stationary; linear; statistical; interpretable | Non-stationary; multivariate; non-linear; exogenous | Statistical; linear; requires stationarity; interpretable |
| **SARIMA** | Seasonal patterns; univariate; statistical; periodic | Non-seasonal; non-linear; multivariate | Seasonal ARIMA; periodic patterns; statistical |
| **Prophet** | Multiple seasonality; holidays; automatic; business time series | Short series; no seasonality; complex patterns | Facebook tool; automatic; handles missing; business-friendly |
| **LSTM for Time Series** | Long dependencies; non-linear; multivariate; complex patterns | Short series; simple linear; ARIMA sufficient | Deep learning; non-linear; multivariate; data-hungry |
| **GRU for Time Series** | Similar to LSTM; faster; moderate complexity; sequences | Very long dependencies; simple patterns | Faster than LSTM; sequence modeling; moderate complexity |
| **Temporal Convolutional Networks (TCN)** | Long sequences; parallel; dilated convolutions; efficient | Very long dependencies; recurrence needed | Convolutional; parallel; receptive field; efficient |
| **Exponential Smoothing** | Simple; weighted average; trend/seasonality; forecasting | Complex patterns; exogenous variables; sophisticated | Weighted average; simple; trend/seasonal; classical |
| **XGBoost for Time Series** | Feature engineering; lags; exogenous; competition | Temporal structure ignored; careful features | Treats as tabular; lag features; powerful but careful |
| **Wavelet Transform** | Frequency analysis; decomposition; multi-resolution; denoising | Time domain sufficient; interpretability; simple | Frequency decomposition; multi-scale; signal processing |
| **Differencing** | Remove trend; stationarity; preprocessing; simple | Already stationary; seasonal decomposition better | Removes trend; creates stationarity; preprocessing |
| **Seasonal Decomposition** | Separate components; understand patterns; preprocessing; visualization | Components not separable; simple trend | Trend + seasonal + residual; understanding; preprocessing |
| **Rolling Window Features** | Lag features; moving average; feature engineering; local patterns | Global patterns; computational cost; storage | Rolling statistics; lags; feature engineering; standard |
| **Autoregressive Models** | Past predicts future; lags important; simple; univariate | Exogenous variables; complex; multivariate | Uses own lags; simple; univariate; foundation |
| **VAR (Vector Autoregression)** | Multivariate; interdependencies; multiple series; Granger causality | Univariate sufficient; high dimensions; non-linear | Multivariate extension; interdependencies; linear |
| **State Space Models** | Latent states; Kalman filter; probabilistic; unobserved components | Observable sufficient; complexity unnecessary | Hidden states; probabilistic; flexible; sophisticated |

---

## 25. Ethical & Practical Considerations

| Consideration | When to Address | Key Questions | Mitigation Strategies |
|---------------|-----------------|---------------|----------------------|
| **Bias & Fairness** | Decisions affect people; protected attributes; hiring/lending/justice | Disparate impact? Protected groups? Fair across demographics? | Fairness metrics; bias testing; diverse data; regular audits |
| **Privacy** | Personal data; medical; financial; GDPR compliance | PII included? Consent obtained? Re-identification risk? | Anonymization; differential privacy; federated learning; secure computation |
| **Transparency** | Regulated; high-stakes; user trust; accountability | Can decisions be explained? Auditability? User understanding? | Interpretable models; documentation; audit trails; explainability tools |
| **Data Quality** | Always; GIGO principle; model reliability | Errors? Missing? Biased collection? Representative? | Validation; cleaning; documentation; quality metrics |
| **Adversarial Robustness** | Security-critical; adversarial attacks; financial/medical | Attack vectors? Adversarial examples? Robustness tested? | Adversarial training; input validation; robustness testing; monitoring |
| **Model Drift** | Production; changing distributions; online systems | Performance degrading? Distribution shift? Concept drift? | Monitoring; retraining; online learning; alerts |
| **Computational Cost** | Limited resources; environmental; scale | Carbon footprint? Training cost? Inference cost? Budget? | Efficient architectures; compression; cloud vs edge; green AI |
| **Interpretability Requirements** | Regulated; medical; finance; legal; high-stakes | Need explanations? Regulatory compliance? User trust? | Interpretable models; XAI techniques; documentation; human-in-loop |
| **Safety & Reliability** | Critical systems; medical; autonomous; safety-critical | Failure modes? Safety testing? Redundancy? Fallback? | Testing; validation; redundancy; human oversight; fail-safes |
| **Maintenance & Monitoring** | Production; long-term; evolving data | Who maintains? Monitoring? Retraining pipeline? Versioning? | MLOps; monitoring; documentation; versioning; retraining schedule |
| **Data Ownership & Licensing** | Commercial use; pretrained models; datasets; legal | License compatible? Data rights? Terms of use? Attribution? | Check licenses; legal review; proper attribution; own data when possible |
| **Explainability vs Performance Trade-off** | Regulated industries; need both accuracy and transparency | Can we sacrifice some accuracy? Post-hoc explainability sufficient? | Model selection; surrogate models; hybrid approaches; stakeholder communication |
| **Feedback Loops** | Recommendation systems; search; ranking; self-reinforcing | Does model affect future data? Feedback loops? Filter bubbles? | Diversity injection; exploration; monitoring; intervention strategies |
| **Generalization vs Memorization** | Privacy; training data leakage; overfitting | Memorizing training data? Can extract training examples? | Differential privacy; regularization; privacy audits; data minimization |

---

## 26. Domain-Specific Approaches

| Domain | Common Approaches | Special Considerations | When to Use Custom Solutions |
|--------|-------------------|------------------------|------------------------------|
| **Healthcare/Medical** | CNNs for imaging; RNNs for EHR; survival analysis; clinical NLP | Regulatory (FDA); interpretability critical; privacy (HIPAA); class imbalance; rare diseases | Novel biomarkers; rare conditions; multi-modal integration |
| **Finance** | Time series; gradient boosting; anomaly detection; risk modeling | Interpretability (regulations); adversarial attacks; non-stationarity; real-time | High-frequency trading; novel instruments; regulatory compliance |
| **Recommendation Systems** | Collaborative filtering; matrix factorization; neural collaborative filtering; transformers | Cold start; sparse data; diversity; feedback loops; scalability | Unique constraints; multi-stakeholder; context-rich |
| **Autonomous Vehicles** | CNN; sensor fusion; reinforcement learning; object detection; SLAM | Safety-critical; real-time; edge deployment; sensor noise; adversarial | Novel sensors; unique environments; safety requirements |
| **Natural Language** | Transformers; BERT/GPT; seq2seq; attention; embeddings | Ambiguity; context; multilingual; domain vocabulary; generation quality | Domain-specific language; low-resource languages; specialized tasks |
| **Manufacturing** | Anomaly detection; predictive maintenance; quality control; time series | Sensor data; imbalanced (failures rare); real-time; edge; domain physics | Unique processes; proprietary equipment; physics integration |
| **Retail/E-commerce** | Recommendation; demand forecasting; price optimization; customer segmentation | Seasonality; promotions; cold start; real-time; personalization | Unique business model; complex inventory; multi-channel |
| **Cybersecurity** | Anomaly detection; classification; sequence models; graph neural networks | Adversarial; imbalanced; evolving threats; real-time; false positives costly | Novel attacks; unique infrastructure; compliance requirements |
| **Agriculture** | Computer vision; remote sensing; IoT sensors; yield prediction | Limited data; environmental factors; seasonality; edge deployment | Specific crops; unique conditions; local optimization |
| **Energy** | Time series; optimization; forecasting; grid management | Physical constraints; safety; real-time; optimization; uncertainty | Renewable integration; grid topology; regulatory constraints |

---

## 27. Specialized Neural Network Layers

| Layer Type | When to Use | When to Avoid | Key Considerations |
|------------|-------------|---------------|-------------------|
| **Dense/Fully Connected** | Tabular; final layers; general purpose; simple | Images (before flatten); sequences; spatial structure | Connects all neurons; high parameters; general |
| **Convolutional (Conv2D)** | Images; spatial data; local patterns; translation invariance | Tabular; sequences; no spatial structure | Spatial patterns; parameter sharing; translation invariant |
| **Depthwise Separable Convolution** | Mobile; efficient CNNs; MobileNet; reduce parameters | Standard conv sufficient; accuracy critical; not mobile | Separates spatial/channel; efficient; mobile |
| **Dilated/Atrous Convolution** | Large receptive field; semantic segmentation; TCN; efficient | Standard conv works; small receptive field okay | Expands receptive field; efficient; no pooling |
| **Transposed Convolution (Deconvolution)** | Upsampling; segmentation; GANs; decoder | Simple upsampling sufficient; checkerboard artifacts issue | Learned upsampling; decoder; checkerboard artifacts possible |
| **Pooling (Max/Average)** | Downsample; translation invariance; reduce dimensions; CNNs | Need exact positions; stride sufficient; information loss issue | Reduces dimensions; max vs average; information loss |
| **Global Average Pooling** | Replace dense; reduce parameters; regularization; final CNN layer | Need spatial info; dense better; localization | Reduces to vector; replaces dense; regularization |
| **Recurrent (RNN/LSTM/GRU)** | Sequences; temporal; variable length; dependencies | Parallel needed; very long sequences; simple patterns | Sequential; temporal; vanishing gradients (RNN) |
| **Bidirectional RNN** | Context from both directions; NLP; full sequence available | Online/streaming; left-to-right sufficient; causal | Both directions; better context; not causal |
| **Attention Layer** | Long dependencies; transformers; focus mechanism; variable importance | Simple aggregation sufficient; attention unnecessary | Weighted importance; transformers; interpretable |
| **Self-Attention** | Relate positions; transformers; long-range; parallel | Local sufficient; sequential okay; computational cost | Position relationships; transformers; O(n²) complexity |
| **Cross-Attention** | Encoder-decoder; multimodal; query-key-value; alignment | Self-attention sufficient; single modality | Between sequences; alignment; multimodal |
| **Multi-Head Attention** | Different representation subspaces; transformers; diverse patterns | Single head sufficient; computational limits | Parallel attention; diverse patterns; transformers |
| **Embedding Layer** | Categorical; NLP; discrete inputs; learned representations | Continuous inputs; one-hot sufficient; very low cardinality | Learned embeddings; reduces dimensions; categorical |
| **Positional Encoding** | Transformers; sequence order; position information; no recurrence | RNN/LSTM (implicit); order unimportant | Adds position info; transformers; various schemes |
| **Dropout Layer** | Regularization; overfitting; training only; ensemble effect | Very small networks; underfitting; inference | Random dropping; regularization; training only |
| **Batch Normalization** | Stabilize training; speed convergence; reduce covariate shift; almost always | Very small batches; RNNs; variable batch size | Normalizes activations; speeds training; batch-dependent |
| **Layer Normalization** | RNNs; transformers; batch-independent; variable batch size | Batch norm works; CNNs (usually) | Across features; batch-independent; transformers |
| **Residual/Skip Connection** | Very deep; vanishing gradients; ResNet; signal preservation | Shallow networks; gradients fine | Adds input to output; enables depth; identity mapping |
| **Squeeze-and-Excitation** | Channel attention; CNNs; recalibrate features; boost performance | Simple sufficient; computational cost; minimal gains | Channel-wise attention; CNNs; adaptive recalibration |

---

## 28. Advanced Training Techniques

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Curriculum Learning** | Complex tasks; structured difficulty; gradual learning; improve convergence | Simple tasks; flat difficulty; unnecessary | Easy to hard; improves convergence; task-dependent |
| **Self-Supervised Learning** | Unlabeled data; pretraining; contrastive; representation learning | Plenty labels; simple supervised sufficient | No labels needed; pretraining; powerful |
| **Contrastive Learning** | Representation learning; SimCLR; vision; self-supervised | Supervised sufficient; no augmentations; small batch | Positive/negative pairs; powerful representations; self-supervised |
| **Meta-Learning** | Learn to learn; few-shot; quick adaptation; MAML | Plenty data per task; single task; computational cost | Task distribution; quick adaptation; sophisticated |
| **Multi-Task Learning** | Related tasks; shared representations; limited data per task; regularization | Unrelated tasks; negative transfer; simple sufficient | Shared layers; related tasks; careful balance |
| **Domain Adaptation** | Source ≠ target domain; transfer knowledge; distribution shift | Same domain; fine-tuning sufficient; very different | Bridge domains; reduce shift; various techniques |
| **Adversarial Training** | Robustness; security; GAN training; defend attacks | Clean data sufficient; computational cost; training instability | Robust to perturbations; expensive; improves robustness |
| **Mixup Training** | Data augmentation; regularization; smooth decision boundaries; limited data | Sufficient data; interpretability; linear invalid | Mixes samples; effective regularization; convex combinations |
| **Label Smoothing** | Prevent overconfidence; calibration; regularization; classification | Regression; need confidence; uncertainty quantification | Softens labels; improves calibration; regularization |
| **Cosine Similarity Loss** | Metric learning; face recognition; embeddings; angular distance | Standard classification sufficient; cross-entropy works | Angular distance; metric learning; normalized |
| **Gradient Clipping** | Exploding gradients; RNNs; stability; large gradients | No gradient issues; adds overhead | Clips gradient norm; stability; RNN common |
| **Gradient Accumulation** | Simulate large batch; memory limited; effective batch size | Memory sufficient; adds steps; unnecessary | Accumulates gradients; simulates large batch; memory-efficient |
| **Stochastic Weight Averaging (SWA)** | Better generalization; end of training; ensemble-like; low cost | Early stopping; online learning; quick training | Averages weights; improves generalization; cheap ensemble |
| **Sharpness-Aware Minimization (SAM)** | Flat minima; generalization; modern; state-of-the-art | Simple sufficient; computational overhead; overkill | Seeks flat minima; better generalization; modern |

---

## 29. Data Augmentation Techniques

| Technique | When to Use | When to Avoid | Key Considerations |
|-----------|-------------|---------------|-------------------|
| **Random Crop** | Images; translation invariance; limited data; standard | Full image needed; resolution critical | Crops randomly; translation invariance; standard |
| **Random Flip (Horizontal/Vertical)** | Images; symmetry valid; data augmentation; standard | Orientation matters (text); asymmetric objects | Flips image; symmetry; check validity |
| **Random Rotation** | Images; rotation invariance; limited data; standard | Orientation critical (text, documents); upright only | Rotates image; angle range; interpolation |
| **Color Jittering** | Images; color variations; lighting; robust to color | Color diagnostic (medical); exact color matters | Adjusts brightness/contrast/saturation; color robustness |
| **Random Erasing/Cutout** | Images; occlusion robustness; prevent overfitting; regularization | Small images; features critical; occlusion invalid | Random masking; occlusion robustness; regularization |
| **MixUp** | Images; linear interpolation; regularization; smooth boundaries | Interpretability; linear invalid; tabular (careful) | Mixes samples linearly; effective; augmentation |
| **CutMix** | Images; paste regions; harder than cutout; label smoothing | Simple sufficient; interpretability | Pastes patches; mixed labels; effective |
| **AutoAugment** | Automated policy; search augmentations; optimal; resources available | Simple sufficient; computational search cost; manual works | Searches policies; automated; expensive search; effective |
| **RandAugment** | Simpler than AutoAugment; fewer hyperparameters; effective; practical | AutoAugment works; simple sufficient | Simplified AutoAugment; practical; fewer hyperparameters |
| **Mosaic (YOLO)** | Object detection; multiple images; context; YOLO | Classification; single object; standard augmentation | Combines 4 images; object detection; context-rich |
| **Noise Injection** | Robustness; denoising; regularization; audio/images | Clean data critical; noise harmful; denoise first | Adds Gaussian/salt-pepper noise; robustness; regularization |
| **Synonym Replacement (NLP)** | Text; limited data; maintain semantics; simple | Change meaning; domain-specific terms; plenty data | Replaces words; maintains meaning; NLP augmentation |
| **Back Translation (NLP)** | Text; paraphrasing; limited data; translation available | Translation unavailable; computational cost; meaning shift | Translate and back; paraphrases; effective NLP |
| **Contextual Word Embeddings Replacement** | Text; context-aware; BERT-based; sophisticated | Simple sufficient; computational cost | BERT-based replacement; context-aware; sophisticated |
| **Time Stretching (Audio)** | Audio; speed variations; speech; music | Tempo critical; real-time; pitch matters | Changes speed; maintains pitch; audio augmentation |
| **Pitch Shifting (Audio)** | Audio; pitch variations; music; speaker invariance | Pitch diagnostic; frequency critical | Changes pitch; maintains tempo; audio augmentation |
| **SpecAugment (Audio)** | Speech recognition; spectrograms; state-of-the-art; audio | Raw audio; simple sufficient; not spectrograms | Masks spectrogram; frequency/time; speech recognition |
| **Elastic Deformation** | Medical imaging; handwriting; realistic deformations | Rigid objects; unrealistic; simple sufficient | Non-linear deformation; medical/handwriting; realistic |

---

## 30. Model Selection Decision Tree

### Based on Data Type:

**TABULAR DATA:**
- Small data (<10K rows) → Classical ML (Random Forest, XGBoost)
- Medium data (10K-1M) → Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Large data (>1M) → Deep Learning (MLP) or LightGBM
- Need interpretability → Linear models, Decision Trees, or XGBoost + SHAP
- High cardinality categoricals → CatBoost or embeddings

**IMAGE DATA:**
- Classification → CNN (ResNet, EfficientNet, ViT with transfer learning)
- Object detection → YOLO, Faster R-CNN, or modern transformers (DETR)
- Segmentation → U-Net, Mask R-CNN, or DeepLab
- Generation → GANs, Diffusion Models, VAE
- Small dataset → Transfer Learning (mandatory)
- Edge deployment → MobileNet, EfficientNet-Lite, or quantized models

**TEXT DATA:**
- Classification → Fine-tuned BERT, RoBERTa, or DistilBERT
- Generation → GPT variants, T5, or BART
- Translation → Transformer seq2seq, MarianMT
- Question Answering → BERT, ALBERT, or domain-specific models
- Named Entity Recognition → BiLSTM-CRF or transformer-based
- Limited resources → DistilBERT, ALBERT, or classical (TF-IDF + LogReg)
- Multilingual → mBERT, XLM-RoBERTa

**TIME SERIES:**
- Short term, simple → ARIMA, SARIMA, Exponential Smoothing
- Multivariate, complex → LSTM, GRU, Transformer
- Business forecasting → Prophet, or classical statistical
- Very long sequences → Temporal Convolutional Networks (TCN)
- With exogenous variables → SARIMAX, VAR, or XGBoost with lag features

**AUDIO:**
- Speech recognition → Whisper, Wav2Vec2, DeepSpeech
- Music → Specialized CNNs on spectrograms
- Classification → Mel-spectrograms + CNN
- Generation → WaveNet, diffusion models

**GRAPH DATA:**
- Node classification → GCN, GAT, GraphSAGE
- Link prediction → Graph embeddings, GNN
- Graph classification → Graph pooling + GNN
- Social networks → DeepWalk, Node2Vec, or GNN

---

## 31. Performance Optimization Checklist

### Data Level:
- [ ] Data quality checked (missing, outliers, errors)
- [ ] Appropriate preprocessing applied (scaling, encoding)
- [ ] Data augmentation if needed and appropriate
- [ ] Class imbalance addressed if present
- [ ] Feature engineering explored
- [ ] Dimensionality reduction considered if high-dimensional
- [ ] Train/val/test split stratified properly
- [ ] Data leakage checked and prevented

### Model Level:
- [ ] Started with simple baseline
- [ ] Appropriate architecture for data type
- [ ] Transfer learning used if applicable
- [ ] Hyperparameters tuned systematically
- [ ] Regularization applied (dropout, L1/L2, early stopping)
- [ ] Appropriate loss function selected
- [ ] Correct evaluation metrics chosen
- [ ] Cross-validation performed
- [ ] Ensemble methods considered

### Training Level:
- [ ] Appropriate optimizer selected (Adam/AdamW usually good)
- [ ] Learning rate tuned (warmup, scheduling)
- [ ] Batch size optimized for hardware
- [ ] Gradient clipping if needed
- [ ] Mixed precision training if available
- [ ] Monitoring training curves (loss, metrics)
- [ ] Overfitting/underfitting diagnosed
- [ ] Training time optimized

### Deployment Level:
- [ ] Model compressed if needed (quantization, pruning)
- [ ] Inference time optimized
- [ ] Error handling implemented
- [ ] Monitoring and logging set up
- [ ] A/B testing strategy planned
- [ ] Rollback strategy prepared
- [ ] Documentation complete
- [ ] Ethical considerations addressed

---

## 32. Common Pitfalls & Solutions

| Pitfall | Signs | Solution |
|---------|-------|----------|
| **Data Leakage** | Unrealistically high performance; fails in production | Careful train/test split; time-aware splits; check preprocessing |
| **Overfitting** | Train accuracy >> test accuracy; high variance | Regularization; more data; simpler model; early stopping |
| **Underfitting** | Low train and test accuracy; high bias | More complex model; more features; longer training; check data quality |
| **Wrong Metric** | Model optimizes but doesn't solve business problem | Align metrics with business goals; custom metrics; domain expertise |
| **Imbalanced Data Ignored** | High accuracy but poor minority class performance | Resampling; class weights; focal loss; different metrics (F1, PR-AUC) |
| **Not Using Transfer Learning** | Poor performance with limited data | Pretrained models; fine-tuning; feature extraction |
| **Ignoring Domain Knowledge** | Model learns spurious correlations; poor generalization | Domain expert collaboration; feature engineering; constraints |
| **No Baseline** | Don't know if model is actually good | Simple baseline first; random/majority; compare improvements |
| **Hyperparameter Neglect** | Using all defaults; suboptimal performance | Systematic tuning; grid/random/Bayesian search; learning rate critical |
| **Batch Size Too Small/Large** | Slow training / poor generalization | Experiment; typically 32-256; consider GPU memory; learning rate scaling |
| **Not Monitoring Training** | Wasted compute; missed issues; suboptimal stopping | Plot losses; validation metrics; early stopping; TensorBoard/wandb |
| **Data Preprocessing Inconsistency** | Train/test handled differently; production fails | Same pipeline for all; save preprocessing objects; careful with validation |
| **Ignoring Class Imbalance** | Model predicts majority class only | Address with techniques from Section 20 |
| **Feature Scaling Neglected** | Poor convergence; slow training; suboptimal | Standardize/normalize; especially for neural networks and distance-based |
| **Not Checking for Drift** | Production performance degrades over time | Monitor distributions; retrain periodically; online learning |
| **Insufficient Data** | Can't train complex model; overfitting | More data collection; augmentation; simpler models; transfer learning |
| **Wrong Problem Formulation** | Solving different problem than needed | Clarify requirements; business understanding; iterate with stakeholders |
| **Ignoring Computational Constraints** | Model too large/slow for deployment | Model compression; efficient architectures; edge optimization |
| **No Ablation Studies** | Don't know what components help | Systematic removal; measure impact; understand contributions |
| **Reproducibility Issues** | Can't replicate results; debugging hard | Set seeds; version control; log everything; document environment |

---

## 33. Quick Reference: Problem → Solution Mapping

| Problem Statement | Recommended Approach | Key Techniques |
|-------------------|---------------------|----------------|
| "Predict house prices" | Regression | XGBoost → feature engineering → ensemble |
| "Classify email as spam" | Binary classification | Logistic Regression / Naive Bayes / BERT |
| "Recognize handwritten digits" | Image classification | CNN → transfer learning (if small data) |
| "Detect objects in images" | Object detection | YOLO / Faster R-CNN → transfer learning |
| "Segment tumors in medical images" | Image segmentation | U-Net → data augmentation → transfer learning |
| "Generate realistic faces" | Image generation | GAN / Diffusion Models → large dataset |
| "Translate English to French" | Seq2seq translation | Transformer → pretrained (MarianMT) |
| "Classify movie reviews" | Text classification | Fine-tuned BERT → or TF-IDF + LogReg (baseline) |
| "Generate product descriptions" | Text generation | Fine-tuned GPT → or T5 |
| "Answer questions from documents" | Question answering | RAG + LLM → or fine-tuned BERT |
| "Recommend products to users" | Recommendation | Collaborative filtering → matrix factorization → neural CF |
| "Forecast stock prices" | Time series forecasting | LSTM / Prophet → feature engineering → ensemble |
| "Detect credit card fraud" | Anomaly detection | Isolation Forest → or supervised with class weights |
| "Cluster customers" | Clustering | K-Means → or hierarchical → dimensionality reduction first |
| "Reduce dimensionality for viz" | Dimensionality reduction | t-SNE / UMAP → PCA preprocessing |
| "Predict customer churn" | Binary classification | XGBoost → feature engineering → handle imbalance |
| "Recognize speech" | Speech recognition | Whisper → or Wav2Vec2 fine-tuned |
| "Play game optimally" | Reinforcement learning | DQN / PPO / AlphaZero → simulation environment |
| "Classify with very little data" | Few-shot learning | Transfer learning → meta-learning → data augmentation |
| "Detect anomalies in network" | Anomaly detection | Autoencoder → Isolation Forest → monitoring |
| "Rank search results" | Learning to rank | LambdaMART → NDCG metric → pairwise learning |
| "Predict protein structure" | Specialized deep learning | Domain-specific (AlphaFold-style) → transformers |
| "Generate music" | Audio generation | WaveNet → transformers → VAE |
| "Optimize route/schedule" | Optimization | Reinforcement learning → OR methods → genetic algorithms |

---

## 34. Resource Requirements Guide

| Model Type | Data Needed | Training Time | Compute | Memory | Expertise Level |
|------------|-------------|---------------|---------|--------|-----------------|
| Linear/Logistic Regression | 100s-1000s | Minutes | CPU | MB | Beginner |
| Decision Trees | 100s-1000s | Minutes | CPU | MB | Beginner |
| Random Forest | 1000s-10Ks | Minutes-Hours | CPU/GPU | MB-GB | Beginner |
| XGBoost/LightGBM | 1000s-100Ks | Minutes-Hours | CPU/GPU | GB | Intermediate |
| Simple Neural Network | 1000s-10Ks | Minutes-Hours | GPU | GB | Intermediate |
| CNN (from scratch) | 10Ks-100Ks | Hours-Days | GPU | GB | Intermediate |
| CNN (transfer learning) | 100s-1000s | Minutes-Hours | GPU | GB | Intermediate |
| LSTM/GRU | 1000s-10Ks | Hours-Days | GPU | GB | Intermediate |
| BERT (fine-tuning) | 1000s-10Ks | Hours-Days | GPU | GB-TB | Intermediate |
| GPT (fine-tuning) | 1000s-100Ks | Days-Weeks | GPU/TPU | GB-TB | Advanced |
| Large Vision Models | 100Ks-Ms | Days-Weeks | GPU/TPU | TB | Advanced |
| GAN | 10Ks-100Ks | Days-Weeks | GPU | GB-TB | Advanced |
| Diffusion Models | 100Ks-Ms | Weeks-Months | GPU/TPU | TB | Advanced |
| LLM (pretraining) | Ms-Bs | Months | GPU/TPU cluster | TB-PB | Expert |
| Reinforcement Learning | Varies | Hours-Weeks | GPU | GB | Advanced |
| Meta-Learning | Ms (across tasks) | Days-Weeks | GPU | GB-TB | Advanced |

**Legend:**
- Data: Number of samples needed
- Time: Typical training time
- Compute: CPU (basic), GPU (standard deep learning), TPU (large scale)
- Memory: RAM/VRAM requirements
- Expertise: Beginner (basics), Intermediate (solid understanding), Advanced (deep expertise), Expert (research level)

---

## 35. Tool & Framework Selection

| Framework/Tool | Best For | Avoid For | Learning Curve |
|----------------|----------|-----------|----------------|
| **scikit-learn** | Classical ML; prototyping; tabular data | Deep learning; production scale | Easy |
| **XGBoost** | Tabular data; competitions; feature importance | Images/text/audio; interpretability | Easy-Medium |
| **LightGBM** | Large tabular data; speed; efficiency | Small data; simple tasks | Easy-Medium |
| **TensorFlow/Keras** | Production; deployment; industry standard; mobile | Quick prototyping; research | Medium |
| **PyTorch** | Research; flexibility; dynamic graphs; experimentation | Quick deployment (improving); mobile | Medium |
| **Hugging Face Transformers** | NLP; pretrained models; transfer learning | Not NLP; from-scratch training | Easy-Medium |
| **FastAI** | Quick prototyping; education; best practices | Custom architectures; control | Easy |
| **JAX** | Research; high performance; functional programming | Beginners; standard tasks | Hard |
| **spaCy** | Production NLP; pipelines; efficiency | Research; latest models | Easy-Medium |
| **NLTK** | Education; NLP basics; linguistic analysis | Production; speed | Easy |
| **OpenCV** | Computer vision; image processing; real-time | Deep learning (use with DL framework) | Medium |
| **Pandas** | Data manipulation; tabular data; analysis | Large-scale (use Dask/Polars); arrays | Easy |
| **NumPy** | Numerical computing; arrays; linear algebra | High-level ML (use frameworks) | Easy-Medium |
| **Optuna** | Hyperparameter tuning; AutoML; optimization | Simple grid search | Easy |
| **Weights & Biases** | Experiment tracking; collaboration; monitoring | Simple projects; offline-only | Easy |
| **MLflow** | Experiment tracking; model registry; deployment | Simple projects | Medium |
| **Ray/Ray Tune** | Distributed; hyperparameter tuning; scaling | Single machine; simple tuning | Medium-Hard |
| **TensorBoard** | Training visualization; TensorFlow/PyTorch | Non-DL; complex tracking needs | Easy |
| **DVC** | Data versioning; ML pipelines; reproducibility | Simple projects; no data versioning needs | Medium |
| **Apache Spark MLlib** | Big data; distributed; Hadoop ecosystem | Small data; deep learning | Medium-Hard |

---

## 36. When to Use Neural Networks vs Classical ML

### Use Neural Networks When:
- ✅ Large amounts of data available (10K+ samples typically)
- ✅ Complex, non-linear patterns and interactions
- ✅ Unstructured data (images, text, audio, video)
- ✅ Feature engineering is difficult or unknown
- ✅ Representation learning is beneficial
- ✅ Transfer learning applicable
- ✅ State-of-the-art performance needed
- ✅ Sufficient computational resources
- ✅ End-to-end learning desired

### Use Classical ML When:
- ✅ Limited data (<10K samples often)
- ✅ Tabular/structured data
- ✅ Interpretability is critical
- ✅ Quick prototyping needed
- ✅ Limited computational resources
- ✅ Feature engineering straightforward
- ✅ Linear or simple non-linear relationships
- ✅ Real-time inference with constraints
- ✅ Proven classical methods work well
- ✅ Regulatory/explainability requirements

---

## 37. Final Decision Framework

```
START
  ↓
What type of data?
  ├─ Tabular → Classical ML / Gradient Boosting
  ├─ Images → CNN / Vision Transformers
  ├─ Text → Transformers (BERT/GPT)
  ├─ Audio → Spectrograms + CNN / Whisper
  ├─ Time Series → Statistical / LSTM / Prophet
  ├─ Graph → GNN
  └─ Mixed → Multimodal models
  ↓
How much data?
  ├─ Very little (<1K) → Transfer Learning / Few-shot / Data Augmentation
  ├─ Small (1K-10K) → Classical ML / Transfer Learning
  ├─ Medium (10K-100K) → Deep Learning possible
  ├─ Large (100K-1M) → Deep Learning recommended
  └─ Very large (>1M) → Deep Learning / Distributed Training
  ↓
What's the goal?
  ├─ Supervised (labels available)
  │   ├─ Classification → See model selection by data type
  │   ├─ Regression → XGBoost / Neural Networks
  │   └─ Ranking → LambdaMART / Pairwise learning
  ├─ Unsupervised (no labels)
  │   ├─ Clustering → K-Means / DBSCAN / Hierarchical
  │   ├─ Dimensionality Reduction → PCA / t-SNE / UMAP
  │   └─ Anomaly Detection → Isolation Forest / Autoencoder
  ├─ Semi-Supervised → Self-supervised pretraining + fine-tuning
  ├─ Reinforcement Learning → DQN / PPO / AlphaZero
  └─ Generation → GANs / Diffusion / VAE / LLMs
  ↓
What are the constraints?
  ├─ Interpretability required → Linear / Trees / SHAP
  ├─ Real-time latency critical → Simple models / Optimization
  ├─ Limited compute → Classical ML / Efficient architectures
  ├─ Edge deployment → Compression / MobileNet / Quantization
  ├─ Privacy concerns → Federated / Differential Privacy
  └─ Cost sensitive → Classical ML / Smaller models
  ↓
IMPLEMENT → EVALUATE → ITERATE
```

---

## Key Takeaways

1. **Start Simple**: Always begin with a baseline model before adding complexity
2. **Know Your Data**: Data type and size drive most decisions
3. **Match Problem to Method**: Use this guide to align your specific problem with appropriate techniques
4. **Consider Constraints**: Real-world constraints (time, compute, interpretability) often override "best" solutions
5. **Iterate**: ML is experimental—try multiple approaches and measure systematically
6. **Monitor Production**: Deployment is not the end—monitor, retrain, and improve
7. **Ethics First**: Always consider bias, fairness, privacy, and societal impact
8. **Stay Updated**: AI/ML evolves rapidly—this guide is a snapshot, keep learning

---

**Remember**: The "best" approach depends on your specific context. Use this guide as a structured starting point, not rigid rules. Experiment, measure, and adapt based on your results!


