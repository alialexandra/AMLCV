# Task 1  Theory 


a. Based on the CLIP paper, research which specific models have been 
used for the image encoder and the text encoder. 
(If you cannot find exact architectures, try to find at least names 
of components.) (Note: If you find conflicting information, 
we are curious about your sources!)

From the paper, under training section. 


- Image Encoder: 

They used two architectures: ResNet and Vision Transformers 
ResNet:
    - ResNet-50, ResNet-101, and 3 more scaled versions 
RN50x4, RN50x16, RN50x64

Vision Transformer (ViT):
    - ViT-B/32, a ViT-B/16, and a ViT-L/14. We train all
 models for 32 epochs

- Text Encoder: 

For text encoder they used: 

GPT-2 style Transformer (specifically the smaller variant)

The text encoder is a Transformer (Vaswani et al., 2017)
 with the architecture modifications described in Radford
 et al. (2019). As a base size we use a 63M-parameter 12
layer 512-wide model with 8 attention heads. The trans
former operates on a lower-cased byte pair encoding (BPE)
 representation of the text with a 49,152 vocab size (Sen
nrich et al., 2015). For computational efficiency, the max
 sequence length was capped at 76

Key differences from original Transformer:

- Layer normalization moved to the input of each sub-layer (pre-norm)
- Additional layer normalization after the final self-attention layer
- Learned positional embeddings instead of sinusoidal
- GPT-2 weight initialization scheme

Raw Text -> Lowercase -> BPE Tokenization -> [Token Embeddings] -> Transformer -> [EOS] Feature




b. Based on the CLIP paper, write out the equations step-by-step 
to match the code presented in Figure 3. (You may want to read 
some code documentation, or use additional material.)



## **Task 1(b): Step-by-Step Equations (matching CLIP Figure 3 and training diagram)**

The CLIP model jointly trains an image encoder and a text encoder to align their embeddings in a shared latent space using a **contrastive objective**.
Each image–text pair ((I_i, T_i)) is treated as a positive match, while all other pairs in the batch act as negatives.


## Task 1(b): Step-by-step equations (matching CLIP Figure 3)

The CLIP model aligns image and text embeddings using a **contrastive InfoNCE loss**.  
For a batch of *N* paired samples \((I_i, T_i)\):

---

### 1️ Encode and normalize

Each encoder produces feature vectors:

$$
\mathbf{f}^{\text{img}}_i = f_{\text{img}}(I_i), \qquad
\mathbf{f}^{\text{txt}}_i = f_{\text{txt}}(T_i)
$$

Project to a shared space and L2-normalize:

$$
\mathbf{z}^{\text{img}}_i =
\frac{W_i^\top \mathbf{f}^{\text{img}}_i}
{\|W_i^\top \mathbf{f}^{\text{img}}_i\|}, \qquad
\mathbf{z}^{\text{txt}}_i =
\frac{W_t^\top \mathbf{f}^{\text{txt}}_i}
{\|W_t^\top \mathbf{f}^{\text{txt}}_i\|}
$$

Cosine similarity between image *i* and text *j*:

$$
s_{ij} = \cos(\theta_{ij})
      = \frac{\mathbf{z}^{\text{img}}_i \cdot \mathbf{z}^{\text{txt}}_j}
             {\|\mathbf{z}^{\text{img}}_i\|\,\|\mathbf{z}^{\text{txt}}_j\|}
$$

---

### 2️ Apply temperature scaling

A learned temperature parameter \(\tau = \exp(t)\) rescales similarities:

$$
\text{logits}_{ij} = \tau\, s_{ij}
$$

---

### 3️ Normalize via softmax

For each image *i*, compute probabilities over all text samples:

$$
p_{ij}^{\text{img}\rightarrow\text{txt}}
= \frac{\exp(\text{logits}_{ij})}
       {\sum_{k=1}^{N}\exp(\text{logits}_{ik})}
$$

Similarly, for each text *j*:

$$
p_{ij}^{\text{txt}\rightarrow\text{img}}
= \frac{\exp(\text{logits}_{ij})}
       {\sum_{k=1}^{N}\exp(\text{logits}_{kj})}
$$

---

### 4️ Contrastive (InfoNCE) cross-entropy loss

Each image’s correct text index is its own pair \((j=i)\).  
The loss is computed symmetrically:

$$
\mathcal{L}_{\text{img}}
= -\frac{1}{N}\sum_{i=1}^{N}
   \log \frac{\exp(\text{logits}_{ii})}
             {\sum_{j=1}^{N}\exp(\text{logits}_{ij})}
$$

$$
\mathcal{L}_{\text{txt}}
= -\frac{1}{N}\sum_{i=1}^{N}
   \log \frac{\exp(\text{logits}_{ii})}
             {\sum_{j=1}^{N}\exp(\text{logits}_{ji})}
$$

$$
\boxed{\mathcal{L}_{\text{CLIP}}
   = \tfrac{1}{2}(\mathcal{L}_{\text{img}}+\mathcal{L}_{\text{txt}})}
$$

---

### 5 Parameter definitions and dimensions


| **Symbol** | **Meaning** | **Dimensions / Type** |
|:------------|:-------------|:-----------------------|
| \( I_i \) | Input image | — |
| \( T_i \) | Input text prompt | — |
| \( f_{\text{img}} \) | Image encoder (ResNet / ViT) | maps image → \( \mathbb{R}^{d_i} \) |
| \( f_{\text{txt}} \) | Text encoder (Transformer) | maps text → \( \mathbb{R}^{d_t} \) |
| \( W_i,\, W_t \) | Linear projection matrices | \( W_i \in \mathbb{R}^{d_i \times d_e} \), \( W_t \in \mathbb{R}^{d_t \times d_e} \) |
| \( \mathbf{z}^{\text{img}}_i,\, \mathbf{z}^{\text{txt}}_i \) | Normalized image / text embeddings | \( \mathbb{R}^{d_e} \) |
| \( s_{ij} \) | Cosine similarity between image *i* and text *j* | scalar |
| \( \tau = \exp(t) \) | Learned temperature scaling parameter | scalar > 0 |
| \( N \) | Batch size (number of paired examples) | integer |
| \( \mathcal{L}_{\text{CLIP}} \) | Total symmetric InfoNCE loss | scalar |

---


- The model **maximizes similarity** of matching pairs and **minimizes** others.  
- The **softmax** turns similarities into probabilities, and **cross-entropy** pushes the correct match’s probability highest.  
- The **temperature τ** adjusts how sharply the model distinguishes hard negatives.




c. Explain why no explicit class labels are required in CLIP training.


Explicit class labels are not required in CLIP training because 
it uses natural language captions as a form of supervision in 
a self-supervised, contrastive learning framework. 
Instead of manually annotated labels for predefined categories, 
the supervision signal comes naturally from the vast number of 
image-text pairs collected from the internet, 
basically CLIP avoids the need for manual labeling by using 
a different kind of "label" that is already present in the data itself.



d. Explain why FID uses the Fréchet (2-Wasserstein) 
distance between Gaussians instead of the KL divergence. 
(Can you find references?)

FID uses the Fréchet (2-Wasserstein) distance instead of KL 
divergence primarily because the Fréchet distance is 
a true metric, and provides stable and meaningful 
gradients even when the distributions being compared have 
little or no overlap, considering mean and covariance.
KL divergence is a divergence, not a metric. It measures information 
loss but doesn't give a reliable "distance" in the geometric sense. 
FID is a true metric that properly captures both the location and shape 
differences between distributions

### References

- Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
  cite: "Unlike the Inception Score, FID is a true metric and captures the similarity of the generated samples to the real ones better than other metrics."

 - Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.)
   "It is not a true distance between distributions since it is not symmetric and does not satisfy the triangle inequality. Nevertheless, it is often useful to think of relative entropy as a 'distance' between distributions."

- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
  "The Kullback-Leibler divergence is not a symmetrical quantity, that is to say, KL(p∥q) ≢ KL(q∥p)."


e. Reflect critically on the ethical implications of large-scale text-to-image models such as Stable Diffusion, DALL·E, etc.


Risks: What are the possible harms related to bias, misinformation, intellectual property, privacy, or environmental impact?

Governance: How can open-source publication and licensing (e.g., CreativeML Open RAIL license) balance innovation with responsible use?

Pick 2 from the following guiding questions:

Benefits: In what ways can generative image models positively impact art, accessibility, education, or industry?
Risks: What are the possible harms related to bias, misinformation, intellectual property, privacy, or environmental impact?
Governance: How can open-source publication and licensing (e.g., CreativeML Open RAIL license) balance innovation with responsible use?
Future direction: What mitigation strategies or policy guidelines can reduce negative effects without halting progress?


## Ethical Implications of Text-to-Image Models

### Benefits

- Creates visual aids for people with disabilities
- Provides affordable graphic design for underserved communities
- Generates custom visualizations for complex concepts
- Supports interactive and personalized learning experiences
- Accelerates prototyping and content creation workflows
- Reduces costs for small businesses and startups


### Risks


- Memorizes and may leak personal data from training sets
- Enables non-consensual image generation of real people
- Amplifies societal biases from training data
- Enables mass production of convincing deepfakes
- Trains on copyrighted works without compensation
- Threatens livelihoods of artists and creatives
- Consumes massive energy resources for training/inference
- Generates significant carbon footprint and e-waste


### Governance

- Explicitly prohibits illegal and harmful applications
- Balances open access with usage restrictions
- Difficult to enforce restrictions across jurisdictions
- Limited protection for original content creators

### Future Directions

- Develop bias detection and content watermarking tools
- Implement energy-efficient model architectures
- Establish international standards and disclosure requirements
- Promote digital literacy about AI-generated content
- Create compensation systems for training data contributors
- Support workforce transition for displaced creative professionals