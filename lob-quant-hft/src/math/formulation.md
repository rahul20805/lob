# Mathematical Formulation

## 1. LOB Mid-Price & Label Generation

Given $N$ levels of the order book with best bid $b_1$ and best ask $a_1$:

$$m_t = \frac{a_1^t + b_1^t}{2}$$

**Smoothed mid-price** over future horizon $H$:

$$\bar{m}_{t+H} = \frac{1}{H} \sum_{i=1}^{H} m_{t+i}$$

**3-class label**:

$$y_t = \begin{cases}
2 & \text{if } \frac{\bar{m}_{t+H} - m_t}{m_t} > \alpha \\
0 & \text{if } \frac{\bar{m}_{t+H} - m_t}{m_t} < -\alpha \\
1 & \text{otherwise}
\end{cases}$$

where $\alpha = 0.0002$ (2 bps threshold).

---

## 2. DeepLOB Architecture

**Input**: $\mathbf{X}_t \in \mathbb{R}^{T \times 4L}$ where $T$ = window size, $L$ = LOB depth.

Each LOB row: $[a_p^1, a_v^1, b_p^1, b_v^1, \ldots, a_p^L, a_v^L, b_p^L, b_v^L]$.

**CNN feature extraction** (3 blocks):

$$\mathbf{h}^{(1)} = \text{BN}(\text{LeakyReLU}(\text{Conv2D}(\mathbf{X}_t, W_1)))$$

**Inception module** (multi-scale):

$$\text{Inception}(\mathbf{h}) = \left[\text{Conv}_{1\times1}(\mathbf{h}) \;\|\; \text{Conv}_{3\times1}(\mathbf{h}) \;\|\; \text{Conv}_{5\times1}(\mathbf{h}) \;\|\; \text{MaxPool}(\mathbf{h})\right]$$

**Bidirectional LSTM**:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_\text{fwd}(\mathbf{h}_{t-1}, \mathbf{x}_t), \quad
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_\text{bwd}(\mathbf{h}_{t+1}, \mathbf{x}_t)$$

$$\mathbf{o} = [\overrightarrow{\mathbf{h}}_T \;\|\; \overleftarrow{\mathbf{h}}_1]$$

**Classifier**:

$$\hat{y} = \text{softmax}(W_c \mathbf{o} + b_c)$$

---

## 3. Focal Loss

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t$$

where $p_t = P(y = c^* \mid \mathbf{x})$ is the predicted probability of the true class, $\alpha_t$ is a class-weighting factor, and $\gamma \geq 0$ is the focusing parameter.

For $\gamma = 0$: reduces to standard cross-entropy.  
For $\gamma = 2$: down-weights well-classified examples by factor $(1-p_t)^2$.

---

## 4. Order Flow Imbalance (OFI)

At level $\ell$, between ticks $t-1$ and $t$:

$$\Delta \text{Bid}_\ell = \begin{cases}
  b_v^\ell(t) & \text{if } b_p^\ell(t) > b_p^\ell(t-1) \\
  b_v^\ell(t) - b_v^\ell(t-1) & \text{if } b_p^\ell(t) = b_p^\ell(t-1) \\
  -b_v^\ell(t-1) & \text{if } b_p^\ell(t) < b_p^\ell(t-1)
\end{cases}$$

Similarly for $\Delta \text{Ask}_\ell$. Then:

$$\text{OFI}_\ell(t) = \Delta \text{Bid}_\ell(t) - \Delta \text{Ask}_\ell(t)$$

---

## 5. Avellaneda–Stoikov Market Making

**Stochastic mid-price** (arithmetic Brownian motion):

$$dS_t = \sigma \, dW_t$$

**Inventory dynamics** (inventory $q_t$ changes by $\pm$ fill size):

$$dq_t = dN_t^b - dN_t^a$$

where $N_t^b$, $N_t^a$ are buy/sell fill Poisson processes.

**Reservation price** (risk-adjusted mid):

$$r(S, q, t) = S - q\,\gamma\,\sigma^2(T - t)$$

**Optimal bid and ask offsets** from reservation price:

$$\delta^a = \delta^b = \frac{1}{2}\left[\gamma\,\sigma^2(T-t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right)\right]$$

so optimal quotes are:

$$a^* = r + \delta^a, \quad b^* = r - \delta^b$$

**Fill probability** per time step $\Delta t$:

$$P(\text{fill} \mid \delta) \approx e^{-\kappa\,\delta} \cdot \Delta t$$

---

## 6. ML-Augmented Quoting

Let $\boldsymbol{\pi} = [\pi_{\downarrow}, \pi_{\text{stat}}, \pi_{\uparrow}]$ be the softmax output of DeepLOB.

**Directional signal**:

$$\phi = \pi_{\uparrow} - \pi_{\downarrow} \in [-1, 1]$$

**Signal-adjusted reservation price**:

$$\tilde{r} = r + \lambda_s \cdot \phi \cdot \delta$$

where $\lambda_s \in [0,1]$ is the signal weight hyperparameter.

---

## 7. Performance Metrics

**Sharpe Ratio**:

$$\text{SR} = \frac{\mathbb{E}[r_t - r_f]}{\text{Std}(r_t - r_f)} \cdot \sqrt{N_{\text{annual}}}$$

**Sortino Ratio** (penalises only downside):

$$\text{SoR} = \frac{\mathbb{E}[r_t - r_f]}{\text{Std}(\min(r_t - r_f, 0))} \cdot \sqrt{N_{\text{annual}}}$$

**Maximum Drawdown**:

$$\text{MDD} = \min_{t} \left(\text{PnL}(t) - \max_{s \leq t} \text{PnL}(s)\right)$$

**Calmar Ratio**:

$$\text{CR} = \frac{\text{Annualised Return}}{|\text{MDD}|}$$
