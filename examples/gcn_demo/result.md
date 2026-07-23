# AlphaEvolve GCN Optimization Results

This document summarizes the results of the evolutionary search performed by AlphaEvolve to optimize the Graph Convolutional Network (GCN) architecture for molecular solubility prediction (`adme_sol` dataset).

---

## Experiment Summary

*   **Experiment ID**: `projects/181550378089/locations/global/collections/default_collection/engines/alpha-evolve-protein-folding/sessions/12248980981927161488/alphaEvolveExperiments/16631855482071082748`
*   **Search Budget**: 20 candidates evaluated
*   **Pacing**: Concurrency = 2
*   **Evaluation Environment**: Cloud Run (1x NVIDIA L4 GPU, 4 vCPUs, 16GiB Memory)
*   **Training Pacing**: 100 epochs per candidate
*   **Primary Metric**: PearsonR correlation (higher is better)
*   **Total Wall-Clock Time**: 20 minutes 9 seconds
*   **Avg. Time per Candidate**: ~2 minutes (includes container startup and repository cloning)
*   **Verified Costs (from Google Cloud BigQuery Billing Export)**:
    *   **Cloud Run GPU Evaluator**: **`$0.5671`** (NVIDIA L4 GPU: $0.3598, vCPU: $0.1456, Memory: $0.0617)
    *   **AlphaEvolve API**: **`$0.4216`** (Gemini 3.5 Flash Output Tokens: $0.3297, Input Tokens: $0.0919)
    *   **Total Experiment Cost**: **`$0.9887`** (approx **~$0.99**)

---

## Evolution Progress

Here is the chronological log of the evaluations. The search successfully navigated through minor regressions and code errors to find a highly optimized architecture:

| Candidate # | Program ID | PearsonR | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `15919755000362051439` | **0.3300** | Success | Seed program (initial optimized config from CPU proxy) |
| 1 | `15110819558846743338` | 0.3712 | Success | Improved |
| 2 | `15110819558846745192` | 0.3324 | Success | |
| 3 | `15110819558846746442` | 0.3006 | Success | Regression |
| 4 | `15110819558846743553` | 0.3031 | Success | |
| 5 | `15110819558846744027` | 0.3975 | Success | Improved |
| 6 | `15110819558846744501` | 0.2935 | Success | |
| 7 | `15110819558846743854` | 0.3718 | Success | |
| 8 | `15110819558846743207` | 0.3424 | Success | |
| 9 | `15110819558846745190` | 0.4276 | Success | Improved |
| 10 | `15110819558846745319` | 0.3709 | Success | |
| 11 | `15110819558846743206` | 0.4102 | Success | |
| 12 | `15110819558846745534` | 0.4256 | Success | |
| 13 | `15110819558846743421` | **0.4324** | Success | **Best Program** |
| 14 | `15110819558846746137` | 0.4231 | Success | |
| 15 | `15110819558846746611` | -1.00e+12 | **Failed** | Runtime/Compile Error |
| 16 | `15110819558846744110` | 0.3088 | Success | |
| 17 | `15110819558846745274` | 0.4239 | Success | |
| 18 | `15110819558846745748` | 0.3856 | Success | |
| 19 | `15110819558846744756` | 0.3295 | Success | |
| 20 | `15110819558846743031` | 0.4021 | Success | |

---

## Comparison: Original vs. Best Evolved Program

Below is a comparison highlighting the architectural improvements made by AlphaEvolve.

### Performance Comparison

| Metric | Original GCN (Baseline) | Best Evolved GCN (100 Epochs) | Best Evolved GCN (501 Epochs) | Max Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **PearsonR (adme_sol)** | **0.3561** (after 501 epochs) | **0.4324** (after 100 epochs) | **0.4549** (after 501 epochs) | **+27.7%** (vs Baseline at 501 epochs) |

### Key Architectural Improvements

AlphaEvolve transformed the GCN model from a basic message-passing network into an advanced architecture:

1.  **Multi-Scale & Global Pooling**:
    *   *Original*: Relied on a single pooling method (defaulting to `global_mean_pool`).
    *   *Evolved*: Dynamically computes **mean, max, and sum pooling** for every layer, concatenating them (`torch.cat([h_mean, h_max, h_sum], dim=-1)`). The inclusion of sum pooling is chemically meaningful as it captures molecular size/weight, which correlates with solubility.
2.  **Jumping Knowledge (JK) Layer Aggregation**:
    *   *Original*: Only used the final layer's output (`h_list[-1]`) for graph representation.
    *   *Evolved*: Aggregates feature representations across different depths by concatenating the pooled features of the **initial, middle, and last layer groups** (`pooled_features[0]`, `pooled_features[num_layer//2]`, `pooled_features[-1]`). This preserves multi-scale graph details.
3.  **Modern Activations & Layer Normalization**:
    *   *Original*: Used standard `ReLU` and basic linear layer mapping.
    *   *Evolved*: Replaced `ReLU` with **`GELU`** (Gaussian Error Linear Unit) inside layers and **`SiLU`** (Sigmoid Linear Unit) in the prediction head. Added **`LayerNorm`** and projection layers (`jk_proj`, `feat_lin` MLP block) to stabilize deep propagation.
4.  **Deeper Residual Prediction Head**:
    *   *Original*: Simple 2-layer MLP head.
    *   *Evolved*: Implemented a deeper prediction head with a **residual skip connection** around the hidden layer (`h = h + F.silu(self.pred_head1(h))`) and Layer Normalization, preventing gradient degradation in the final regression mapping.
5.  **Sigmoid Removal for Regression**:
    *   *Original*: Ended the forward pass with `torch.sigmoid(output)`, which restricted outputs to `[0, 1]`.
    *   *Evolved*: Removed the sigmoid activation, allowing the network to output arbitrary real numbers suitable for regression targets (solubility values can be negative or > 1).

---

### Code Diff

```diff
--- examples/gcn_demo/gcn_original.py
+++ examples/gcn_demo/gcn_optimized.py
@@ -39,7 +39,8 @@
-        self.feat_lin = nn.Linear(self.emb_dim, feat_dim)
+        # JK representation size: 3 (mean, max, sum) * emb_dim * 3 layers selected (initial, middle, last)
+        self.jk_proj = nn.Linear(self.emb_dim * 3 * 3, self.emb_dim * 2)
+
+        # Use a combination of mean and max pooling to capture diverse graph properties
+        self.feat_lin = nn.Sequential(
+            nn.Linear(self.emb_dim * 2, feat_dim),
+            nn.LayerNorm(feat_dim),
+            nn.GELU()
+        )
 
-        self.pred_head = nn.Sequential(
-            nn.Linear(feat_dim, feat_dim // 2),
-            nn.ReLU(),
-            nn.Linear(feat_dim // 2, out),
-        )
+        # A deeper prediction head with LayerNorm and residual connection to stabilize learning
+        self.pred_head1 = nn.Sequential(
+            nn.Linear(feat_dim, feat_dim),
+            nn.LayerNorm(feat_dim),
+            nn.GELU(),
+            nn.Dropout(drop_ratio),
+        )
+        self.pred_head2 = nn.Linear(feat_dim, out)
 
     def forward(self, data):
@@ -53,9 +59,9 @@
         h_list = [x]
         for layer in range(self.num_layer):
             h = self.convs[layer](h_list[layer], edge_index)
             h = self.batch_norms[layer](h)
-            if layer == self.num_layer - 1:
-                h = F.dropout(h, self.drop_ratio, training=self.training)
-            else:
-                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
+            h = F.gelu(h)
+            h = F.dropout(h, self.drop_ratio, training=self.training)
+            # Residual skip connection with scaling to stabilize deep propagation
+            h = h + h_list[layer]
             h_list.append(h)
 
-        node_representation = h_list[-1]
-        h = self.pool(node_representation, batch)
-        h = self.feat_lin(h)
-        output = self.pred_head(h)
-        output = torch.sigmoid(output)
+        # Jump Knowledge (JK) layer aggregation: Pool each layer's representations.
+        # We include global sum pooling to capture molecular size/weight, which is critical for solubility.
+        pooled_features = []
+        for h_layer in h_list:
+            h_mean = global_mean_pool(h_layer, batch)
+            h_max = global_max_pool(h_layer, batch)
+            h_sum = global_add_pool(h_layer, batch)
+            pooled_features.append(torch.cat([h_mean, h_max, h_sum], dim=-1))
+        
+        # Concatenate final representation from initial, middle, and last layer groups to preserve diverse scales
+        # and project back to the expected dimension.
+        h_jk = torch.cat([pooled_features[0], pooled_features[self.num_layer // 2], pooled_features[-1]], dim=-1)
+        
+        h = self.jk_proj(h_jk)
+        h = self.feat_lin(h)
+        
+        # Residual MLP head with SiLU activation
+        h = h + F.silu(self.pred_head1(h))
+        output = self.pred_head2(h)
 
         return output
```
