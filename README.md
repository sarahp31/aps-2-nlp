# aps-2-nlp


## Step 1: find embeddings

* **A description of your dataset**

    The dataset comprises job listings from multiple companies, including titles, locations, descriptions, and URLs for each listing. Each entry represents a unique job, with fields that store essential details about the position.

* **A description of the process to generate embeddings, including the neural network topology and hyperparameters**
    
    We used OpenAI's text-embedding-3-large model to generate embeddings for job postings. Each job title and description became a 3072-dimensional vector. We then designed a denoising autoencoder, where the encoder reduces the input size from 3072 to 512 through layers of sizes 2048, 1024, and 512. The decoder reconstructs the original size through layers of sizes 1024, 2048, and 3072. The neural network’s structure is shown in the figure below.

    ### Neural Network Topology

    ```mermaid
    flowchart LR
    Input["3072\nInput Layer"]
    E1["2048"]
    E2["1024"]
    Latent["512\nLatent Space"]
    D1["1024"]
    D2["2048"]
    Output["3072\nOutput Layer"]

    Input --> E1 --> E2 --> Latent
    Latent --> D1 --> D2 --> Output

    ```
    ### Hyperparameters

    - **Input Size**: 3072
    - **Encoder Layer Sizes**: [3072, 2048, 1024, 512]
    - **Decoder Layer Sizes**: [512, 1024, 2048, 3072]
    - **Activation Function**: ReLU for each layer except the output layer
    - **Loss Function**: Mean Squared Error (MSE), as it’s typical for reconstruction tasks
    - **Optimizer**: Adam optimizer with:
    - Learning rate (`lr`): \(1 \times 10^{-3}\)


* **A description of the training process, including a description of the loss function and why it makes sense to your problem**

    We trained a denoising autoencoder to reconstruct original embeddings from corrupted inputs, using Gaussian noise to create these inputs. The Mean Squared Error (MSE) loss function, defined as

    ```math
    mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \hat{X}_i)^2
    ```

    encourages the model to capture essential features and reduce noise. This approach refined pre-trained embeddings to our dataset, enhancing relevance by focusing on significant features.

    -- remover ? --
    where $X_i$ is the original embedding and $\hat{X}_i$ is the reconstructed embedding. Minimizing this loss function encourages the model to learn robust features that capture essential information while reducing noise.


    We adapted pre-trained embeddings to our specific dataset, enhancing their relevance and reducing dimensionality. The use of a denoising autoencoder allowed us to refine the embeddings by learning to recover original embeddings from noisy inputs, thereby capturing the most significant features for our problem domain.
    -- remover --


## Step 2: visualize your embeddings

* **Figures showing the embeddings for pre-trained and tuned embeddings**

    ![Pre-trained t-SNE Embeddings](./images/Pre_trained_Embeddings.png)

    Figure 1: t-SNE visualization of pre-trained embeddings
    ![Refined t-SNE Embeddings](./images/Refined_Embeddings.png)

    Figure 2: t-SNE visualization of refined embeddings

* **Discussion on what can be seen in each figure, clusters and so on**

    The pre-trained t-SNE embeddings display clearer, denser clusters with gaps, suggesting distinct themes based on job titles and descriptions. In contrast, the refined embeddings are more dispersed, likely due to changes in their semantic structure.

    -- remover -- The pre-trained and refined t-SNE embeddings display clusters, but the pre-trained embeddings show clearer, denser groupings, with gaps between some clusters. This separation suggests that the pre-trained embeddings may capture distinct themes based on job titles and descriptions, while the refined embeddings appear more dispersed, possibly due to changes in their semantic structure. -- remover --


## Step 3: test the search system


