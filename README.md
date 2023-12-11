<h2>Accelerating Defect Prediction in Semiconductors Using Graph Neural Networks</h2>
<p>First principles computations reliably predict the energetics of point defects in semiconductors, but are constrained by the expense of using large supercells and advanced levels of theory. Machine learning models trained on computational data, especially ones that sufficiently encode defect coordination environments, can be used to accelerate defect predictions.</p>
<p>Here, we develop a framework for the prediction and screening of native defects and functional impurities in a chemical space of Group IV, III-V, and II-VI zinc blende (ZB) semiconductors, powered by crystal Graph-based Neural Networks (GNNs) trained on high-throughput density functional theory (DFT) data. Using an innovative approach of sampling partially optimized defect configurations from DFT calculations, we generate one of the largest computational defect datasets to date, containing many types of vacancies, self-interstitials, anti-site substitutions, impurity interstitials and substitutions, as well as some defect complexes.</p>
<p>We applied three types of established GNN techniques, namely Crystal Graph Convolutional Neural Network (CGCNN), Materials Graph Network (MEGNET), and Atomistic Line Graph Neural Network (ALIGNN), to rigorously train models for predicting defect formation energy (DFE) in multiple charge states and chemical potential conditions. We find that ALIGNN yields the best DFE predictions with root mean square errors around 0.3 eV, which represents a prediction accuracy of 98% given the range of values within the dataset, improving significantly on the state-of-the-art.</p>
<p>Models are tested for different defect types as well as for defect charge transition levels. We further show that GNN-based defective structure optimization can take us close to DFT-optimized geometries at a fraction of the cost of full DFT. DFT-GNN models enable prediction and screening across thousands of hypothetical defects based on both unoptimized and partially-optimized defective structures, helping identify electronically active defects in technologically-important semiconductors.</p>

<p>To train the GNN models, the following resources were used:</p>
<ol>
    <li>
        Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties—
        <a href="https://github.com/txie-93/cgcnn">https://github.com/txie-93/cgcnn</a>
    </li>
    <li>
        Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals —
        <a href="https://github.com/materialsvirtuallab/megnet">https://github.com/materialsvirtuallab/megnet</a>
    </li>
    <li>
        Atomistic Line Graph Neural Network for improved materials property predictions— 
        <a href="https://github.com/usnistgov/alignn">https://github.com/usnistgov/alignn</a>
    </li>
</ol>

<p>ALIGNN hyperparameters were used in the paper:</p>
<ul>
    <li>train_ratio: 0.6 (60% of the data is used for training)</li>
    <li>val_ratio: 0.2 (20% of the data is used for validation)</li>
    <li>test_ratio: 0.2 (20% of the data is used for testing)</li>
    <li>epochs: 100-120 (The model is trained for 100-120 epochs)</li>
    <li>batch_size: 8-32 (Size of each batch of data during training)</li>
    <li>weight_decay: 1e-05 (Regularization parameter to prevent overfitting)</li>
    <li>learning_rate: 0.001 (Step size for the optimization algorithm)</li>
    <li>optimizer: 'adamw' (Optimizer used for training, a variant of Adam optimizer)</li>
    <li>cutoff: 8.0 (Cutoff distance for graph construction)</li>
    <li>max_neighbors: 12 (Maximum number of neighbors for each atom)</li>
    <li>alignn_layers: 4 (Number of ALIGNN layers)</li>
    <li>gcn_layers: 4 (Number of GCN layers)</li>
    <li>atom_input_features: 92 (Number of atom input features)</li>
    <li>edge_input_features: 80 (Number of edge input features)</li>
    <li>triplet_input_features: 40 (Number of triplet input features)</li>
    <li>embedding_features: 64 (Number of embedding features)</li>
    <li>hidden_features: 256 (Number of hidden features)</li>
</ul>

<p>All the data (crystals) being used to train the model are added to this repository. Also, the script to perform gradient-free energy minimization is also added. Additionally, the checkpoint of the optimized ALIGNN models trained on Dataset (1+2+3+4) is included, which could be leveraged to predict unoptimized formation energies.</p>

