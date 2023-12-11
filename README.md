<h2>Accelerating Defect Prediction in Semiconductors Using Graph Neural Networks</h2>
<p>First principles computations reliably predict the energetics of point defects in semiconductors, but are constrained by the expense of using large supercells and advanced levels of theory. Machine learning models trained on computational data, especially ones that sufficiently encode defect coordination environments, can be used to accelerate defect predictions.</p>
<p>Here, we develop a framework for the prediction and screening of native defects and functional impurities in a chemical space of Group IV, III-V, and II-VI zinc blende (ZB) semiconductors, powered by crystal Graph-based Neural Networks (GNNs) trained on high-throughput density functional theory (DFT) data. Using an innovative approach of sampling partially optimized defect configurations from DFT calculations, we generate one of the largest computational defect datasets to date, containing many types of vacancies, self-interstitials, anti-site substitutions, impurity interstitials and substitutions, as well as some defect complexes.</p>
<p>We applied three types of established GNN techniques, namely Crystal Graph Convolutional Neural Network (CGCNN), Materials Graph Network (MEGNET), and Atomistic Line Graph Neural Network (ALIGNN), to rigorously train models for predicting defect formation energy (DFE) in multiple charge states and chemical potential conditions. We find that ALIGNN yields the best DFE predictions with root mean square errors around 0.3 eV, which represents a prediction accuracy of 98% given the range of values within the dataset, improving significantly on the state-of-the-art.</p>
<p>Models are tested for different defect types as well as for defect charge transition levels. We further show that GNN-based defective structure optimization can take us close to DFT-optimized geometries at a fraction of the cost of full DFT. DFT-GNN models enable prediction and screening across thousands of hypothetical defects based on both unoptimized and partially-optimized defective structures, helping identify electronically active defects in technologically-important semiconductors.</p>

To train the GNN models we used

(1) Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties—https://github.com/txie-93/cgcnn
(2) Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals —https://github.com/materialsvirtuallab/megnet
(3) Atomistic Line Graph Neural Network for improved materials property predictions— https://github.com/usnistgov/alignn


All the data (crystals) being used to train the model are added to this repository. 
