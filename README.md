# Improving Enzyme Catalytic Residue Prediction by Leveraging Complementary Evolutionary Information from Sequences and Structures
## abstract
Accurate enzyme catalytic residue prediction is essential for understanding biological mechanisms and advancing enzyme engineering. Although existing methods have made significant progress, 
they heavily rely on sequence-based evolutionary information and overlook structural conservation, which is equally important in reflecting catalytic function.
We propose DualEvo, which uses a dual-stream strategy to leverage complementary evolutionary information from both sequence and structure to enhance prediction accuracy.
Specifically, in the feature learning phase, DualEvo implements an adaptive sequence evolutionary feature fusion module to capture mutation patterns via dynamically weighted pooling. 
Furthermore, in the inference phase, we construct a retrieval-augmented module based on structural evolutionary information for post-processing refinement. 
Experimental results on strictly partitioned datasets demonstrate that DualEvo significantly outperforms existing state-of-the-art methods. 
Extensive ablation studies and case analyzes further confirm that the integration of sequence and structural evolutionary information offers complementary benefits, substantially improving catalytic site identification.
