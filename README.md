**Proposal for Research Initiative: Enhancing Predictive Coding Neural Learning via Information Geometry**

**Principal Investigator:** Charlie Derr  
**Collaborators:** Faezeh Habibi (primary collaborator), other potential team members (Bezawit, Daniel)

**Research Context:**
This initiative aims to extend and enhance the Neural Generative Coding (NGC) framework developed by Alex Ororbia's lab at MIT, to which Faezeh Habibi has contributed significantly. The overarching goal, proposed by Ben Goertzel, is to explore the application of information geometry metrics—specifically Fisher-Rao and Wasserstein—to accelerate and improve neural learning performance within predictive coding models.

**Objectives:**
1. Integrate and leverage the existing NAClab NGClearn package as the foundational framework for predictive coding neural training.
2. Replace the standard loss function currently employed in NGClearn (e.g., Kullback-Leibler divergence or Euclidean metrics) with more advanced, information geometry-based metrics (Fisher-Rao and Wasserstein).
3. Develop computationally efficient approximation methods for these information geometry-based metrics due to their inherent computational complexity.

**Methodological Approach:**
The proposed research will emphasize a highly modular software architecture, enabling:
- Easy interchangeability between different information geometry metrics (Fisher-Rao, Wasserstein).
- The incorporation and testing of multiple approximation strategies, including:
  - Numerical approximation methods.
  - Regression-based approximation techniques, optimizing towards improved metric accuracy.
  - Active Predictive Coding methods specifically aimed at developing robust approximations.

The modular framework will allow for seamless integration of these approximated metrics back into the NGClearn training loops, providing flexibility for future experimentation and potential adaptation beyond predictive coding models, including traditional TNNs, CNNs, and other established machine learning architectures.

**Implementation Details:**
- Development will initially utilize object-oriented Python, consistent with the existing design principles of NGClearn and related libraries (ngc-sim-lib).
- Preliminary tests will employ smaller datasets to validate conceptual and computational viability before scaling to larger datasets, which are anticipated to better exploit the advantages offered by information geometry metrics.
- Evaluation of the necessity and optimal strategy for concurrent training loops (main NGClearn loop and approximation mechanism loop) will be explored, initially focusing on establishing suitable synchronization and cadence between these loops.

**Team Contributions:**
Charlie Derr will lead framework development, system modularity design, and integration strategies. Faezeh Habibi’s prior contributions and expertise with the NGC package will be essential, providing both domain-specific insight and direct assistance in developing computational approximations. Additional support may be sought from Bezawit, Daniel, or other qualified members as necessary, acknowledging that substantial effort will involve collaborative learning and development.

**Expected Outcomes:**
- A robust, scalable, and modular software framework integrating information geometry-based approximations into predictive coding.
- Empirical validation of the improved effectiveness and efficiency of neural learning with information geometry metrics.
- Documentation facilitating future extensions and applications across diverse machine learning paradigms.

