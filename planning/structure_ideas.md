#### Language

There are 2 primary components. Here's a proposal for what we call them (certainly open to improvement if anyone has any better ideas):
1. **core learning**
2. **information geometry metric approximating**


#### Research

An initial goal will be to compare various choices with respect to 2 above (the **information geometry metric approximating**) against each other, and also against using the more standard K-L divergence based loss function in order to see whether there are any gains made by swapping in machine learned approximations of Fisher-Rao and/or Wasserstein metrics as we initially have 1 above (the **core learning**) crunch standard datasets (mnist, and possibly also some text processing too).


#### Architecture

##### core learning

- This is relatively straightforward (just using ngclearn as has been done traditionally by the NACLab team and others), with the single exception of providing a mechanism by which the new metric(s) can be used instead of K-L divergence internally there.

##### infromation geometry metric approximating components

- Fisher-Rao metric 
- Wasserstein metric

For each of those, a number of approximation attempts should be evaluated:

- predictive coding
- perhaps several different more tradtional deep neural net approaches


#### Questions


- Obviously the priority (in terms of computational resources) will be on the **core learning** so how can the information geometry metric(s) be approximated efficiently?

- In terms of scale, how much data needs to be fed into the **core learning** process so that the information geometry metrics will be able to bring value? Are there scale issues that make experimentation on smaller data sets inadequate?

- Will it make sense to try to combine the two information geometry metrics (rather than just trying one or the other) in order to use a hybrid/combinatorial metric?
