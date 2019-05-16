* Affine set
* Hyperplane
* Differentiate matrix equations
* Nonsingular
* Mixture of Gaussians/Generative Model vs. bivariate Gaussian distributions
* Voronoi tessellation
* The Four Fundamental Subspaces of the Data Matrix


1. One of the most fascinating theorems of linear algebra proves that every square matrix, no matter what numbers it contains, must map a certain set of vectors back to themselves with some scaling. In the general case of a rectangular matrix, it maps a set of input vectors into a corresponding set of output vectors, and its transpose maps those outputs back to the original inputs. The technical terminology is that square matrices have eigenvectors with eigenvalues, and rectangular matrices have left and right singular vectors with singular values.

```
Let  A be a rectangular matrix. If there are vectors u and v and a scalar σ such that Av = σu and ATu = σv, then u and v are called left and right singular vectors and σ is a singular value of A.

```
1. The ordered set of singular values of a matrix is called its spectrum, and it reveals a lot about the matrix. The gap between the singular values affects how stable the solutions are, and the ratio between the maximum and minimum absolute singular values (the condition number) affects how quickly an iterative solver can find the solution. Both of these properties have notable impacts on the quality of the solution one can find.



# Vector Space
## Why we need vector space? [1]
Loosely speaking mathematicans invented vector space to mean any type of mathematical object that can be multiplied by numbers and added together.

## Definition of vector space?

### Field
* A not so rigor definition of Field
```
A field is a set F of numbers with the property that if a, b ∈ F, then a+b, a−b, ab and a/b are also in F (assuming, of course, that b <> 0 in the expression a/b).
```

### Definition of Vector Space[1][2]
```
A vector space consists of a set V (elements of V are called vectors), a field F (elements of F are called scalars), 
and two operations :
• An operation called vector addition that takes two vectors v, w ∈ V, and produces a third vector, written v+w ∈ V.
• An operation called scalar multiplication that takes a scalar c ∈ F and a vector v ∈ V, and produces a new vector, written cv ∈ V.
```

### Examples of Vector Space[2]

## Affine Sets and Hyperplane

### References
1. What is a Vector Space? Geoffrey Scott
1. Mathematical Tools for Physics, Chapter 6, James Nearing. 

