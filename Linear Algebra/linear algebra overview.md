* Differentiate matrix equations
* Nonsingular
* Mixture of Gaussians/Generative Model vs. bivariate Gaussian distributions
* Voronoi tessellation
* The Four Fundamental Subspaces of the Data Matrix
* Relationship between Affine space, Vector Space, Subspace with Solutions of linear sysotems, and matrix's column space


1. One of the most fascinating theorems of linear algebra proves that every square matrix, no matter what numbers it contains, must map a certain set of vectors back to themselves with some scaling. In the general case of a rectangular matrix, it maps a set of input vectors into a corresponding set of output vectors, and its transpose maps those outputs back to the original inputs. The technical terminology is that square matrices have eigenvectors with eigenvalues, and rectangular matrices have left and right singular vectors with singular values.

```
Let  A be a rectangular matrix. If there are vectors u and v and a scalar σ such that Av = σu and ATu = σv, then u and v are called left and right singular vectors and σ is a singular value of A.

```
1. The ordered set of singular values of a matrix is called its spectrum, and it reveals a lot about the matrix. The gap between the singular values affects how stable the solutions are, and the ratio between the maximum and minimum absolute singular values (the condition number) affects how quickly an iterative solver can find the solution. Both of these properties have notable impacts on the quality of the solution one can find.



# Vector Space
## Why we need vector space? [1]
Loosely speaking mathematicans invented vector space to mean any type of mathematical object that can be multiplied by numbers and added together.

## Definition of vector space?
[In mathematics, and more specifically in linear algebra, a **linear subspace**](https://en.wikipedia.org/wiki/Linear_subspace), also known as a vector subspace is a vector space (The term linear subspace is sometimes used for referring to **flats** and **affine subspaces**. In the case of vector spaces over the reals, linear subspaces, flats, and affine subspaces are also called **linear manifolds** for emphasizing that there are also **manifolds**.) that is a subset of some larger vector space. A linear subspace is usually called simply a subspace when the context serves to distinguish it from other kinds of subspace.


### Field
* A not so rigor definition of Field
```
A field is a set F of numbers with the property that if a, b ∈ F, then a+b, a−b, ab and a/b are also in F (assuming, of course, that b <> 0 in the expression a/b).
```

### Definition of Vector Space[1][2, Chapter 6]
```
A vector space consists of a set V (elements of V are called vectors), a field F (elements of F are called scalars), 
and two operations :
• An operation called vector addition that takes two vectors v, w ∈ V, and produces a third vector, written v+w ∈ V.
• An operation called scalar multiplication that takes a scalar c ∈ F and a vector v ∈ V, and produces a new vector, written cv ∈ V.
```

### Examples of Vector Space[2, Chapter 6]

## Affine Space
### [Why do we need affine space? ](http://www.cis.upenn.edu/~cis610/geombchap2.pdf)
1. One reason is that the point in a vector space corresponding to the zero vector(0), called the origin, plays a special role, when there is really no reason to have a privileged origin. 
1. The other reason is that certain notions, such as parallelism, are handled in an awkward manner in vector space. 
1. But the deeper reason is that vector spaces and affine spaces really have **different geometries**. 
   * The geometric properties of a vector space are invariant under **the group of bijective linear maps**. 
   * The geometric properties of an affine space are invariant under **the group of bijective affine maps**.
   * These two groups are **not isomorphic**. **Roughly speaking, there are more affine maps than linear maps.**
1. Affine spaces provide a better framework for doing geometry. In particular, it is possible to deal with points, curves, surfaces, etc., in an **intrinsic manner**, that is, independently of any specific choice of a coordinate system. **Use coordinate systems only when needed!** Affine spaces are the right framework for dealing with motions, trajectories, and physical forces, among other things.

### Linear Maps vs. Affine Maps
### Isomorphic

### Examples of Affine Space
1. The solutions of the system ```Ax=b``` is an affine space, but not a vector space (linear space) in general.

### Affine Space Definitions
1. Almost every affine concept is the counterpart of some concept in linear algebra.


## Affine Sets and Hyperplane
### Hyperplane
1. [A hyperplane in an n-dimensional Euclidean space is a flat, n-1 dimensional subset of that space](https://www.quora.com/Support-Vector-Machines-What-is-an-intuitive-explanation-of-hyperplane) that divides the space into two disconnected parts. 
For example, a point in real line, a line in 2d plane, a plane in 3d space.
1. [A hyperplane is something you can shift parallel to itself, by all possible amounts, and by so doing, fill up all of space.](https://math.stackexchange.com/questions/292066/intuition-about-hyperplane) Of course, all the lines parallel to a given line also fill 3-space; the shifts have to all be in one chosen direction for this to work. Also, one generally excludes curvy sets (like y=cos(x) in the plane) which nonetheless fill up space with their translations.
1. [If V is a vector space, one distinguishes **"vector hyperplanes"**](https://en.wikipedia.org/wiki/Hyperplane) (which are linear subspaces, and therefore must pass through the origin) and **"affine hyperplanes"** (which need not pass through the origin; they can be obtained by translation of a vector hyperplane).

### Affine Space

## [Homogeneous Coordinates](https://hackernoon.com/programmers-guide-to-homogeneous-coordinates-73cbfd2bcc65)


### References
1. What is a Vector Space? Geoffrey Scott
1. Mathematical Tools for Physics, James Nearing. 

