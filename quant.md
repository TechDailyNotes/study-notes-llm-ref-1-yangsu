### Linear Algebra

**unitary matrices (U)**

- definition (either one)
	- the columns of U form an orthonormal basis of C^n under the standard Hermitian inner product on C^n
	- the rows of U form an orthonormal basis of C^n under the standard Hermitian inner product on C^n
	- U* = U^(-1), where U* = conj(U)^T
- property
	- if λ is eigenvalue of U, |λ| = 1 (i.e. it lies on the complex unit circle)

**orthogonal matrices (O)**

- definition (either one)
	- the columns of O form an orthonormal basis of R^n
	- the rows of O form an orthonormal basis of R^n
	- O^T = O^(-1)
- property
	- all eigenvalues of O are real
	- if λ is eigenvalue of O, |λ| = 1, (i.e. λ = 1 or -1)

**hermitian matrices (H)**

- definition (either one)
	- H* = H
	- conj(H) = H^T
- property
	- all eigenvalues of H are real
	- if λ1 and λ2 are distinct eigenvalues of H with corresponding eigenvectors v1 and v2, then v1 and v2 are orthogonal
	- There exists an orthonormal basis of C^n consisting of eigenvectors of H

**skew-hermitian matrices (Hs)**

- definition
	- Hs* = -Hs
- property
	- the eigenvalues of Hs are purely imaginary or 0

**symmetric matrices (S)**

- definition
	- S^T = S (i.e. s_ij = s_ji for all 1 <= i, j <= n)
- property
	- all eigenvalues of S are real
	- If λ1 and λ2 are distinct eigenvalues of A with corresponding eigenvectors v1 and v2, then v1 and v2 are orthogonal
	- There exists an orthonormal basis of R^n consisting of eigenvectors of S

**normal matrices (N)**

- definition
	- N* N = N N*
- property
	- If λ1 and λ2 are distinct eigenvalues of N with corresponding eigenvectors v1 and v2, respectively, then v1 and v2 are orthogonal
	- There exists an orthonormal basis of C^n consisting of eigenvectors of N

**other special matrices**

- correlation matrix C
	- diagonal entries are 1s
	- must be [symmetric positive semidefinite](1)
	- determine (1): *Sylvester's criterion* -> all principle minors of C >= 0 -> the deteriminants of all square matrices obtained by eliminating all same row and columns from C >= 0

**eigenvalues (λ) and eigenvectors (v)**

- eigen means characteristics, proper, (λ, v) is called an eigenpair
- for square matrix A (n x n)
	- we can always find Av = λv, v != 0v
	- C(λ) = det(A - λI) is called the characteristic polynomial of A
	- λ is eigenvalue of A iff C(λ) = 0
	- sum(eigenvalues of A) = trace(A) = sum of diagonal entries of A
	- prod(eigenvalues of A) = det(A)

**singularity**

- a point at which a given mathematical object is not defined
- i.e. lack differentiability or has discontinuity
- singular matrix A property
	- a square matrix that is not invertible
	- det(A) = 0

### Math (Non Linear Algebra)

- Cauchy-Schwartz inequality
	- (sum(i=1->n: ai^2)) * (sum(i=1->n: bi^2)) >= (sum(i=1->n: ai * bi)) ^ 2

### Probability



### Statistics (Non Probability)



### Brain Teaser



### Finance



### Programming


