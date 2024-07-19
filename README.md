## Rock's relative vector spaces

An attempt at scaling cognitive vector in lower dimensional spaces and exploring kissing numbers.  This little code experiment takes GPT-2's 768 dimensional vectors lowers their dimensionality to 8.

- Given 50,257 semantic vectors (GPT-2), each of dimension 768, find their relative distances
- Given 50,257 concepts with relative distances:
  - put each vector randomly into an n-dimension space (n=8)
  - using a small step "learning rate" move each vector a little bit each time to re-create their required distances
    in order to re-create the distances of the original vector set
- we an already see that 3- and 4- dimensions can't converge even a smaller subset of the vectors (i.e. the kissing numbers / Newton numbers) can't fit the constraints.
- use one central point in the n-dimensional space so the space doesn't "migrate" off the charts when it can't converge


*todo*: lots 
- rewrite the code in Rust or C so that it performs better, maybe even using a GPU.
- if 768 dimensions represent an absolute encoding in terms of semantics, then the lower set of dimensions represents a relative encoding between the vectors.  Information is lost, but - we've got a smaller vector space, if possible, we retain "relative" meaning and continuous meaning through the vector spaces.




![example plot 1](images/example1.jpg)
the experiment using only 3 dimension (n=3), here we get an orbiting effect where the newly mapped points can't meet the constraints and start "orbitting" the fixed centroid.


![example plot 2](images/example2.jpg)
more points in 8 dimensions (n=8), still only using 100s of vectors (slow compute in pure python), but the system does converge for that many points and the delta (error rate) drops until we are satisfied.
