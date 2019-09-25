# Statistics

> id: intro
## Introduction

Mathematical probability is about drawing conclusions about the outcomes of random experiments whose randomness is known and specified precisely. *Statistics* works in the opposite direction: the outcomes are observed, but the probability measure giving rise to those outcomes is unknown. The goal of statistics is to draw conclusions about probability distributions based observations sampled from them. 

For example, consider the eventual adult height $X$ of a particular newborn child. There are no pure mathematical considerations that would suggest a specific distribution for $X$. Our best bet is to collect data on the heights of adults and try to **infer** a probability distribution which is compatible with the observed data. Suppose we measure the height (in inches) of 10 randomly selected folks and get the following numbers:

    pre(julia-executable)
      | heights = [71.54, 66.62, 64.11, 62.72, 68.12, 
      |            69.07, 64.82, 61.92, 68.45, 66.3, 
      |            66.99, 62.2, 61.04, 63.31, 68.94, 
      |            66.27, 66.8, 71.7, 68.93, 66.65, 
      |            71.97, 60.27, 62.81, 70.64, 71.61, 
      |            65.51, 63.1, 66.21, 68.23, 72.32, 
      |            62.29, 63.12, 64.94, 71.89, 65.48, 
      |            63.66, 56.11, 65.63, 61.26, 65.12, 
      |            66.93, 68.51, 67.2, 71.57, 66.65, 
      |            59.77, 61.51, 63.25, 69.12, 64.98]

Each observation provides some evidence about where the probability mass of the height distribution is. We would expect that regions with many observations have more probability mass than regions with few observations, although we should not take this too literally: none of the 50 observations in the list above fall in the interval $[70.7, 71.4]$, but it would not make sense to conclude that adults who are taller than 70.7 inches are necessarily also taller than 71.4 inches. 

::: .exercise
**Exercise**  
Brainstorm at least two ways to come up with a plausible density function given a list of observations like the one given above. 
:::

[Continue](btn:next)

---
> id: step-histogram-density
### Nonparametric estimation

A simple way to obtain a probability distribution from a list of observations is to make a [[**histogram**|**scatter plot**|**regression estimate**]]. The idea is to subdivide the interval from the smallest to the largest observation into smaller intervals and make a bar chart showing the number of observations which fall into each of these intervals. 

---
> id: step-histogram-graph

    pre(julia-executable)
      | using Plots
      | pyplot()
      | histogram(heights,
      |           nbins=12,
      |           label="",
      |           xlabel="height (inches)",
      |           ylabel="count")
      
You might think of a histogram as just a visualization of the data, but it does give an actual distribution: we consider the function whose graph consists of the tops of the histogram bars, and we divide that function by the total number of observations: 

    pre(julia-executable)
      | using Plots
      | histogram(heights,
      |           nbins=12,
      |           label="",
      |           xlabel="height (inches)",
      |           ylabel="count",
      |           normed=true)
      
The arbitrariness in the density function we obtain by normalizing the histogram is hardly disguised: we would have gotten a different result if we'd used a different number of bins, and we could have even decided to use bins of different widths. Nevertheless, the histogram density approximates the actual distribution quite well if we have a lot of data: 

::: .exercise
**Exercise**  
Call the function `{jl} mysample` 10000 times and make a histogram of the resulting observations. Compare the histogram density to the actual density, and observe that the two are very close.

Note: you can evaluate the pdf of `{jl} N₁` at `{jl} x` using `{jl} pdf(N₁,x)`. 
:::

    pre(julia-executable)
      | 
      | function mysample()
      |     if rand() > 0.2
      |         3 + 0.8*randn()
      |     else
      |         -1 + randn()
      |     end
      | end
      | 
      | using Distributions
      | histogram([mysample() for _ in 1:10000],
      |           nbins=80,
      |           normed=true,
      |           label="histogram density")
      | 
      | N₁ = Normal(3,0.8)
      | N₂ = Normal(-1,1)
      | #actualdensity(x) = DENSITYFUNCTIONHERE
      |           
      | plot!(-6:0.1:6, 
      |       actualdensity,
      |       linewidth=3,
      |       label="actual density",
      |       legend=:topright)    
        
    x-quill

---
> id: step-histogram-solution

*Solution*. The density function describing the distribution that `{jl} mysample` draws from is a linear combination of the two given Gaussian density functions, with weights $\frac{4}{5}$ and $\frac{1}{5}$: 

``` julia
actualdensity(x) = 0.8pdf(N₁,x)+0.2pdf(N₂,x)
```

[Continue](btn:next)

---
> id: gaussiandensity
### Parametric estimation

Another way to come up with a density function for some data is to assume that the density function belongs to a specific parametric family of densities, like the set of Gaussian distributions. Then we approximate the parameters using the data. 

::: .exercise
**Exercise**  
Use the sliders to find the μ and σ values for which the normal distribution $\mathcal{N}(\mu, \sigma)$ does the best job of fitting the data. Compare your results to the values obtained using standard methods for this problem by entering your choices for μ and σ in the last line below. 

{.text-center} `μ =`${μ}{μ|60|55,75,0.5}

{.text-center} `σ =`${σ}{σ|1|1,8,0.2}

    x-coordinate-system(x-axis="55|100|5" y-axis="0|0.5|0.1")
    
The best μ value is [[66±2]], and the best σ value is [[3.69±0.3]]. 
:::

[Continue](btn:next)

Later in this course, we will discuss some approaches to choosing parameters optimally, and we'll leave behind the eyeball-it strategy we used in this exercise. 

The histogram estimator is called a [[**nonparametric**|**parametric**]] estimation method, because it doesn't involve assuming that the distribution comes from a particular parametric family. The advantage is that histograms are flexible to represent a variety of density shapes, while parametric methods have the advantage of making more efficient use of data in the situations where the parametric assumption happens to be valid. 

---
> id: regression
### Regression

Statistics is not limited to estimating the distribution of a single real-valued random variable like human height. Typically we want to have information about the *joint* distribution of such a variable with other variables whose values we are in a position to know. Such joint information allows us to make more accurate predictions, and that increased accuracy is usually critical for the business or research purposes that motivated the inquiry. 

For example, if we're able to collect the heights of many adults together along the heights of each of their parents, then we can aim to understand the *conditional* expectation of a person's height, conditioned on the heights of their parents. Since we can measure the heights of a child's parents, we can use this information to make a better prediction for how tall the child will grow up to be. The problem of estimating the conditional expectation of one random variable given others is called **regression**. 

[Continue](btn:next)

---
> id: step-preview-joint-density-estimation

In the next section, we will develop some intuitive techniques for estimating density functions for joint distributions. 

::: .exercise
**Exercise**  
Consider a random variable $X$ that you know takes values in $\\{0,1,2\\}$. Suppose that 100 independent observations are made from the distribution of $X$, and suppose they are the values given below. Propose an estimate of the distribution of $X$. 
:::

    pre(julia-executable)
      | observations = [
      | 0, 2, 2, 2, 2, 2, 0, 2, 2, 1, 
      | 0, 2, 2, 1, 0, 1, 0, 2, 1, 2, 
      | 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 
      | 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 
      | 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 
      | 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 
      | 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 
      | 2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 
      | 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
      | 2, 0, 2, 2, 2, 2, 2, 1, 0, 2
      | ]
      
    x-quill

---
> id: step-discrete-solution

**Solution**. Since 70% of the observations are 2's, we posit that the probability of the event $\\{X = 2\\}$ is 70%. Likewise, the probabilities of the events $\\{X = 1\\}$ and $\\{X = 0\\}$ we estimate to be 13% and 17%, respectively. 

---
> id: joint-density-estimation
## Estimating Joint Densities

The following setup will be our running example throughout this section. We will begin by assuming that we know the joint distribution of the two random variables $X$ and $Y$, and then we will explore what happens when we drop that assumption. 

::: .example
**Example**  
 Suppose that $X$ is a random variable which represents the number of hours a student studies for an exam, and $Y$ is a random variable which represents their performance on the exam. Suppose that the joint density of $X$ and $Y$ is given by

``` latex

    f(x,y) = \frac{3}{4000(3/2)\sqrt{2\pi}}x(20-x)e^{-\frac{1}{2(3/2)^2}\left(y - 2 - \frac{1}{50}x(30-x)\right)^2}.

```
 Given that a student's number of hours of preparation is known to be $x$, what is the best prediction for $Y$ (as measured by mean squared error)?
:::

    figure
      img(src="images/exam-density.png" width=350)

      p.caption.md Figure 1.1: The probability density function of the joint distribution of $(X,Y)$, where $X$ is the number of hours of studying and $Y$ is the exam score.

[Continue](btn:next)

---
> id: example-solution-1

<!--TODO: fix caption and images (exam-density and exam-density-line)-->

*Solution*.  The best guess of a random variable (as measured by mean squared error) is the expectation of the random variable. Likewise, the best guess of $Y$ given $\\{X=x\\}$ is the *conditional* expectation of $Y$ given $\\{X=x\\}$. Inspecting the density function, we recognize that restricting it to a vertical line at position $x$ gives an expression which is proportional to the Gaussian density centered at $2 + \frac{1}{50}x(30-x)$. Therefore,

``` latex
\mathbb{E}[Y | X=x] = 2 + \frac{1}{50}x(30-x).
```

[Continue](btn:next)

---
> id: step-regression-definition

A graph of this function is shown in red in the figure below:

    figure
      img(src="images/exam-density-line.png" width=240)

      p.caption.md Figure 1.2: The conditional expectation of $Y$ given $\\{X = x\\}$, as a function of $x$.

We call $r(x) = \mathbb{E}[Y | X = x]$ the **regression** function for the problem of predicting $Y$ given the value of $X$. 

Given the distribution of $(X,Y)$, finding the regression function is a probability problem. However, in practice the probability measure is typically only known *empirically*, that is, by way of a collection of independent observations sampled from the measure. For example, to understand the relationship between exam scores and studying hours, we would record the values $(X\_i,Y\_i)$ for many past exams $i \in \\{1,2,\ldots,n\\}$ (see Figure 1.3 below).

    figure
      img(src="images/exam-samples.png" width = 350)
      p.caption.md Figure 1.3: One thousand independent samples from the joint distribution of $(X,Y)$.


::: .exercise
**Exercise**  
Describe an example of a random experiment where the probability measure may be reasonably inferred without the need for experimentation or data collection. What is the difference between such an experiment and one for which an empirical approach is required?
:::

    x-quill

---
> id: solution-1

*Solution*. There are many random experiments for which it is reasonable to use a measure derived from principle rather than from experimentation. For example, if we are drawing a card from a well-shuffled deck, or drawing a marble from a well-stirred urn, then we would assume that the probability measure is *uniform*. This assumption would lead to more accurate predictions than if we used an inferred measure based on repeated observations of the experiment. By extension, if we have many random variables whose distributions are known from principle and which are known from principle to be independent, then we know that their average is approximately normally distributed.

The key difference between these examples and those requiring an empirical approach is the use of symmetry. There is no *a  priori* reason to believe that, for example, the possible scores on an exam are equally likely, or that an exam score is a sum of independent random variables.

[Continue](btn:next)

---
> id: step-regression-recoverable

You can convince yourself that $r$, and indeed the whole joint distribution of $X$ and $Y$, is approximately recoverable from the samples shown in Figure 1.3: if we draw a curve roughly through the middle of the point cloud, it would be pretty close to the graph of $r$. If we shade the region occupied by the point cloud, darker where there are more points and lighter where there are fewer, it will be pretty close to the heat map of the density function. 

We will want to sharpen this intuition into a computable algorithm, but the picture gives us reason to be optimistic.

[Continue](btn:next)

---
> id: step-delta-mass-approximation

::: .exercise
**Exercise**  
In the context of Figure 1.3, why doesn't it work to approximate the conditional distribution of $Y$ given $\\{X = x\\}$ using all of the samples which are along the vertical line at position $x$?
:::

    x-quill

---
> id: solution-2

*Solution*. Because for almost all values of $x$, there will be *no* points along the vertical line at position $x$!

[Continue](btn:next)

---
> id: kernel-density-estimation
#### Kernel Density Estimation

The **empirical** distribution of $(X,Y)$ is the probability measure which assigns mass $\frac{1}{n}$ to each of the $n$ observed samples from $(X,Y)$. The previous exercise illustrated shortcomings of using the empirical distribution directly for this problem: the mass is not appropriately spread out in the rectangle. It makes more sense to conclude from the presence of a sample at a point $(x,y)$ that the unknown distribution of $(X,Y)$ has some mass *near* $(x,y)$, not necessarily exactly *at* $(x,y)$.

We can capture this idea mathematically by spreading out $\frac{1}{n}$ units of probability mass around each sample $(x\_i,y\_i)$. We have to choose a function—called the **kernel**—to specify how the mass is spread out. 

[Continue](btn:next)

---
> id: tricubegraph

There are many kernel functions in common use; we will base our kernel on the *tricube* function, which is defined by

``` latex

    D(u) = \left\{
      \begin{array}{cl}
        \frac{70}{81}(1-|u|^3)^3 & \text{if }|u| < 1\\
        0 & \text{if }|u| \geq 1.
      \end{array}\right.      

```
We define $D\_\lambda(u) = \frac{1}{\lambda} D\left(\frac{u}{\lambda}\right)$; The parameter $\lambda > 0$ can be tuned to adjust how tightly the measure is concentrated at the center. Experiment with the slider below to get a feel for how changing $\lambda$ affects the shape of the graph of $D_\lambda$. 

{.text-center} `λ =`${λ}{λ|1|0.3,4,0.05}

    x-coordinate-system(x-axis="-3|3|0.5" y-axis="0|3|0.5")

[Continue](btn:next)

---
> id: step-tricube-2d

We define the kernel function $K\_\lambda$ as a product of two copies of $D\_\lambda$, one for each coordinate (see Figure 1.4 for a graph):

``` latex
K_\lambda(x,y) = D_\lambda(x)D_\lambda(y).
```

    figure
      img(src="images/tricube-100.svg" width=240)

      p.caption.md Figure 1.4: The graph of the kernel $K\_\lambda(x,y)$ with $\lambda = 1$.

::: .exercise
**Exercise**  
Is the probability mass more or less tightly concentrated around the origin when $\lambda$ is small?
:::

    x-quill

---
> id: solution-3

*Solution*. Note that $K\_\lambda(x,y)$ as soon as either $x$ or $y$ is larger than $\lambda$. Therefore, all of the probability mass is less than $\lambda \sqrt{2}$ units from the origin. So small $\lambda$ corresponds to tight concentration of the mass near zero.

[Continue](btn:next)

---
> id: step-piles-of-mass

Now, let's place small piles of probability mass with the shape shown in Figure 1.4 at each sample point. The resulting approximation of the joint density $f(x,y)$ is

``` latex
\widehat{f}_\lambda(x,y) = \frac{1}{n}\sum_{i=1}^n K_\lambda(x-X_i,y-Y_i).
```

Graphs of $\widehat{f}\_\lambda$ for various values of $\lambda$ are shown below. We can see that the best match is going to be somewhere between the $\lambda \to 0$ and $\lambda\to\infty$ extremes (which are too concentrated and too diffuse, respectively).

    figure
      img(src="images/kde-figures.png" width="80%")
      p.caption.md The density estimator $\widehat{f}_\lambda$ for $\lambda = 0.25, 1, 3,$ and $10$. The last figure shows the actual density function.

::: .exercise
**Exercise**  
Estimate the best value of $\lambda$ by eyeballing the figure above (the values of $\lambda$ in the first four figures are 0.25, 1, 3, and 10. The last figure shows the actual density).
:::

    x-quill

---
> id: step-solution-eyeball-lambda

*Solution*. In the $\lambda=1$ picture, it looks like the estimated measure is taking individual points too seriously, since there are lots of points with more mass around them than in the gaps between observations. In the $\lambda = 10$ picture, the mass looks way too spread out. Among the ones shown, $\lambda = 3$ appears to be the best. 

[Continue](btn:next)

---
> id: step-best-lambda

We can take advantage of our knowledge of the density function to determine which $\lambda$ value works best. In practice, we would not have access to this information, but let's postpone dealing with that issue till the next section.

We measure the difference between $f$ and $\widehat{f}\_\lambda$ by evaluating both functions at each point in a square grid and finding the sum of the squares of these differences. This sum is approximately proportional to 
[[$\int (f - \hat{f})^2$|$\nabla f$|$\int f$]].

---
> id: step-julia-best-lambda

    pre(julia-executable)
      | using LinearAlgebra, Statistics, Roots, Optim, Plots, Random
      | Random.seed!(1234)
      | n = 1000 # number of samples to draw
      |
      | # the true regression function
      | r(x) = 2 + 1/50*x*(30-x)
      | # the true density function
      | σy = 3/2  
      | f(x,y) = 3/4000 * 1/√(2π*σy^2) * x*(20-x)*exp(-1/(2σy^2)*(y-r(x))^2)
      |
      | # x values and y values for a grid
      | xs = 0:1/2^3:20
      | ys = 0:1/2^3:10
      |
      | # F(t) is the CDF of X
      | F(t) = -t^3/4000 + 3t^2/400
      |
      | "Sample from the distribution of X (inverse CDF trick)"
      | function sampleX()
      |     U = rand()
      |     find_zero(t->F(t)-U,(0,20),Bisection())
      | end
      |
      | "Sample from joint distribution of X and Y"
      | function sampleXY(r,σ)
      |     X = sampleX()
      |     Y = r(X) + σ*randn()
      |     (X,Y)
      | end
      |
      | # Sample n observations
      | samples = [sampleXY(r,σy) for i=1:n]
      |
      | D(u) = abs(u) < 1 ? 70/81*(1-abs(u)^3)^3 : 0 # tri-cube function
      | D(λ,u) = 1/λ*D(u/λ) # scaled tri-cube
      | K(λ,x,y) = D(λ,x) * D(λ,y) # kernel
      | kde(λ,x,y,samples) = sum(K(λ,x-Xi,y-Yi) for (Xi,Yi) in samples)/length(samples)
      | # `optimize` takes functions which accept vector arguments, so we treat
      | # λ as a one-entry vector
      | L(λ) = sum((f(x,y) - kde(λ,x,y,samples))^2 for x=xs,y=ys)*step(xs)*step(ys)
      |
      | # minimize L using the BFGS method
      | λ_best = optimize(λ->L(first(λ)),[1.0],BFGS())
      |


We find that the best $\lambda$ value comes out to about $\lambda = 1.92$.

[Continue](btn:next)

---
> id: cross-validation
#### Cross-validation

Since we only know the density function in this case because we're in the controlled environment of an exercise, we need to think about how we could have chosen the optimal $\lambda$ value without that information. 

One idea, which we will find is applicable in many statistical contexts, is to reserve one sample point and form an estimator which only uses the other $n-1$ samples. We evaluate this density approximation at the reserved sample point, and we repeat all of these steps for each sample in the data set. If the resulting density value is consistently very small, then our density estimator is being consistently "surprised" by the location of the reserved sample. This suggests that our $\lambda$ value is too large or too small. This idea is called **cross-validation**.

---
> id: kdecrossvalidate

::: .exercise
**Exercise**  
Experiment with the sliders below to adjust the bandwidth $\lambda$ and the omitted point `i` to find a value of $\lambda$ you find satisfactory (set `i` to `7` to include all six points).

{.text-center} `λ =`${λ}{λ|1|0.7,20,0.1} 

{.text-center} `i = `${i}{i|1|1,7,1}

    x-coordinate-system(x-axis="-5|10|2" y-axis="0|0.5|0.1")

The density value at the omitted point is small when $\lambda$ is too small because [[the density is too concentrated at other points|the density is too spread out]], and the value is small when $\lambda$ is too large because [[the density is too spread out|the density is zero everywhere]]. 
:::

[Continue](btn:next)

---
> id: step-kde-cv

Let's flesh this out mathematically. We are aiming to minimize the **loss** (or **error**, or **risk**) of our estimator, which we define to be the integrated squared difference between $\widehat{f}\_\lambda$ and the true density $f$. We can write this integral as

``` latex
\int (\widehat{f}_\lambda - f)^2 = \int \widehat{f}_\lambda^{2} - 2
\int \widehat{f}_\lambda f + \int f^2, 
```

using [[linearity of integration|the product rule|the chain rule]]. 
The third term does not involve $\widehat{f}$, so minimizing $\int (\widehat{f}\_\lambda - f)^2$ is the same as minimizing $\int \widehat{f}\_\lambda^{2} - 2 \int \widehat{f}\_\lambda f$, which we will call $J(\lambda)$. 

---
> id: step-expectation-formula-trick

Recalling the expectation formula

``` latex
\mathbb{E}[g(X,Y)] = \int g(x,y) f_{X,Y}(x,y) \, \mathrm{d} x \, \mathrm{d} y,
```
we recognize the second term in the top equation as $-2\mathbb{E}[\widehat{f}\_\lambda(X,Y)]$, where $(X,Y)$ is a sample from the true density $f$ which is *independent* of $\widehat{f}\_\lambda$. To get samples which are independent of the estimator, we will define $\widehat{f}\_{\lambda}^{(-i)}$ to be the density estimator obtained using the samples other than the $i$th one. Since the samples $(X\_i,Y\_i)$ are drawn from the joint density of $(X,Y)$ and are assumed to be independent, we can suggest the approximation

``` latex
\mathbb{E}[\widehat{f}_\lambda(X,Y)]  \approx  
  \frac{1}{n} \sum_{i=1}^n \widehat{f}_{\lambda}^{\:(-i)}(X_i,Y_i),
```

[Continue](btn:next)

---
> id: step-cross-validation-loss-estimator

We call

``` latex
\widehat{J}(\lambda) = \int \widehat{f}_\lambda^{2} -
  \frac{2}{n} \sum_{i=1}^n \widehat{f}_{\lambda}^{\:(-i)}(X_i,Y_i)
```

the **cross-validation loss estimator**. Let's find the value of $\lambda$ which minimizes $\widehat{J}(\lambda)$ for the present example:

    pre(julia-executable)
      | "Evaluate the summation Σᵢ f⁽⁻ⁱ⁾(Xᵢ,Yᵢ) in J(λ)'s second term"
      | function kdeCV(λ,i,samples)
      |     x,y = samples[i]
      |     newsamples = copy(samples)
      |     deleteat!(newsamples,i)
      |     kde(λ,x,y,newsamples)
      | end
      |
      | # first line approximates ∫f̂², the second line approximates -(2/n)∫f̂f
      | J(λ) = sum([kde(λ,x,y,samples)^2 for x=xs,y=ys])*step(xs)*step(ys) -
      |         2/length(samples)*sum(kdeCV(λ,i,samples) for i=1:length(samples))
      | λ_best_cv = optimize(λ->J(first(λ)),[1.0],BFGS())
      |


The cross-validated minimizing value is $\lambda = 1.88$. Recall that the true minimizer is $\lambda =1.92$. So in this case, cross validation gets us quite close to the optimal $\lambda$ value without needing to know the actual density. In fact, the cross-validation estimator performs well in general given enough data, in the following sense:

::: .theorem
**Theorem** (Stone's Theorem)  
Suppose that $f$ is a bounded probability density function. Let $\widehat{f}^{\mathrm{CV}}\_n$ be the kernel density estimator with bandwidth $\lambda$ obtained by cross-validation, and let $\widehat{f}\_n$ be the kernel density estimator with optimal bandwidth $\lambda^{\mathrm{min}}\_n$. Then

``` latex

    \frac{\int (f - \widehat{f}^{\mathrm{CV}}_n)^2}{ \int
      (f-\widehat{f}_n)^2}

```
 converges in probability to 1 as $n\to\infty$. Furthermore, there are constants $C\_1$ and $C\_2$ such that $\int (f-\widehat{f}\_n)^2\approx C_1n^{-4/5}$ and $\lambda^{\mathrm{min}}\_n \approx C_2n^{-1/5}$ for large $n$.
:::

---
> id: nonparametric regression
#### Nonparametric Regression

With an estimate of the joint density of $X$ and $Y$ in hand, we can turn to the problem of estimating the regression function $r(x) = \mathbb{E}[Y | X = x]$. If we restrict the density estimate $\widehat{f}\_\lambda$ to the vertical line at position $x$, we find that the conditional density is

``` latex

    \widehat{f}_{Y | X = x}(y) = \frac{\displaystyle{\frac{1}{n}\sum_{i=1}^n
        K_\lambda(x-X_i,y-Y_i)}}{\displaystyle{
        \int \frac{1}{n}\sum_{i=1}^n K_\lambda(x-X_i,y-Y_i)\, \mathrm{d} y}}.

```

    figure
      img(src="images/kde-slice.png" width=350)

      p.caption.md Figure 1.6: A kernel density estimator with 16 samples and $\lambda = 1$.

So the conditional expectation of $Y$ given $\\{X = x\\}$ is

``` latex


  \widehat{r}(x) = \int y \widehat{f}_{Y | X = x}(y) \, \mathrm{d} y =
  \frac{\displaystyle{\int y \frac{1}{n}\sum_{i=1}^n
    K_\lambda(x-X_i,y-Y_i)\, \mathrm{d} y}}{\displaystyle{
    \int \frac{1}{n}\sum_{i=1}^n K_\lambda(x-X_i,y-Y_i)\, \mathrm{d} y}}.

```

Let's try to simplify this expression. Looking at Figure 1.6, we can see by the symmetry of the function $D$ that each sample $(X\_i,Y\_i)$ contributes $Y\_iD\_\lambda(x-X\_i)$ to the numerator of the above equation. To arrive at the same conclusion by manipulating the formula directly, we write

``` latex

  \int \frac{1}{n}\sum_{i=1}^n K_\lambda(x-X_i,y-Y_i) \, \mathrm{d} y =
  \frac{1}{n} \sum_{i=1}^n D_\lambda(x-X_i) \int yD_\lambda(y-Y_i) \,
  \mathrm{d} y.

```
Then $\int yD(y-Y\_i) \mathrm{d} y$ is the average of the probability measure with density $D$ centered at $Y\_i$. Since $D$ is symmetric, this average is the center point $Y\_i$.

[Continue](btn:next)

---
> id: step-kde-same-idea-denominator

Applying a similar idea to the denominator (note that instead of $Y\_i$, we get the total mass 1 from integrating $D(y-Y\_i)$), we find that

``` latex

  \widehat{r}(x) =
  \frac{\displaystyle{\sum_{i=1}^nD_\lambda(x-X_i)Y_i}}{\displaystyle{\sum_{i=1}^nD_\lambda(x-X_i)}}.

```

    figure
      img(src="images/reg-approx.png" width=240)

      p.caption.md Figure 1.7: The Nadaraya-Watson estimator $\widehat{r}$.

The graph of this function is shown in Figure 1.7. We see that it matches the graph of $r$ quite closely except near the ends of the interval.

    pre(julia-executable)
      | λ = first(λ_best_cv.minimizer)
      | r̂(x) = sum(D(λ,x-Xi)*Yi for (Xi,Yi) in samples)/sum(D(λ,x-Xi) for (Xi,Yi) in samples)
      | pyplot()
      | plot(0:0.2:20, r̂)

The approximate integrated squared error of this estimator is `{jl} sum((r(x)-r̂(x))^2 for x in xs)*step(xs) = 1.90`.

[Continue](btn:next)

---
> id: step-kde-commentary

Kernel density estimation is just one approach to estimating a joint density, and the Nadaraya-Watson estimator is just one approach to estimating a regression function. In the machine learning course following this one, we will explore a wide variety of machine learning models which take quite different approaches, and each will have its own strengths and weaknesses. 

---
> id: point-estimation
## Point Estimation

In the previous section, we discussed the problem of estimating a distribution given a list of independent samples from it. Now we turn to the simpler task of **point estimation**: estimating a single real-valued feature (such as the mean, variance, or maximum) of a distribution. We begin by formalizing the notion of a real-valued feature of a distribution. 

::: .definition
**Definition** (Statistical functional)  
A **statistical functional** is any function $T$ from the set of distributions to $[-\infty,\infty]$. 
:::

For example, if we define $T\_1(\nu)$ to be the mean of the distribution $\nu$, then $T\_1$ is a statistical functional. Similarly, consider the *maximum* functional $T\_2(\nu) = F^{-1}(1)$ where $F$ is the CDF of $\nu$. To give a more complicated example, we can define $T\_3(\nu)$ to be the expected value of the difference between the greatest and least of 10 independent random variables with common distribution $\nu$. Then $T\_3$ also a statistical functional. 

Given a statistical functional, our goal will be to use a list of independent samples from $\nu$ to estimate $T(\nu)$. 

::: .definition
**Definition** (Estimator)  
An **estimator** $\widehat{\theta}$ is a random variable which is a function of $n$ i.i.d.\ random variables.
:::

::: .example
**Example**  
Draw 500 independent samples from an exponential distribution with parameter 1. Plot the function $\widehat{F}$ which maps $x$ to the proportion of samples at or to the left of $x$ on the number line. We call $\widehat{F}$ the **empirical CDF**. Compare the graph of the empirical CDF to the graph of the CDF of the exponential distribution with parameter 1. 
:::


*Solution*.  We can graph $\widehat{F}$ using a step plot: 

    pre(julia-executable)
      | using Plots, Distributions
      | pyplot()
      | n = 500
      | xs = range(0, 8, length=100)
      | plot(xs, x-> 1-exp(-x), label = "true CDF", legend = :bottomright)
      | step!(sort(rand(Exponential(1),n)), (1:n)/n, 
      |       seriestype = :steppre, label = "empirical CDF")
      
    pre.rblock(r-executable)
      | n <- 500
      | xvals = seq(0,8,length=100)
      | 
      | ggplot() +
      |   geom_line(aes(x=xvals,y=1-exp(-xvals))) + 
      |   geom_step(aes(x=sort(rexp(n)),y=(1:n)/n))

[Continue](btn:next)

---
> id: step-empirical-measure

This example suggests an idea for estimating $\widehat{\theta}$: since the unknown distribution $\nu$ is typically close to the measure $\widehat{\nu}$ which places mass $\frac{1}{n}$ at each of the observed samples, we can build an estimator of $T(\nu)$ by plugging $\widehat{\nu}$ into $T$. 

::: .definition
**Definition** (Plug-in estimator)  
The plug-in estimator of $\theta = T(\nu)$ is $\widehat{\theta} = T(\widehat{\nu})$. 
::: 

::: .example
**Example**  
Find the plug-in estimator of the mean of a distribution. Find the plug-in estimator of the variance. 
:::

*Solution*. The plug-in estimator of the mean is the mean of the empirical distribution, which is the average of the locations of the samples. We call this the **sample mean**: 

``` latex
\overline{X} = \frac{X_1 + \cdots + X_n}{n}. 
```

Likewise, the plug-in estimator of the variance is **sample variance** 

``` latex
S^2 = \frac{1}{n}\left( (X_1 - \overline{X})^2 + (X_2 -
        \overline{X})^2 + \cdots +  (X_n - \overline{X})^2\right). 
```

Ideally, an estimator $\widehat{\theta}$ is close to $\theta$ with high probability. We will see that we can decompose the question of whether $\widehat{\theta}$ is close to $\theta$ into two sub-questions: is the *mean* of $\widehat{\theta}$ close to $\theta$, and is $\widehat{\theta}$ close to its mean with high probability? 

---
> id: bias-section
#### Bias

::: .definition
**Definition** (Bias)  
The **bias** of an estimator $\widehat{\theta}$ is 

``` latex
\mathbb{E}[\widehat{\theta}] - \theta.
``` 

An estimator is said to be **biased** if its bias is nonzero and **unbiased** if its bias is zero. 
:::

::: .example
**Example**  
Consider the estimator 

``` latex
\widehat{\theta} = \max(X_1, \ldots, X_n) 
```

of the maximum functional. Assuming that the distribution has a density function, show that $\widehat{\theta}$ is biased. 
:::

[Continue](btn:next)

---
> id: step-biased-estimator-solution

*Solution*. If $\nu$ is a continuous distribution, then the probability of the event $\\{X\_i < T(\nu)\\}$ is $1$ for all $i=1,2,\ldots,n$. This implies that $\widehat{\theta} < T(\nu)$ with probability 1. Taking expectation of both sides, we find that $\mathbb{E}[\widehat{\theta}] < T(\nu)$. Therefore, this estimator has negative bias. 

[Continue](btn:next)

---
> id: step-standard-error
#### Standard Error

Zero or small bias is a desirable property of an estimator: it means that the estimator is accurate *on average*. The second desirable property of an estimator is for the probability mass of its distribution to be concentrated near its mean: 

::: .definition
**Definition** (Standard error)  
The standard error $\operatorname{se}(\widehat{\theta})$ of an estimator $\widehat{\theta}$ is its standard deviation. 
:::

::: .example
**Example**  
Find the standard error of the sample mean if the distribution $\nu$ with variance $\sigma^2$. 
:::

 

*Solution*. We have 

``` latex
\operatorname{Var}\left(\frac{X_1 + X_2 + \cdots + X_n}{n}\right) =
           \frac{1}{n^2}(n\operatorname{Var} X_1) = \frac{\sigma^2}{n}. 
```
Therefore, the standard error is $\sigma/\sqrt{n}$. 

If the expectation of an estimator of $\theta$ is close to $\theta$ and if the estimator close to its average with high probability, then it makes sense that $\widehat{\theta}$ and $\theta$ are close to each other with high probability. We can measure the discrepancy between $\widehat{\theta}$ and $\theta$ directly by computing their average squared difference: 

::: .definition
**Definition** (Mean squared error)  
The mean squared error of an estimator $\widehat{\theta}$ is $\mathbb{E}[(\widehat{\theta} - \theta)^2]$. 
:::

As advertised, the mean squared error decomposes as a sum of *squared bias* and *squared standard error*: 

::: .theorem
**Theorem**  
The mean squared error of an estimator $\theta$ is equal to its variance plus its squared bias: 

``` latex
\mathbb{E}[(\widehat{\theta} - \mathbb{E}[\widehat{\theta}])^2] +
  (\mathbb{E}[\widehat{\theta}] - \theta)^2. 
```
:::


*Proof*. The idea is to add and subtract the mean of $\widehat{\theta}$. We find that 

``` latex
\mathbb{E}[(\widehat{\theta} - \theta)^2] &=
\mathbb{E}[(\widehat{\theta} - \mathbb{E}[\widehat{\theta}] + \mathbb{E}[\widehat{\theta}] -  \theta)^2] \\
&= \mathbb{E}[(\widehat{\theta} - \mathbb{E}[\widehat{\theta}])^2] + 
  2\mathbb{E}[(\widehat{\theta} - \mathbb{E}[\widehat{\theta}])(\mathbb{E}[\widehat{\theta}] -  \theta)] +
  (\mathbb{E}[\widehat{\theta}] -  \theta)^2. 
```

The middle term is zero by [[linearity of expectation|the symmetry of $\theta$]]. 

---
> id: step-bias-variance-both-converge

If the bias and standard error of an estimator both converge to 0, then the estimator is *consistent*: 

::: .definition
**Definition** (Consistent)  
An estimator is **consistent** if $\widehat{\theta}$ converges to $\theta$ in probability as $n\to\infty$. 
:::

::: .example
**Example**  
Show that the plug-in maximum estimator $\widehat{\theta}\_n = \max(X\_1, \ldots, X\_n)$ of $\theta = T(\nu) = F^{-1}(1)$ is consistent, assuming that the distribution belongs to the parametric family $\\{\operatorname{Unif}([0,b]) \,: \, b \in \mathbb{R}\\}$. 
:::

[Continue](btn:next)

---
> id: step-consistent-solution

*Solution*. The probability that $\widehat{\theta}\_n$ is more than $\epsilon$ units from $\theta$ is equal to the probability that every sample is less than $\theta - \epsilon$, which by independence is equal to 

``` latex
\left(\frac{\theta - \epsilon}{\theta}\right)^n. 
```

This converges to 0 as $n \to \infty$, since $\frac{\theta - \epsilon}{\theta} < 1$. 

    figure
      img(src="images/biasvariance.svg" width=400)
      p.caption.md An estimator of $\theta$ has high or low bias depending on whether its mean is far from or close to $\theta$. It has high or low variance depending on whether its mass is spread out or concentrated. 

::: .example
**Example**  
Show that the sample variance $S^2 = \frac{1}{n}\sum\_{i=1}^n (X\_i -
  \overline{X})^2$ is biased. 
::: 

*Solution*. We will perform the calculation for $n = 3$. It may be generalized to other values of $n$ by replacing 3 with $n$ and $2$ with $n-1$. We have 

``` latex
\mathbb{E}[S^2] = \frac{1}{3}\mathbb{E}\left[ \left(\frac{2}{3}X_1 -
    \frac{1}{3}X_2 - \frac{1}{3}X_3\right)^2
  + \left(\frac{2}{3}X_2 - \frac{1}{3}X_3 -
    \frac{1}{3}X_1\right)^2
  + \left(\frac{2}{3}X_3 - \frac{1}{3}X_1 -
    \frac{1}{3}X_2\right)\right]^2
```

Squaring out each trinomial, we get $\frac{4}{9}X\_1^2$ from the first term and $\frac{1}{9}X\_1^2$ from each of the other two. So altogether the $X\_1^2$ term is $\frac{6}{9}X\_1^2$. By symmetry, the same is true of $X\_2^2$ and $X\_3^2$. For cross-terms, we get $-\frac{4}{9}X\_1X\_2$ from the first squared expression, $-\frac{4}{9}X\_1X\_2$ from the second, and $\frac{2}{9}X\_1X\_2$ from the third. Altogether, we get $-\frac{6}{9}X\_1X\_2$. By symmetry, the remaining two terms are $-\frac{6}{9}X\_1X\_3 -\frac{6}{9}X\_2X\_3$. 

Recalling that $\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ for any random variable $X$, we have $\mathbb{E}[X\_1^2] = \mu^2 + \sigma^2$, where $\mu$ and $\sigma$ are the mean and standard deviation of the distribution of $X\_1$(and similarly for $X\_2$ and $X\_3$. So we have 

``` latex
\mathbb{E}[S^2] &= \frac{1}{3}\left(\frac{6}{9}(X_1^2 + X_2^2 +
          X_3^2) - \frac{6}{9}(X_1X_2 +X_1X_3 + X_2X_3)\right) \\ 
&= \frac{1}{3}\cdot\frac{6}{9}(3(\sigma^2 + \mu^2) - 3\mu^2) =
  \frac{2}{3}\sigma^2.                                           
```

If we repeat the above calculation with $n$ in place of 3, we find that the resulting expectation is $\frac{n-1}{n}\sigma^2$. 

Motivated by this example, we define the **unbiased sample variance** 

``` latex 
\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2
```

::: .exercise
**Exercise**  
Let's revisit the adult height distribution from the first section. We observed the human adult heights shown below (in inches). If we want to approximate the height distribution with a Gaussian, it seems reasonable to estimate μ and σ² using the unbiased estimators $\mu = \frac{1}{n}(X_1 + \cdots + X_n)$ and $\widehat{\sigma}^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i-\mu)^2$. 

Calculate these estimators for the height data.
:::

    pre(julia-executable)
      | heights = [71.54, 66.62, 64.11, 62.72, 68.12, 
      |            69.07, 64.82, 61.92, 68.45, 66.3, 
      |            66.99, 62.2, 61.04, 63.31, 68.94, 
      |            66.27, 66.8, 71.7, 68.93, 66.65, 
      |            71.97, 60.27, 62.81, 70.64, 71.61, 
      |            65.51, 63.1, 66.21, 68.23, 72.32, 
      |            62.29, 63.12, 64.94, 71.89, 65.48, 
      |            63.66, 56.11, 65.63, 61.26, 65.12, 
      |            66.93, 68.51, 67.2, 71.57, 66.65, 
      |            59.77, 61.51, 63.25, 69.12, 64.98]

    x-quill
    
---
> id: height-parameters-solution

*Solution*. Julia has built-in functions for this: 

    pre(julia-executable)
      | mean(heights), var(heights)
      
We could also write our own:

    pre(julia-executable)
      | μ = sum(heights)/length(heights)
      | σ̂² = 1/(length(heights)-1) * sum((h-μ)^2 for h in heights)

---
> id: confidence-intervals
## Confidence Intervals

It is often of limited use to know the value of an estimator given an observed collection of samples, since the single value does not indicate how close we should expect $\theta$ to be to $\widehat{\theta}$. For example, if a poll estimates that a randomly selected voter has 46% probability of being a supporter of candidate A and a 42% probability of being a supporter of candidate B, then knowing more information about the distributions of the estimators is essential if we want to know the probability of winning for each candidate. Thus we introduce the idea of a *confidence interval*. 

::: .definition
**Definition** (Confidence interval)  
Consider an unknown probability distribution $\nu$ from which we get $n$ independent samples $X\_1, \ldots, X\_n$, and suppose that $\theta$ is the value of some statistical functional of $\nu$. A **confidence interval** for $\theta$ is an interval-valued function of the sample data $X\_1, \ldots, X\_n$. A confidence interval has **confidence level** $1-\alpha$ if it contains $\theta$ with probability at least $1-\alpha$. 
:::


::: .exercise
**Exercise**  
Use Chebyshev's inequality to show that if $\widehat{\theta}$ is unbiased, then $(\widehat{\theta} - k \operatorname{se}(\widehat{\theta}), \widehat{\theta} + k \operatorname{se}(\widehat{\theta}))$ is a $1 - \frac{1}{k^2}$ confidence interval. 
:::

If $\widehat{\theta}$ is approximately normally distributed, then we can give tighter confidence intervals using the normal approximation: 

::: .exercise
**Exercise**  
Show that if $\widehat{\theta}$ is approximately normally distributed, then $(\widehat{\theta} - k \operatorname{se}(\widehat{\theta}), \widehat{\theta} + k \operatorname{se}(\widehat{\theta}))$ is a $1 - 2(1 - \Phi(k))$ confidence interval, where $\Phi$ is the CDF of the standard normal distribution. 
:::

If we are estimating a *function*-valued feature of $\nu$ rather than a single number (for example, a regression function), then we might want to provide a confidence *band* which traps the whole graph of the function with specified probability (see the DKW theorem for an example). 

::: .definition
**Definition** (Confidence band)  
Let $I \subset \mathbb{R}$, and suppose that $T$ is a function from the set of distributions to the set of real-valued functions on $I$. A $1-\alpha$ **confidence band** for $T(\nu)$ is pair of random functions $y\_{\textrm{min}}$ and $y\_{\textrm{max}}$ from $I$ to $\mathbb{R}$ defined in terms of $n$ independent samples from $\nu$ and having $y\_{\textrm{min}} \leq T(\nu) \leq y\_{\textrm{max}}$ everywhere on $I$ with probability at least $1-\alpha$. 
:::

---
> id: empirical-cdf-convergence
## Empirical CDF Convergence


Let's revisit the observation from the first section that the CDF of the empirical distribution of an independent list of samples from a distribution tends to be close to the CDF of the distribution itself. The **Glivenko-Cantelli** theorem is a mathematical formulation of the idea that these two functions are indeed close.

::: .theorem
**Theorem** (Glivenko-Cantelli)  
  If $F$ is the CDF of a distribution $\nu$ and $\widehat{F}_n$ is the
  CDF of the empirical distribution $\widehat{\nu}_n$ of $n$ samples
  from $\nu$, then $F_n$ converges to $F$ along the whole number line:
  ``` latex
  \max_{x\in \mathbb{R}} |F(x) - \widehat{F}_n(x)| \to 0 \quad \text{as }n \to \infty, 
  ```
:::

[Continue](btn:next)

---
> id: step-DKW

The **Dvoretzky-Kiefer-Wolfowitz inequality** quantifies this result by providing a confidence band.  

::: .theorem

    img(src="images/uniformcdf.svg" width=240 style="float: right;")

**Theorem**  (Dvoretzky-Kiefer-Wolfowitz inequality)  
If $X_1, X_2, \ldots$ are independent random variables with common CDF $F$, then for all $\epsilon > 0$, we have 
  
``` latex
\mathbb{P}\left(\max_{x}|F(x) - \widehat{F}_n(x)|\geq \epsilon\right) \leq 2
    \operatorname{e}^{-2n\epsilon^2}. 
```

In other words, the probability that the graph of $\widehat{F}_n$ lies in the $\epsilon$-band around $F$ (or vice versa) is at least $1 - 2 \operatorname{e}^{-2n\epsilon^2}$.
:::


::: .exercise
**Exercise**  
  Show that if $\epsilon_n = \sqrt{\frac{1}{2n}\log(\frac{2}{\alpha})}$, then with probability at least $1 - \alpha$, we have
  $|F(x) - F_n(x)| \leq \epsilon$ for all $x \in \mathbb{R}$.
:::

---
> id: Bootstrapping
## Bootstrapping

**Bootstrapping** is the use of simulation to approximate the value of the plug-in estimator of a statistical functional $T$ which is expressed in terms of independent samples from the input distribution $\nu$. The key idea is that drawing $k$ samples from $\widehat{\nu}$ is the same as drawing $k$ times with replacement from the list of samples. 

::: .example
**Example**  
Consider the statistical functional $T(\nu) = $ the expected difference between the greatest and least of 10 independent samples from $\nu$. Suppose that 50 samples $X_1, \ldots , X_{50}$ from $\nu$ are observed, and that $\widehat{\nu}$ is the associated empirical CDF. Explain how $T(\widehat{\nu})$ may be estimated with arbitrarily small error.
:::

*Solution*. The value of $T(\widehat{\nu})$ is defined to be the expectation of a distribution that we have instructions for how to sample from. So we sample 10 times with replacement from $X_1, \ldots , X_{50}$, identify the largest and smallest of the 10 samples, and record the difference. We repeat $B$ times for some large integer $B$, and we return the sample mean of these $B$ values.

By the law of large numbers, the result can be made arbitrarily close to $T(\widehat{\nu})$ with arbitrarily high probability by choosing $B$ sufficiently large. 

Although this example might seem a bit contrived, bootstrapping is useful in practice because of a common source of statistical functionals that fit the bootstrap form: \textit{standard errors}.

::: .example
**Example**  
Suppose that we estimate the median $\theta$ of a distribution using the plug-in estimator $\widehat{\theta}$ for 75 observed samples, and we want to produce a confidence interval for $\theta$. Show how to use bootstrapping to estimate the standard error of the estimator.
:::

*Solution*. By definition, the standard error of $\widehat{\theta}$ is the square root of the variance of the median of 75 independent draws from $\nu$. Therefore, the plug-in estimator of the standard error is the square root of the variance of the median of 75 independent draws from $\widehat{\nu}$. This can be readily simulated. If the samples are stored in a vector `{jl} X`, then 

    pre(julia-executable)
      | using Random
      | X = rand(75)
      | std(median(sample(X, 75)) for _ in 1:10^5)
      
    pre.rblock(r-executable)
      | sd(sapply(1:10^5,function(n) {median(sample(X,75,replace=TRUE))}))
      
returns a very accurate approximation of $T(\widehat{\nu})$. 

::: .exercise
**Exercise**  
Suppose that $\nu$ is the uniform distribution on $[0,1]$. Generate 75 samples from $\nu$, store them in a vector $X$, and compute the bootstrap estimate of $T(\widehat{\nu})$. Use Monte Carlo simulation to directly estimate $T(\nu)$. Can the gap between your approximations of $T(\widehat{\nu})$ and $T(\nu)$ be made arbitrarily small by using more bootstrap samples?
:::

    pre(julia-executable)
      | 

    x-quill

[Continue](btn:next)

---
> id: maximum-likelihood-estimation
## Maximum Likelihood Estimation

So far we've only had one idea for building an estimator for a statistical functional $T$, which is to plug $\widehat{\nu}$ into $T$. In this section, we'll learn another approach which is quite general and has some compelling properties. 

Consider a parametric family $\\{f\_{\boldsymbol{\theta}}(x)\,:\, \boldsymbol{\theta} \in \mathbb{R}^d\\}$ of PDFs or PMFs. Given $\mathbf{x} \in \mathbb{R}^n$, the **likelihood** $\mathcal{L}\_{\mathbf{x}}: \mathbb{R}^d \to \mathbb{R}$ is defined by 

``` latex
\mathcal{L}_{\mathbf{x}}(\boldsymbol{\theta}) = f_{\boldsymbol{\theta}}(x_{1})f_{\boldsymbol{\theta}}(x_{2})\cdots
f_{\boldsymbol{\theta}}(x_{n}).
```

The idea is that if $\mathbf{X}$ is a vector of $n$ independent samples drawn from $f\_{\boldsymbol{\theta}}(x)$, then $\mathcal{L}\_{\mathbf{X}}(\boldsymbol{\theta})$ is small or zero when $\boldsymbol{\theta}$ is not in concert with the observed data. 

::: .example
**Example**  
Suppose $x\mapsto f(x;\theta)$ is the density of a uniform random variable on $[0,\theta]$. We observe four samples drawn from this distribution: $1.41, 2.45, 6.12$, and $4.9$. Find $\mathcal{L}(5)$, $\mathcal{L}(10^6)$, and $\mathcal{L}(7)$. 
:::

*Solution*. The likelihood at 5 is zero, since $f\_{5}(x\_{3}) = 0$. The likelihood at $10^6$ is very small, since $\mathcal{L}(10^6) = (1/10^6)^4 = 10^{-24}$. The likelihood at 7 is larger: $(1/7)^4 = 1/2401$. 

We can see from this example that likelihood has the property of being zero or small at implausible values of $\boldsymbol{\theta}$, and larger at more reasonable values. Thus we propose the **maximum likelihood estimator** 

``` latex
\widehat{\boldsymbol{\theta}}_{\mathrm{MLE}} = \operatorname{argmax}_{\boldsymbol{\theta} \in
\mathbb{R}^d}\mathcal{L}_{\mathbf{X}}(\boldsymbol{\theta}).
```
 
::: .example
**Example**  
Suppose that $x\mapsto f(x;\mu,\sigma^2)$ is the normal density with mean $\mu$ and variance $\sigma^2$. Find the maximum likelihood estimator for $\mu$ and $\sigma^2$. 
:::


*Solution*. The maximum likelihood estimator is the minimizer of the logarithm of the likelihood function, which is 

``` latex
-\frac{n}{2}\log 2\pi - n \log \sigma - \frac{n}{2}\log 2\pi - n
\log \sigma - \frac{(X_1-\mu)^2}{2\sigma^2} - \cdots - \frac{(X_n
  - \mu)^2}{2\sigma^2}
```

Setting the derivatives with respect to $\mu$ and $\sigma^2$ equal to zero, we find $\mu = \overline{X} = \frac{1}{n}(X\_1+\cdots+X\_n)$ and $\sigma^2 = \frac{1}{n}((X\_1-\overline{X})^2 + \cdots + (X\_n-\overline{X})^2)$. So the maximum likelihood estimator agrees with the plug-in estimator for $\mu$ and $\sigma^2$. 

MLE enjoys several nice properties: under certain regularity conditions, we have  
* **Consistency**: $\mathbb{E}[(\widehat{\theta}\_{\mathrm{MLE}} - \theta)^2] \to 0$ as the number of samples goes to $\infty$. 
* **Asymptotic normality**: $(\widehat{\theta}\_{\mathrm{MLE}} - \theta)/\sqrt{\operatorname{Var} \widehat{\theta}\_{\mathrm{MLE}}}$ converges to $\mathcal{N}(0,1)$ as the number of samples goes to $\infty$. 
* **Asymptotic optimality**: the MSE of the MLE converges to 0 approximately as fast as the MSE of any other consistent estimator. 

 
Potential difficulties with MLE:  
* **Computational difficulties**. It might be difficult to work out where the maximum of the likelihood occurs, either analytically or numerically. 
* **Misspecification**. The MLE may be inaccurate if the distribution of the samples is not in the specified parametric family. 
* **Unbounded likelihood**. If the likelihood function is not bounded, then $\widehat{\theta}\_{\mathrm{MLE}}$ is not well-defined. 

 
---
> id: hypothesis-testing
## Hypothesis Testing

**Hypothesis testing** is a disciplined framework for adjudicating whether observed data do not support a given hypothesis. 

 Consider an unknown distribution from which we will observe $n$ samples $X\_1, \ldots X\_n$. 
* We state a hypothesis $H\_0$—called the **null hypothesis**—about the distribution. 
* We come up with a **test statistic** $T$, which is a function of the data $X\_1, \ldots X\_n$, for which we can evaluate the distribution of $T$ assuming the null hypothesis. 
* We give an **alternative hypothesis** $H\_{\mathrm{a}}$ under which $T$ is expected to be significantly different from its value under $H\_0$. 
* We give a significance level $\alpha$(like 5% or 1%), and based on $H\_{\mathrm{a}}$ we determine a set of values for $T$—called the *critical region*—which $T$ would be in with probability at most $\alpha$ under the null hypothesis. 
* **After setting $\boldsymbol{H_0}$, $\boldsymbol{H_{\mathrm{a}}}$, $\boldsymbol{\alpha}$, $\boldsymbol{T}$, and the critical region**, we run the experiment, evaluate $T$ on the samples we get, and record the result as $t\_{\mathrm{obs}}$. 
* If $t\_{\mathrm{obs}}$ falls in the critical region, we reject the null hypothesis. The corresponding **_p_-value** is defined to be the minimum $\alpha$-value which would have resulted in rejecting the null hypothesis, with the critical region chosen in the same way*. 

::: .example
**Example**  
Muriel Bristol claims that she can tell by taste whether the tea or the milk was poured into the cup first. She is given eight cups of tea, four poured milk-first and four poured tea-first. 

We posit a null hypothesis that she isn't able to discern the pouring method, and an alternative hypothesis that she can tell the difference. How many cups does she have to identify correctly to reject the null hypothesis with 95% confidence? 
:::

*Solution*. Under the null hypothesis, the number of cups identified correctly is 4 with probability $1/\binom{8}{4} \approx 1.4\%$ and at least 3 with probability $17/70 \approx 24\%$. Therefore, at the 5% significance level, only a correct identification of all the cups would give us grounds to reject the null hypothesis. The $p$-value in that case would be 1.4%. 

 Failure to reject the null hypothesis is not necessarily evidence *for* the null hypothesis. The **power** of a hypothesis test is the conditional probability of rejecting the null hypothesis given that the alternative hypothesis is true. A $p$-value may be low either because the null hypothesis is true or because the test has low power. 

::: .definition
**Definition**  
The **Wald test** is based on the normal approximation. Consider a null hypothesis $\theta = 0$ and the alternative hypothesis $\theta \neq 0$, and suppose that $\widehat{\theta}$ is approximately normally distributed. The Wald test rejects the null hypothesis at the 5% significance level if $|\widehat{\theta}| &gt; 1.96 \operatorname{se}(\widehat{\theta})$. 
:::

::: .example
**Example**  
Consider the alternative hypothesis that 8-cylinder engines have lower fuel economy than 6-cylinder engines (with null hypothesis that they are the same). Apply the Wald test, using the `{r} mtcars` dataset available in R. 
:::

 

*Solution*. We frame the problem as a question about whether the *difference in means* between the distribution of 8-cylinder `{r} mpg` values and the distribution of 6-cylinder `{r} mpg` values is zero. We use the difference between the sample means $\overline{X}$ and $\overline{Y}$ of the two populations as an estimator of the difference in means. If we think of the records in the data frame as independent, then $\overline{X}$ and $\overline{Y}$ are independent. Since each is approximately normally distributed by the central limit theorem, their difference is therefore also approximately normal. So, let's calculate the sample mean and sample variance for the 8-cylinder cars and for the 6-cylinder cars. 

    pre.rblock(r-executable)
      | 
      | library(tidyverse)
      | 
      | stats <- mtcars %>% 
      |   group_by(cyl) %>% 
      |   filter(cyl %in% c(6,8)) %>% 
      |   summarise(m = mean(mpg), S2 = var(mpg), n = n(), se = sqrt(S2/n))
      
Given that the distribution of 8-cylinder `{r} mpg` values has variance $\sigma\_{\mathrm{eight}}^2$, the variance of the sample mean $\overline{X}$ is $\sigma\_{\mathrm{eight}}^2/n\_{\mathrm{eight}}$, where $n\_{\mathrm{eight}}$ is the number of 8-cylinder vehicles (and similarly for $\overline{Y}$). Therefore, we estimate the variance of the difference in sample means as 

``` latex
 \operatorname{Var}(\overline{X} - \overline{Y}) = \operatorname{Var}(\overline{X}) +
\operatorname{Var}(\overline{Y}) =\sigma_{\mathrm{eight}}^2/n_{\mathrm{eight}} +
  \sigma_{\mathrm{six}}^2/n_{\mathrm{six}}. 
```

Under the null hypothesis, therefore, $\overline{X} - \overline{Y}$ has mean zero and standard error $\sqrt{\sigma\_{\mathrm{eight}}^2/n\_{\mathrm{eight}} +
\sigma\_{\mathrm{six}}^2/n\_{\mathrm{six}}}$. We therefore reject the null hypothesis with 95% confidence if the value of $\overline{X} - \overline{Y}$ divided by its estimated standard error exceeds 1.96. We find that 

    pre.rblock(r-executable)
      | z <- (stats$m[1] - stats$m[2]) / sqrt(sum(stats$se^2))

returns $5.29$, so we do reject the null hypothesis at the 95% confidence level. The $p$-value of this test is `{r} 1-pnorm(z)` $= 6.08 \times 10^{-6}$. 

[Continue](btn:next)

---
> id: random-permutation-test
### Random Permutation Test

The following test is more flexible than the Wald test, since it doesn't rely on the normal approximation. It's based on a simple idea: if there's no difference in labels, the data shouldn't look very different if we shuffle them around. 

::: .definition
**Definition**  
The **random permutation test** is applicable when the null hypothesis is that two distributions are the same. 
* We compute the difference between the sample means for the two groups. 
* We randomly re-assign the group labels and compute the resulting sample mean differences. Repeat many times. 
* We check where the original difference falls in the sorted list of re-sampled differences. 
:::

::: .example
**Example**  
Suppose the heights of the Romero sons are 72, 69, 68, and 66 inches, and the heights of the Larsen sons are 70, 65, and 64 inches. Consider the null hypothesis that the height distributions for the two families are the same, with the alternative hypothesis that they are not. Determine whether a random permutation test applied to the absolute sample mean difference rejects the null hypothesis at significance level $\alpha = 5\%$. 
:::


*Solution*. We find that the absolute sample mean difference of about 2.4 inches is larger than only about 68% of the mean differences obtained by resampling many times. 

    pre.rblock(r-executable)
      | 
      | set.seed(123)
      | romero <- c(72, 69, 68, 66)
      | larsen <- c(70, 65, 64)
      | actual.diff <- abs(mean(romero) - mean(larsen))
      | 
      | resample.diff <- function(n) {
      |   shuffled <- sample(c(romero,larsen))
      |   abs(mean(shuffled[1:4]) - mean(shuffled[5:7]))
      | }
      | 
      | sum(sapply(1:10000,resample.diff) < actual.diff)
      |    

Since 68% < 95%, we retain the null hypothesis. 

---
> id: multiple-testing
### Multiple testing

If we conduct many hypothesis tests, then the probability of obtaining some false rejections is high. This is called the **multiple testing problem**. 

    figure
      img(src="https://imgs.xkcd.com/comics/significant.png")
      p.caption Credit: xkcd.com

The **Bonferroni method** is to reject the null hypothesis only for those tests whose $p$-values are less than $\alpha$ divided by the number of hypothesis tests being run. This ensures that the probability of having even one false rejection is less than $\alpha$, so it is very conservative. 

::: .example
**Example**  
Suppose that 10 different genes are tested to determine whether they have an affect on heart disease. The 10 $p$-values resulting from these hypothesis tests are (rounded to the nearest hundredth of a percent): 

``` latex
0.89\%,   
2.71\%,             
9.11\%,           
2.18\%,             
9.17\%,             
7.48\%,
5.0\%,              
2.02\%,             
5.22\%,
9.46\% 
```      
      
Which results are reported as significant at the 5% level, according to the Bonferroni method? 
::: 

*Solution*. At the 5% level, only $p$ values less than 5%/10 = 0.5% are reported as significant (since we ran ten hypothesis tests). Since none of the $p$ values are below 0.5%, none of the genes will be considered significant. 