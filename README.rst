.. -*- mode: rst -*-

MEmPaMaL
========

Means for EMbarrassingly PArallel MAchine Learning.  

*Même pas mal!* is a french exclamation standing for *Did not hurt!*.
So, if your computer cannot manage your machine learning load, just
respond MEmPaMaL and give it a try.

MEmPaMaL is a Python module for some machine learning workflows built on top of
Scikit-learn and distributed under the 3-Clause BSD license.

Purpose
-------

MEmPaMaL is a set of python helpers to produce and run some 
embarrassingly parallel machine learning workflows. The goal of
MEmPaMaL is to produce a list of commands and dependencies. We state
that today, commands and files are a very portable and effective
approach for quick and dirty data exploration or new algorithms
development. *If your workflow runs on your personal computer but takes
too many times, MEmPaMal can help you to scale out!*

For the machine learning description it relies on the Scikit-learn [1]
design (pipelines, estimators with fit/predict, transformers with
fit/transform, etc.) and can accept estimators that respect the
Scikit-learn conventions. One possible back-end to execute the list of
commands and dependencies is Soma-Workflow [2]. This approach was
successfully applied to personal multicore computers, clusters (with
more than 2,000 cores) and cloud. To see more on the origins and
motivations of this project, you can read [3].

So, what is this set of machine learning workflows and the restrictions?

Restrictions 
------------ 

- **Suited for supervised learning**. For a given set of samples we
  measure features, an array denoted X of shape (n_samples,
  n_features), and targets, an array y of shape (n_samples,
  n_targets). The goal is to construct a predictive model and to
  measure its goodness of fit on unseen data (see examples for more
  insight).

- **Did you say Big Data?** MEmPaMaL don't care about this marketing
  term, it just some scripts to help you distribute your computation
  as seamlessly as possible. If your code and computing infrastructure
  is able to perform *Big Data* analysis with simple files and
  commands, it should work but it is NOT the purpose of MEmPaMaL.

- **Need your knowledge**. You need do understand what you do. For
  instance, most of the times multi-targets are inappropriate for a
  given estimator. Knowing about your data is probably more important
  than being able to compute many models.

- **It's not magic!** You need the computing resources corresponding
  to your load then MEmPaMaL can help you to scale out.

- **It's really not magic!** If your algorithm is not efficient or
  consume too many memory, MEmPaMaL will just help you to run it many
  times in parallel. So, be careful of the snowball effect and if not
  sure, try it at small scale first.

- **It's definitely not magic!** Some codes are already parallel
  (multiprocessing, OpenMP, etc.) and you need to understand the
  implications and make choices. For instance, you can limit OpenMP to
  one thread: ``export OMP_NUM_THREADS=1``. In some cases, you would
  like to play with the limits of your system and for that you must
  deeply understand the underlying technologies and the possibilities
  of the execution back-end.

Examples
--------

a simple example:
http://nbviewer.ipython.org/github/BenoitDamota/mempamal/blob/master/mempamal/examples/MEmPaMaL%20_%20first%20example.ipynb

References
----------

[1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR
12, pp. 2825-2830, 2011. (http://scikit-learn.org)

[2] Soma-workflow: a unified and simple interface to parallel
computing resources, S. Laguitton et al., in MICCAI Workshop on High
Performance and Distributed Computing for Medical Imaging,
2011. (http://neurospin.github.io/soma-workflow/)

[3] Machine learning patterns for neuroimaging-genetic studies in the cloud,
B. Da Mota et al., in Frontiers in neuroinformatics, 2014.
